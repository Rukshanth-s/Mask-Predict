# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""Structured (CRF) decoder output layer.

Implements the output layer of "Fast Structured Decoding for Sequence Models"
(Sun et al., NeurIPS 2019, arXiv:1910.11555) -- a linear-chain CRF over adjacent
target tokens, made tractable by a low-rank factorisation of the transition
matrix and by truncating each position to its top-k candidates.
"""

import torch
import torch.nn as nn

# How the CRF turns its globally normalised path distribution into the
# (tokens, per-position confidence) pair that the decoding strategies need.
#
#   viterbi_marginal  Viterbi picks the tokens (the paper's decoding rule) and
#                     the forward-backward marginal of the chosen token is its
#                     confidence.
#   marginal          The per-position posterior marginal picks the token and is
#                     also its confidence.
#   viterbi_emission  Viterbi picks the tokens; the confidence is the plain
#                     emission softmax probability, skipping the backward pass.
CRF_INFERENCE_CHOICES = ['viterbi_marginal', 'marginal', 'viterbi_emission']


# ---------------------------------------------------------------------------
# Linear-chain dynamic programs.
#
# These operate on an already-truncated beam, so they cost O(T k^2) rather than
# O(T V^2). Throughout:
#
#   emissions    B x T x k        label score of each candidate at each position
#   transitions  B x (T-1) x k x k    transitions[b, i, a, c] scores candidate a
#                                     at position i followed by candidate c at
#                                     position i+1
#   mask         B x T            True at real tokens; padding is right-aligned
#                                 and every row has at least one real token
#
# A padded position contributes nothing: the recursions carry their state across
# it unchanged, so reading the final column is the same as reading the column at
# each row's own last real token.
# ---------------------------------------------------------------------------

def _default_mask(emissions, mask):
    if mask is not None:
        return mask.to(torch.bool)
    return emissions.new_ones(emissions.shape[:2], dtype=torch.bool)


def forward_algorithm(emissions, transitions, mask=None):
    """Sum over all paths in the beam.

    Returns:
        ``(alpha, log_partition)`` where ``alpha[b, i, c]`` is the log sum of the
        scores of all path prefixes ending in candidate ``c`` at position ``i``,
        and ``log_partition`` is ``log Z`` per sequence.
    """
    mask = _default_mask(emissions, mask)
    length = emissions.size(1)

    alpha = [emissions[:, 0]]
    for i in range(1, length):
        # B x k x k: previous state broadcast over the transition's target axis.
        scores = alpha[i - 1].unsqueeze(2) + transitions[:, i - 1]
        candidate = emissions[:, i] + torch.logsumexp(scores, dim=1)
        alpha.append(torch.where(mask[:, i].unsqueeze(1), candidate, alpha[i - 1]))

    alpha = torch.stack(alpha, dim=1)
    return alpha, torch.logsumexp(alpha[:, -1], dim=-1)


def backward_algorithm(emissions, transitions, mask=None):
    """Sum over all path suffixes; the mirror image of :func:`forward_algorithm`.

    Returns ``beta`` with ``beta[b, i, a]`` the log sum of the scores of all path
    suffixes *after* position ``i`` given candidate ``a`` there. The emission at
    ``i`` itself is excluded, so ``alpha + beta`` double-counts nothing.
    """
    mask = _default_mask(emissions, mask)
    length = emissions.size(1)

    beta = [None] * length
    beta[length - 1] = emissions.new_zeros(emissions.size(0), emissions.size(2))
    for i in range(length - 2, -1, -1):
        # B x k x k: the successor's emission and suffix broadcast over the
        # transition's source axis.
        scores = transitions[:, i] + (emissions[:, i + 1] + beta[i + 1]).unsqueeze(1)
        candidate = torch.logsumexp(scores, dim=2)
        # If position i+1 is padding there is no edge out of i, and beta[i+1] is
        # already zero, so the carry produces the correct "nothing follows".
        beta[i] = torch.where(mask[:, i + 1].unsqueeze(1), candidate, beta[i + 1])

    return torch.stack(beta, dim=1)


def log_marginals(emissions, transitions, mask=None):
    """Per-position posterior ``log p(candidate c at position i | x)``.

    Globally normalised: unlike an emission softmax this discounts a candidate
    that scores well on its own but cannot agree with its neighbours.
    """
    alpha, log_partition = forward_algorithm(emissions, transitions, mask)
    beta = backward_algorithm(emissions, transitions, mask)
    return alpha + beta - log_partition.unsqueeze(-1).unsqueeze(-1)


def viterbi_decode(emissions, transitions, mask=None):
    """Highest-scoring path through the beam.

    Returns:
        ``(path, score)`` where ``path[b, i]`` is a *beam slot* index (the caller
        maps it back to a vocabulary id) and ``score`` is that path's unnormalised
        score. Slots at padded positions repeat the last real position's slot.
    """
    mask = _default_mask(emissions, mask)
    length, beam = emissions.size(1), emissions.size(2)
    identity = torch.arange(beam, device=emissions.device).expand(emissions.size(0), beam)

    delta = emissions[:, 0]
    back_pointers = []
    for i in range(1, length):
        scores = delta.unsqueeze(2) + transitions[:, i - 1]
        best, pointer = scores.max(dim=1)
        candidate = emissions[:, i] + best
        keep = mask[:, i].unsqueeze(1)
        delta = torch.where(keep, candidate, delta)
        # Padded steps point at themselves so backtracking walks the tail
        # without disturbing the slot chosen at the last real position.
        back_pointers.append(torch.where(keep, pointer, identity))

    score, last = delta.max(dim=-1)

    path = [last]
    for pointer in reversed(back_pointers):
        path.append(pointer.gather(1, path[-1].unsqueeze(1)).squeeze(1))
    path.reverse()

    return torch.stack(path, dim=1), score


class DynamicCRFOutputLayer(nn.Module):
    """Structured output layer: emission scores plus a linear-chain CRF.

    Wraps a plain output head (which supplies the emission/label scores
    ``s(y_i, x, i)``) and adds transition scores between adjacent positions, so
    the tokens of a non-autoregressive decoder are scored jointly instead of
    independently.

    Two approximations from arXiv:1910.11555 keep this affordable at a 32k
    vocabulary. The transition matrix is factorised into rank-``low_rank_dim``
    embeddings so no ``V x V`` matrix is ever built, and each position is
    truncated to its ``beam_size`` best candidates so the dynamic programs cost
    ``O(T k^2)`` rather than ``O(T V^2)``.
    """

    def __init__(self, emission_layer, vocab_size, feature_dim, low_rank_dim=32,
                 beam_size=64, dynamic_transition=True, dynamic_hidden_dim=None,
                 inference='viterbi_marginal'):
        super().__init__()
        if inference not in CRF_INFERENCE_CHOICES:
            raise ValueError('unknown crf inference mode {!r}, expected one of {}'.format(
                inference, CRF_INFERENCE_CHOICES))

        self.emission_layer = emission_layer
        self.vocab_size = vocab_size
        self.low_rank_dim = low_rank_dim
        self.beam_size = beam_size
        self.inference = inference

        # E1 / E2 of M = E1 E2^T: indexed by the source and target token of an edge.
        self.transition_source = nn.Embedding(vocab_size, low_rank_dim)
        self.transition_target = nn.Embedding(vocab_size, low_rank_dim)

        if dynamic_transition:
            hidden = feature_dim if dynamic_hidden_dim is None else dynamic_hidden_dim
            # f: R^{2 d_model} -> R^{d_t x d_t}, the paper's two-layer FFN.
            self.dynamic_transition = nn.Sequential(
                nn.Linear(2 * feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, low_rank_dim * low_rank_dim),
            )
        else:
            self.dynamic_transition = None

    def forward(self, features):
        """Emission logits, so this head is a drop-in for the unstructured ones."""
        return self.emission_layer(features)

    def build_beam(self, emissions, target=None, mask=None):
        """Truncate each position to its best candidates.

        Args:
            emissions: ``B x T x V`` label scores.
            target: optional ``B x T`` gold tokens to force into the beam. During
                training this is what guarantees the gold path is one of the
                summands of the approximated partition, and hence that the
                resulting negative log-likelihood cannot go negative.
            mask: optional ``B x T``; forcing is skipped where it is False.

        Returns:
            ``(beam_scores, beam_index)``, both ``B x T x k`` with
            ``k = min(beam_size, V)``.
        """
        beam = min(self.beam_size, emissions.size(-1))
        beam_scores, beam_index = emissions.topk(beam, dim=-1)

        if target is not None:
            missing = (beam_index != target.unsqueeze(-1)).all(dim=-1)
            if mask is not None:
                missing = missing & mask.to(torch.bool)
            # Displace the weakest candidate, which topk leaves in the last slot.
            beam_index = beam_index.clone()
            beam_index[..., -1] = torch.where(missing, target, beam_index[..., -1])
            beam_scores = emissions.gather(-1, beam_index)

        return beam_scores, beam_index

    def _prepare(self, features, mask):
        """Emission scores in float32, plus a materialised mask.

        The dynamic programs accumulate long chains of logsumexp, which is not
        reliable in half precision, so every score the CRF reasons over is float32
        even when the surrounding model runs under ``--fp16``. Note that the
        submodules are applied in their *own* dtype and only their outputs are
        upcast -- upcasting the features first would mismatch half weights.
        """
        emissions = self.emission_layer(features).float()
        return emissions, _default_mask(emissions, mask)

    def gold_score(self, features, target, mask=None):
        """Unnormalised score of the reference sequence -- computed exactly.

        Only the partition function is approximated by the beam; the numerator
        needs no truncation, being a single path.
        """
        emissions, mask = self._prepare(features, mask)
        return self._gold_score(features, emissions, target, mask)

    def _gold_score(self, features, emissions, target, mask, dynamic=None):
        """:meth:`gold_score` on emissions the caller has already computed.

        ``dynamic`` optionally supplies pre-computed transition matrices.
        """
        emitted = emissions.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        score = (emitted * mask).sum(dim=-1)

        if target.size(1) > 1:
            source = self.transition_source(target[:, :-1]).float()
            successor = self.transition_target(target[:, 1:]).float()
            if self.dynamic_transition is None:
                edges = (source * successor).sum(dim=-1)
            else:
                if dynamic is None:
                    dynamic = self._dynamic_matrices(features)
                edges = torch.einsum('btd,btde,bte->bt', source, dynamic, successor)
            # An edge counts only when both of its endpoints are real tokens.
            edge_mask = mask[:, :-1] & mask[:, 1:]
            score = score + (edges * edge_mask).sum(dim=-1)

        return score

    def log_partition(self, features, target=None, mask=None):
        """Approximated ``log Z(x)``: the sum over all paths inside the beam.

        Pass ``target`` to force the gold tokens into the beam, which is required
        for training -- see :meth:`crf_nll`.
        """
        emissions, mask = self._prepare(features, mask)
        return self._log_partition(features, emissions, target, mask)

    def _log_partition(self, features, emissions, target, mask, dynamic=None):
        """:meth:`log_partition` on emissions the caller has already computed.

        ``dynamic`` optionally supplies pre-computed transition matrices.
        """
        beam_scores, beam_index = self.build_beam(emissions, target=target, mask=mask)
        transitions = self.build_transitions(features, beam_index, dynamic=dynamic)
        return forward_algorithm(beam_scores, transitions, mask)[1]

    def crf_nll(self, features, target, mask=None):
        """``-log P(y|x)`` per sequence, the CRF training loss.

        The gold tokens are forced into the beam so that the gold path is one of
        the summands of the approximated partition; without that the result could
        be negative and would not be a log-likelihood at all.
        """
        # The two terms share one emission matrix and one set of transition
        # matrices. The emission matrix is B x T x V in float32 -- at a 32k
        # vocabulary the largest tensor in the step -- so computing it once
        # rather than once per term is what keeps --max-tokens usable.
        emissions, mask = self._prepare(features, mask)
        dynamic = None if self.dynamic_transition is None else self._dynamic_matrices(features)
        return (self._log_partition(features, emissions, target, mask, dynamic)
                - self._gold_score(features, emissions, target, mask, dynamic))

    def decode(self, features, mask=None):
        """Turn the path distribution into tokens with per-position confidences.

        Which rule is used is set by ``--crf-inference``; see
        :data:`CRF_INFERENCE_CHOICES`.

        Returns:
            ``(tokens, confidence, emission_probs)``. ``tokens`` and
            ``confidence`` are ``B x T``; ``emission_probs`` is the unstructured
            ``B x T x V`` softmax, returned so callers that want a full
            distribution keep getting one. Padded positions are given confidence
            1.0 so that the iterative strategies never choose to re-mask them.
        """
        if self.inference not in CRF_INFERENCE_CHOICES:
            raise ValueError('unknown crf inference mode {!r}, expected one of {}'.format(
                self.inference, CRF_INFERENCE_CHOICES))

        emissions, mask = self._prepare(features, mask)
        # No gold to force here: at inference the beam is simply the top-k.
        beam_scores, beam_index = self.build_beam(emissions)
        transitions = self.build_transitions(features, beam_index)
        emission_probs = torch.softmax(emissions, dim=-1)

        if self.inference == 'marginal':
            best = log_marginals(beam_scores, transitions, mask).max(dim=-1)
            slots, confidence = best.indices, best.values.exp()
        else:
            slots, _ = viterbi_decode(beam_scores, transitions, mask)
            if self.inference == 'viterbi_marginal':
                confidence = log_marginals(beam_scores, transitions, mask).gather(
                    -1, slots.unsqueeze(-1)).squeeze(-1).exp()
            else:  # viterbi_emission
                confidence = None  # filled in below from the emission softmax

        tokens = beam_index.gather(-1, slots.unsqueeze(-1)).squeeze(-1)
        if confidence is None:
            confidence = emission_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

        return tokens, confidence.masked_fill(~mask, 1.0), emission_probs

    def _dynamic_matrices(self, features):
        """``A^i = f([h_(i-1), h_i])``, one ``d_t x d_t`` matrix per edge.

        Float32, like every other score the CRF reasons over. As in
        :meth:`_prepare` the FFN itself runs in its own dtype and only its output
        is upcast, so half weights are not mismatched.
        """
        pair = torch.cat([features[:, :-1], features[:, 1:]], dim=-1)
        return self.dynamic_transition(pair).float().view(
            pair.size(0), pair.size(1), self.low_rank_dim, self.low_rank_dim)

    def build_transitions(self, features, beam_index, dynamic=None):
        """Transition scores restricted to the beam's candidate pairs.

        Returns ``B x (T-1) x k x k`` where entry ``[b, i, a, c]`` scores
        candidate ``a`` at position ``i`` followed by candidate ``c`` at ``i+1``.
        Always float32, for the same reason as :meth:`_prepare`.

        ``dynamic`` optionally supplies the ``A^i`` matrices from
        :meth:`_dynamic_matrices`, so a caller that also needs them elsewhere can
        compute the FFN once; by default they are computed here.
        """
        source = self.transition_source(beam_index[:, :-1]).float()  # B x (T-1) x k x d_t
        target = self.transition_target(beam_index[:, 1:]).float()   # B x (T-1) x k x d_t

        if self.dynamic_transition is None:
            # M = E1 E2^T
            scores = torch.einsum('btad,btcd->btac', source, target)
        else:
            # M^i = E1 A^i E2^T
            if dynamic is None:
                dynamic = self._dynamic_matrices(features)
            scores = torch.einsum('btad,btde,btce->btac', source, dynamic, target)
        return scores
