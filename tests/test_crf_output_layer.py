# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""The CRF training objective: ``L_CRF = log Z(x) - score(y)``."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import FixedEmission, exact_log_partition, sequence_score
from fairseq.modules.dynamic_crf_output_layer import DynamicCRFOutputLayer
from fairseq.modules.output_layer import SharedEmbeddingOutputLayer

VOCAB, DIM, RANK = 4, 6, 3
BATCH, LENGTH = 2, 3


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


def build_crf(emission=None, vocab_size=VOCAB, feature_dim=DIM, **overrides):
    settings = dict(
        low_rank_dim=RANK,
        beam_size=VOCAB,
        dynamic_transition=True,
        dynamic_hidden_dim=8,
        inference='viterbi_marginal',
    )
    settings.update(overrides)
    if emission is None:
        emission = SharedEmbeddingOutputLayer(
            embed_tokens=nn.Embedding(vocab_size, feature_dim, padding_idx=1))
    return DynamicCRFOutputLayer(
        emission_layer=emission, vocab_size=vocab_size, feature_dim=feature_dim, **settings)


@pytest.fixture
def features():
    return torch.randn(BATCH, LENGTH, DIM)


@pytest.fixture
def target():
    return torch.randint(0, VOCAB, (BATCH, LENGTH))


# --- exactness against enumeration ------------------------------------------

@pytest.mark.parametrize('dynamic', [True, False])
def test_crf_nll_is_exact_when_the_beam_covers_the_vocabulary(features, target, dynamic):
    """With ``k >= V`` nothing is truncated, so the loss must equal the exact
    ``-log P(y|x)`` obtained by enumerating all ``V ** T`` sequences."""
    crf = build_crf(beam_size=VOCAB, dynamic_transition=dynamic)
    emissions = crf(features)

    nll = crf.crf_nll(features, target)
    log_partition = exact_log_partition(crf, features, emissions)
    for b in range(BATCH):
        gold = sequence_score(crf, features, emissions, b, target[b].tolist())
        torch.testing.assert_close(nll[b], log_partition[b] - gold, rtol=1e-4, atol=1e-5)


def test_crf_nll_has_one_value_per_sequence(features, target):
    crf = build_crf()
    assert crf.crf_nll(features, target).shape == (BATCH,)


def test_crf_nll_with_zero_transitions_is_plain_cross_entropy(features, target):
    """Zeroing the transitions removes the structure, so the CRF must collapse
    onto the independent per-token cross-entropy of the emission head."""
    crf = build_crf(beam_size=VOCAB, dynamic_transition=False)
    with torch.no_grad():
        crf.transition_source.weight.zero_()
        crf.transition_target.weight.zero_()

    emissions = crf(features)
    expected = F.cross_entropy(
        emissions.reshape(-1, VOCAB), target.reshape(-1), reduction='none'
    ).view(BATCH, LENGTH).sum(dim=-1)
    torch.testing.assert_close(crf.crf_nll(features, target), expected, rtol=1e-4, atol=1e-5)


# --- the gold-forcing guarantee ---------------------------------------------

@pytest.mark.parametrize('seed', range(12))
def test_crf_nll_is_never_negative(seed):
    """The reason the gold path is forced into the beam.

    ``log Z`` is approximated by a sum over the beam only. Forcing each gold
    token into its position's beam makes the gold path one of the summands, so
    the approximated ``Z`` can never fall below the gold score and the loss stays
    a genuine negative log-likelihood.
    """
    torch.manual_seed(seed)
    vocab, length, batch = 20, 6, 4
    crf = build_crf(vocab_size=vocab, beam_size=3)  # k far below V: heavy truncation
    features = torch.randn(batch, length, DIM)
    target = torch.randint(0, vocab, (batch, length))

    nll = crf.crf_nll(features, target)
    assert bool((nll >= -1e-5).all()), nll


def test_crf_nll_could_go_negative_without_gold_forcing():
    """Documents the failure mode that gold forcing exists to prevent.

    Emissions favour token 0 everywhere, so an unforced beam of one keeps only
    token 0; but the transition embeddings make the gold bigram (1, 1) score far
    higher than that path. The gold score then exceeds the truncated partition
    and a naive implementation reports a negative "likelihood".
    """
    vocab, length, rank = 4, 2, 2
    emissions = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]]])
    crf = DynamicCRFOutputLayer(
        emission_layer=FixedEmission(emissions),
        vocab_size=vocab, feature_dim=DIM, low_rank_dim=rank,
        beam_size=1, dynamic_transition=False,
    )
    with torch.no_grad():
        crf.transition_source.weight.zero_()
        crf.transition_target.weight.zero_()
        crf.transition_source.weight[1, 0] = 10.0
        crf.transition_target.weight[1, 0] = 10.0   # E1[1] . E2[1] = 100

    features = torch.zeros(1, length, DIM)
    target = torch.ones(1, length, dtype=torch.long)

    # Truncated to token 0: log Z = 10 + 10 + 0. Gold: 0 + 0 + 100.
    unforced = crf.log_partition(features, target=None) - crf.gold_score(features, target)
    torch.testing.assert_close(unforced, torch.tensor([-80.0]))
    assert bool((unforced < 0).all())

    # Forcing the gold token in puts the gold path inside the beam, so the loss
    # is non-negative -- here exactly zero, the beam holding only that path.
    torch.testing.assert_close(crf.crf_nll(features, target), torch.tensor([0.0]))


def test_gold_score_matches_the_manual_sum(features, target):
    crf = build_crf()
    emissions = crf(features)
    gold = crf.gold_score(features, target)
    for b in range(BATCH):
        expected = sequence_score(crf, features, emissions, b, target[b].tolist())
        torch.testing.assert_close(gold[b], expected, rtol=1e-4, atol=1e-5)


# --- padding ----------------------------------------------------------------

def test_crf_nll_ignores_padded_positions():
    """Appending padding must not change the loss of the real prefix."""
    crf = build_crf(beam_size=VOCAB)
    short_features = torch.randn(1, 3, DIM)
    short_target = torch.randint(0, VOCAB, (1, 3))

    padded_features = torch.cat([short_features, torch.randn(1, 2, DIM)], dim=1)
    padded_target = torch.cat([short_target, torch.ones(1, 2, dtype=torch.long)], dim=1)
    mask = torch.tensor([[True, True, True, False, False]])

    torch.testing.assert_close(
        crf.crf_nll(padded_features, padded_target, mask),
        crf.crf_nll(short_features, short_target),
        rtol=1e-4, atol=1e-5)


def test_crf_nll_keeps_batch_rows_independent_across_lengths():
    crf = build_crf(beam_size=VOCAB)
    features = torch.randn(3, 4, DIM)
    target = torch.randint(0, VOCAB, (3, 4))
    mask = torch.tensor([
        [True, True, True, True],
        [True, True, True, False],
        [True, True, False, False],
    ])

    batched = crf.crf_nll(features, target, mask)
    for b, length in enumerate([4, 3, 2]):
        alone = crf.crf_nll(features[b:b + 1, :length], target[b:b + 1, :length])
        torch.testing.assert_close(batched[b:b + 1], alone, rtol=1e-4, atol=1e-5)


# --- training mechanics -----------------------------------------------------

def test_crf_nll_gradients_reach_every_trainable_part():
    embed_tokens = nn.Embedding(VOCAB, DIM, padding_idx=1)
    crf = build_crf(emission=SharedEmbeddingOutputLayer(embed_tokens=embed_tokens))
    features = torch.randn(BATCH, LENGTH, DIM, requires_grad=True)
    target = torch.randint(0, VOCAB, (BATCH, LENGTH))

    crf.crf_nll(features, target).sum().backward()

    assert crf.transition_source.weight.grad.abs().sum() > 0
    assert crf.transition_target.weight.grad.abs().sum() > 0
    for name, param in crf.dynamic_transition.named_parameters():
        assert param.grad is not None, name
    # The emission head is trained through the CRF too, even though it is shared
    # with the decoder's input embeddings.
    assert embed_tokens.weight.grad is not None
    assert embed_tokens.weight.grad.abs().sum() > 0
    assert features.grad.abs().sum() > 0


def test_crf_nll_runs_in_float32_under_half_precision(features, target):
    """logsumexp chains are unreliable in fp16, so the CRF must upcast internally.

    Mirrors what ``--fp16`` does: weights and activations are halved together.
    The borrowed embedding has to be halved separately here because the head
    deliberately does not own it; in the real model ``model.half()`` reaches it
    through the decoder.
    """
    embed_tokens = nn.Embedding(VOCAB, DIM, padding_idx=1)
    crf = build_crf(emission=SharedEmbeddingOutputLayer(embed_tokens=embed_tokens))
    reference = crf.crf_nll(features, target)

    crf.half()
    embed_tokens.half()
    nll = crf.crf_nll(features.half(), target)
    assert nll.dtype == torch.float32
    torch.testing.assert_close(nll, reference, rtol=2e-2, atol=2e-2)


def test_crf_transitions_are_upcast_under_half_precision(features):
    crf = build_crf()
    crf.half()
    beam_index = torch.randint(0, VOCAB, (BATCH, LENGTH, 2))
    assert crf.build_transitions(features.half(), beam_index).dtype == torch.float32


def test_forward_still_returns_emission_logits(features):
    """The CRF must stay a drop-in head: ``forward`` is the emission projection,
    so existing consumers of ``decoder_out[0]`` keep working."""
    embed_tokens = nn.Embedding(VOCAB, DIM, padding_idx=1)
    crf = build_crf(emission=SharedEmbeddingOutputLayer(embed_tokens=embed_tokens))
    torch.testing.assert_close(crf(features), F.linear(features, embed_tokens.weight))


def test_crf_rejects_an_unknown_inference_mode():
    with pytest.raises(ValueError):
        build_crf(inference='beam_search')
