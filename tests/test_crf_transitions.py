# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""The CRF's transition scores (arXiv:1910.11555 Eq. 7-10) and beam truncation."""

import pytest
import torch
import torch.nn as nn

from fairseq.modules.dynamic_crf_output_layer import DynamicCRFOutputLayer
from fairseq.modules.output_layer import SharedEmbeddingOutputLayer

VOCAB, DIM, RANK, BEAM = 13, 6, 4, 3
BATCH, LENGTH = 2, 5


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


def build_crf(**overrides):
    settings = dict(
        vocab_size=VOCAB,
        feature_dim=DIM,
        low_rank_dim=RANK,
        beam_size=BEAM,
        dynamic_transition=True,
        dynamic_hidden_dim=8,
        inference='viterbi_marginal',
    )
    settings.update(overrides)
    embed_tokens = nn.Embedding(VOCAB, DIM, padding_idx=1)
    emission = SharedEmbeddingOutputLayer(embed_tokens=embed_tokens)
    return DynamicCRFOutputLayer(emission_layer=emission, **settings)


@pytest.fixture
def features():
    return torch.randn(BATCH, LENGTH, DIM)


# --- low-rank transitions ---------------------------------------------------

def test_transition_embeddings_have_the_low_rank_shape():
    crf = build_crf()
    assert crf.transition_source.weight.shape == (VOCAB, RANK)
    assert crf.transition_target.weight.shape == (VOCAB, RANK)


def test_static_transitions_match_the_naive_bilinear_form(features):
    """``M = E1 E2^T`` evaluated only at the beam's candidate pairs (Eq. 7)."""
    crf = build_crf(dynamic_transition=False)
    beam_index = torch.randint(0, VOCAB, (BATCH, LENGTH, BEAM))

    transitions = crf.build_transitions(features, beam_index)
    assert transitions.shape == (BATCH, LENGTH - 1, BEAM, BEAM)

    e1, e2 = crf.transition_source.weight, crf.transition_target.weight
    for b in range(BATCH):
        for i in range(LENGTH - 1):
            for a in range(BEAM):
                for c in range(BEAM):
                    expected = torch.dot(e1[beam_index[b, i, a]], e2[beam_index[b, i + 1, c]])
                    torch.testing.assert_close(transitions[b, i, a, c], expected)


def test_dynamic_transitions_match_the_naive_bilinear_form(features):
    """``M^i = E1 f([h_(i-1), h_i]) E2^T`` (Eq. 8-10)."""
    crf = build_crf(dynamic_transition=True)
    beam_index = torch.randint(0, VOCAB, (BATCH, LENGTH, BEAM))

    transitions = crf.build_transitions(features, beam_index)
    assert transitions.shape == (BATCH, LENGTH - 1, BEAM, BEAM)

    e1, e2 = crf.transition_source.weight, crf.transition_target.weight
    for b in range(BATCH):
        for i in range(LENGTH - 1):
            pair = torch.cat([features[b, i], features[b, i + 1]], dim=-1)
            dynamic = crf.dynamic_transition(pair).view(RANK, RANK)
            for a in range(BEAM):
                for c in range(BEAM):
                    u = e1[beam_index[b, i, a]]
                    v = e2[beam_index[b, i + 1, c]]
                    expected = u @ dynamic @ v
                    torch.testing.assert_close(transitions[b, i, a, c], expected)


def test_dynamic_matrix_is_square_in_the_low_rank_dimension(features):
    crf = build_crf(dynamic_transition=True)
    pair = torch.cat([features[:, :-1], features[:, 1:]], dim=-1)
    produced = crf.dynamic_transition(pair)
    assert produced.shape == (BATCH, LENGTH - 1, RANK * RANK)


def test_dynamic_transition_is_a_two_layer_ffn():
    crf = build_crf(dynamic_transition=True, dynamic_hidden_dim=8)
    linears = [m for m in crf.dynamic_transition.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 2
    assert linears[0].in_features == 2 * DIM
    assert linears[0].out_features == 8
    assert linears[1].out_features == RANK * RANK


def test_dynamic_with_an_identity_matrix_reproduces_the_static_transitions(features):
    """The dynamic form generalises the static one: forcing ``f`` to emit the
    identity must recover ``E1 E2^T`` exactly."""
    dynamic = build_crf(dynamic_transition=True)
    static = build_crf(dynamic_transition=False)
    static.transition_source.weight.data.copy_(dynamic.transition_source.weight.data)
    static.transition_target.weight.data.copy_(dynamic.transition_target.weight.data)

    final = [m for m in dynamic.dynamic_transition.modules() if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(torch.eye(RANK).reshape(-1))

    beam_index = torch.randint(0, VOCAB, (BATCH, LENGTH, BEAM))
    torch.testing.assert_close(
        dynamic.build_transitions(features, beam_index),
        static.build_transitions(features, beam_index))


def test_static_mode_has_no_dynamic_ffn():
    crf = build_crf(dynamic_transition=False)
    assert crf.dynamic_transition is None
    assert not any('dynamic_transition' in name for name in crf.state_dict())


def test_transitions_are_differentiable(features):
    crf = build_crf(dynamic_transition=True)
    beam_index = torch.randint(0, VOCAB, (BATCH, LENGTH, BEAM))
    crf.build_transitions(features, beam_index).sum().backward()
    assert crf.transition_source.weight.grad.abs().sum() > 0
    assert crf.transition_target.weight.grad.abs().sum() > 0


def test_single_position_yields_no_transitions():
    crf = build_crf()
    features = torch.randn(BATCH, 1, DIM)
    beam_index = torch.randint(0, VOCAB, (BATCH, 1, BEAM))
    assert crf.build_transitions(features, beam_index).shape == (BATCH, 0, BEAM, BEAM)


# --- beam construction ------------------------------------------------------

def test_beam_without_forcing_is_plain_topk():
    crf = build_crf()
    emissions = torch.randn(BATCH, LENGTH, VOCAB)
    scores, index = crf.build_beam(emissions)
    expected_scores, expected_index = emissions.topk(BEAM, dim=-1)
    torch.testing.assert_close(scores, expected_scores)
    assert torch.equal(index, expected_index)


def test_beam_scores_are_the_emissions_at_the_beam_indices():
    crf = build_crf()
    emissions = torch.randn(BATCH, LENGTH, VOCAB)
    scores, index = crf.build_beam(emissions)
    torch.testing.assert_close(scores, emissions.gather(-1, index))


def test_beam_is_clamped_to_the_vocabulary_size():
    crf = build_crf(beam_size=VOCAB + 50)
    emissions = torch.randn(BATCH, LENGTH, VOCAB)
    scores, index = crf.build_beam(emissions)
    assert scores.shape == (BATCH, LENGTH, VOCAB)
    assert index.shape == (BATCH, LENGTH, VOCAB)


def test_forcing_puts_every_gold_token_in_the_beam():
    """The guarantee that keeps the approximated log Z above the gold score."""
    crf = build_crf()
    emissions = torch.randn(BATCH, LENGTH, VOCAB)
    target = torch.randint(0, VOCAB, (BATCH, LENGTH))

    scores, index = crf.build_beam(emissions, target=target)
    assert index.shape == (BATCH, LENGTH, BEAM)
    present = (index == target.unsqueeze(-1)).any(dim=-1)
    assert bool(present.all())
    # And the forced entry still carries its true emission score.
    torch.testing.assert_close(scores, emissions.gather(-1, index))


def test_forcing_keeps_the_beam_free_of_duplicates():
    crf = build_crf()
    emissions = torch.randn(BATCH, LENGTH, VOCAB)
    target = torch.randint(0, VOCAB, (BATCH, LENGTH))
    _, index = crf.build_beam(emissions, target=target)
    for b in range(BATCH):
        for i in range(LENGTH):
            row = index[b, i].tolist()
            assert len(set(row)) == len(row)


def test_forcing_leaves_the_beam_alone_when_gold_is_already_in_it():
    crf = build_crf()
    emissions = torch.randn(BATCH, LENGTH, VOCAB)
    _, plain = crf.build_beam(emissions)
    # Use the current best candidate as the target: it is already in the beam.
    target = plain[:, :, 0]
    _, forced = crf.build_beam(emissions, target=target)
    assert torch.equal(plain, forced)


def test_forcing_displaces_only_the_weakest_candidate():
    crf = build_crf()
    emissions = torch.randn(BATCH, LENGTH, VOCAB)
    _, plain = crf.build_beam(emissions)
    # Pick a target guaranteed to be outside the beam: the lowest-scoring token.
    target = emissions.argmin(dim=-1)
    _, forced = crf.build_beam(emissions, target=target)
    # The strongest BEAM-1 candidates survive; only the weakest slot is replaced.
    assert torch.equal(forced[:, :, :BEAM - 1], plain[:, :, :BEAM - 1])
    assert torch.equal(forced[:, :, BEAM - 1], target)


def test_forcing_is_skipped_at_padded_positions():
    """Padding carries no gold token worth spending a beam slot on."""
    crf = build_crf()
    emissions = torch.randn(1, 4, VOCAB)
    target = emissions.argmin(dim=-1)
    mask = torch.tensor([[True, True, False, False]])

    _, plain = crf.build_beam(emissions)
    _, forced = crf.build_beam(emissions, target=target, mask=mask)
    assert torch.equal(forced[:, 2:], plain[:, 2:])
    assert bool((forced[:, :2] == target[:, :2].unsqueeze(-1)).any(dim=-1).all())


def test_beam_of_one_collapses_onto_the_forced_target():
    crf = build_crf(beam_size=1)
    emissions = torch.randn(BATCH, LENGTH, VOCAB)
    target = emissions.argmin(dim=-1)
    _, index = crf.build_beam(emissions, target=target)
    assert torch.equal(index.squeeze(-1), target)
