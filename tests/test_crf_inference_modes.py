# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""The three ``--crf-inference`` modes.

A CRF scores whole paths, but the decoding strategies in ``fairseq/strategies``
need a *per-position* confidence to decide which positions to re-mask. These
tests pin down how each mode derives tokens and confidences.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules.dynamic_crf_output_layer import (
    CRF_INFERENCE_CHOICES, DynamicCRFOutputLayer, log_marginals, viterbi_decode,
)
from fairseq.modules.output_layer import SharedEmbeddingOutputLayer

VOCAB, DIM, RANK, BEAM = 9, 6, 3, 4
BATCH, LENGTH = 2, 5


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


def build_crf(inference='viterbi_marginal', vocab_size=VOCAB, **overrides):
    settings = dict(
        low_rank_dim=RANK,
        beam_size=BEAM,
        dynamic_transition=True,
        dynamic_hidden_dim=8,
    )
    settings.update(overrides)
    emission = SharedEmbeddingOutputLayer(
        embed_tokens=nn.Embedding(vocab_size, DIM, padding_idx=1))
    return DynamicCRFOutputLayer(
        emission_layer=emission, vocab_size=vocab_size, feature_dim=DIM,
        inference=inference, **settings)


@pytest.fixture
def features():
    return torch.randn(BATCH, LENGTH, DIM)


# --- the common contract ----------------------------------------------------

@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_decode_returns_tokens_confidence_and_emission_probs(features, mode):
    crf = build_crf(mode)
    tokens, confidence, probs = crf.decode(features)

    assert tokens.shape == (BATCH, LENGTH)
    assert tokens.dtype == torch.long
    assert confidence.shape == (BATCH, LENGTH)
    assert probs.shape == (BATCH, LENGTH, VOCAB)


@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_decoded_tokens_are_valid_vocabulary_ids(features, mode):
    tokens, _, _ = build_crf(mode).decode(features)
    assert bool((tokens >= 0).all()) and bool((tokens < VOCAB).all())


@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_confidences_are_probabilities(features, mode):
    _, confidence, _ = build_crf(mode).decode(features)
    assert bool((confidence > 0).all())
    assert bool((confidence <= 1.0 + 1e-6).all())


@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_emission_probs_are_the_softmax_of_the_emission_head(features, mode):
    crf = build_crf(mode)
    _, _, probs = crf.decode(features)
    torch.testing.assert_close(probs, F.softmax(crf(features), dim=-1))


@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_decoded_tokens_come_from_the_beam(features, mode):
    """Truncation means only top-k candidates can ever be emitted."""
    crf = build_crf(mode)
    emissions = crf(features)
    _, beam_index = crf.build_beam(emissions)
    tokens, _, _ = crf.decode(features)
    assert bool((beam_index == tokens.unsqueeze(-1)).any(dim=-1).all())


# --- reduction to the unstructured head -------------------------------------

def unstructured_crf(mode, **overrides):
    """A CRF with its transition scores zeroed out."""
    crf = build_crf(mode, dynamic_transition=False, **overrides)
    with torch.no_grad():
        crf.transition_source.weight.zero_()
        crf.transition_target.weight.zero_()
    return crf


@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_zero_transitions_reduce_every_mode_to_emission_argmax(features, mode):
    """The key sanity check tying option 3 back to option 1.

    With no transition scores and no beam truncation the CRF is exactly a stack
    of independent softmaxes, so all three modes must return the plain
    per-position argmax and its softmax probability -- what the shared-embedding
    head produces on its own.
    """
    crf = unstructured_crf(mode, beam_size=VOCAB)
    emissions = crf(features)
    tokens, confidence, _ = crf.decode(features)

    assert torch.equal(tokens, emissions.argmax(dim=-1))
    expected = F.softmax(emissions, dim=-1).max(dim=-1).values
    torch.testing.assert_close(confidence, expected)


def test_all_three_modes_agree_when_there_is_no_structure(features):
    """Same reduction, stated as agreement between the modes."""
    results = []
    for mode in CRF_INFERENCE_CHOICES:
        torch.manual_seed(0)
        results.append(unstructured_crf(mode, beam_size=VOCAB).decode(features)[:2])

    for tokens, confidence in results[1:]:
        assert torch.equal(tokens, results[0][0])
        torch.testing.assert_close(confidence, results[0][1])


@pytest.mark.parametrize('mode', ['viterbi_marginal', 'marginal'])
def test_marginal_confidence_is_normalised_within_the_beam(features, mode):
    """A consequence of the beam approximation worth being explicit about.

    The partition function is only summed over the beam, so marginal-derived
    confidences are normalised across the k candidates rather than the whole
    vocabulary, and are therefore systematically larger than an emission softmax
    probability. That is harmless for the iterative strategies, which only ever
    compare confidences against each other within one sequence.
    """
    crf = unstructured_crf(mode, beam_size=BEAM)
    emissions = crf(features)
    beam_scores, _ = crf.build_beam(emissions)

    tokens, confidence, _ = crf.decode(features)
    assert torch.equal(tokens, emissions.argmax(dim=-1))
    torch.testing.assert_close(
        confidence, F.softmax(beam_scores, dim=-1).max(dim=-1).values)
    assert bool((confidence >= F.softmax(emissions, dim=-1).max(dim=-1).values).all())


# --- mode-specific behaviour ------------------------------------------------

def test_viterbi_marginal_uses_the_viterbi_path_and_marginal_confidence(features):
    crf = build_crf('viterbi_marginal')
    emissions = crf(features)
    beam_scores, beam_index = crf.build_beam(emissions)
    transitions = crf.build_transitions(features.float(), beam_index)

    slots, _ = viterbi_decode(beam_scores, transitions)
    expected_tokens = beam_index.gather(-1, slots.unsqueeze(-1)).squeeze(-1)
    expected_confidence = log_marginals(beam_scores, transitions).gather(
        -1, slots.unsqueeze(-1)).squeeze(-1).exp()

    tokens, confidence, _ = crf.decode(features)
    assert torch.equal(tokens, expected_tokens)
    torch.testing.assert_close(confidence, expected_confidence)


def test_marginal_mode_uses_the_argmax_marginal_for_both(features):
    crf = build_crf('marginal')
    emissions = crf(features)
    beam_scores, beam_index = crf.build_beam(emissions)
    transitions = crf.build_transitions(features.float(), beam_index)

    best = log_marginals(beam_scores, transitions).max(dim=-1)
    expected_tokens = beam_index.gather(-1, best.indices.unsqueeze(-1)).squeeze(-1)

    tokens, confidence, _ = crf.decode(features)
    assert torch.equal(tokens, expected_tokens)
    torch.testing.assert_close(confidence, best.values.exp())


def test_viterbi_emission_confidence_is_the_plain_emission_probability(features):
    crf = build_crf('viterbi_emission')
    tokens, confidence, probs = crf.decode(features)
    torch.testing.assert_close(
        confidence, probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1))


def test_viterbi_modes_agree_on_tokens_and_differ_only_in_confidence(features):
    """``viterbi_marginal`` and ``viterbi_emission`` share the decoding rule."""
    torch.manual_seed(0)
    marginal = build_crf('viterbi_marginal')
    torch.manual_seed(0)
    emission = build_crf('viterbi_emission')

    marginal_tokens, marginal_conf, _ = marginal.decode(features)
    emission_tokens, emission_conf, _ = emission.decode(features)

    assert torch.equal(marginal_tokens, emission_tokens)
    assert not torch.allclose(marginal_conf, emission_conf)


def test_marginal_and_viterbi_are_genuinely_different_decoding_rules():
    """Position-wise best marginals need not form the globally best path.

    If these two modes never disagreed the flag would be pointless, so search a
    few random models for a disagreement.
    """
    disagreements = 0
    for seed in range(40):
        torch.manual_seed(seed)
        viterbi = build_crf('viterbi_marginal', vocab_size=12)
        torch.manual_seed(seed)
        marginal = build_crf('marginal', vocab_size=12)
        features = torch.randn(2, 6, DIM)
        if not torch.equal(viterbi.decode(features)[0], marginal.decode(features)[0]):
            disagreements += 1
    assert disagreements > 0


# --- padding ----------------------------------------------------------------

@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_padded_positions_get_full_confidence(mode):
    """``MaskPredict.select_worst`` re-masks the least confident positions, so
    padding must look maximally confident and never be selected. This matches
    the existing convention in fairseq/strategies/mask_predict.py, which assigns
    1.0 to padded positions before every selection."""
    crf = build_crf(mode)
    features = torch.randn(1, 5, DIM)
    mask = torch.tensor([[True, True, True, False, False]])

    _, confidence, _ = crf.decode(features, mask)
    torch.testing.assert_close(confidence[:, 3:], torch.ones(1, 2))


@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_padding_does_not_change_the_decoded_prefix(mode):
    crf = build_crf(mode)
    short = torch.randn(1, 3, DIM)
    padded = torch.cat([short, torch.randn(1, 2, DIM)], dim=1)
    mask = torch.tensor([[True, True, True, False, False]])

    padded_tokens, padded_conf, _ = crf.decode(padded, mask)
    short_tokens, short_conf, _ = crf.decode(short)
    assert torch.equal(padded_tokens[:, :3], short_tokens)
    torch.testing.assert_close(padded_conf[:, :3], short_conf)


# --- switching the mode -----------------------------------------------------

def test_inference_mode_can_be_switched_after_construction(features):
    """``--model-overrides`` is applied before the model is built, but keeping
    the mode readable at call time makes the head easy to probe and re-use."""
    crf = build_crf('viterbi_marginal')
    first = crf.decode(features)[1]
    crf.inference = 'viterbi_emission'
    assert not torch.allclose(first, crf.decode(features)[1])


def test_decode_rejects_an_unknown_mode_set_after_construction(features):
    crf = build_crf('marginal')
    crf.inference = 'nucleus'
    with pytest.raises(ValueError):
        crf.decode(features)
