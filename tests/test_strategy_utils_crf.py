# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""How a structured head feeds the iterative decoding strategies.

``generate_step_with_prob`` is the single point where every strategy in
fairseq/strategies turns a decoder output into (tokens, confidence), so making it
CRF-aware keeps mask_predict, easy_first and left_to_right working unchanged.
"""

import argparse

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data import Dictionary
from fairseq.modules.dynamic_crf_output_layer import (
    CRF_INFERENCE_CHOICES, DynamicCRFOutputLayer,
)
from fairseq.modules.output_layer import SharedEmbeddingOutputLayer
from fairseq.strategies.mask_predict import MaskPredict
from fairseq.strategies.strategy_utils import generate_step_with_prob

DIM, RANK, BEAM = 6, 3, 4
BATCH, LENGTH = 3, 6


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


@pytest.fixture
def dictionary():
    d = Dictionary()
    for token in ['the', 'cat', 'sat', 'on', 'a', 'mat', 'and', 'slept']:
        d.add_symbol(token)
    return d


def build_crf(dictionary, inference='viterbi_marginal'):
    embed_tokens = nn.Embedding(len(dictionary), DIM, dictionary.pad())
    return DynamicCRFOutputLayer(
        emission_layer=SharedEmbeddingOutputLayer(embed_tokens=embed_tokens),
        vocab_size=len(dictionary), feature_dim=DIM, low_rank_dim=RANK,
        beam_size=BEAM, dynamic_transition=True, dynamic_hidden_dim=8,
        inference=inference,
    )


class StubDecoder(nn.Module):
    """Stands in for SelfTransformerDecoder, returning the same shape of output."""

    def __init__(self, dictionary, output_layer=None):
        super().__init__()
        self.dictionary = dictionary
        self.output_layer = output_layer
        self.embed = nn.Embedding(len(dictionary), DIM, dictionary.pad())

    def forward(self, prev_output_tokens, encoder_out=None):
        features = self.embed(prev_output_tokens)
        extra = {'attn': None, 'inner_states': [], 'predicted_lengths': None}
        if self.output_layer is None:
            return F.linear(features, self.embed.weight), extra
        extra['features'] = features
        extra['output_layer'] = self.output_layer
        extra['output_mask'] = prev_output_tokens.ne(self.dictionary.pad())
        return self.output_layer(features), extra


class StubModel(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder


# --- the unstructured path is untouched -------------------------------------

def test_unstructured_output_keeps_the_original_behaviour():
    logits = torch.randn(BATCH, LENGTH, 9)
    tokens, confidence, probs = generate_step_with_prob((logits, {}))

    expected = F.softmax(logits, dim=-1)
    torch.testing.assert_close(probs, expected)
    assert torch.equal(tokens, expected.argmax(dim=-1))
    torch.testing.assert_close(confidence, expected.max(dim=-1).values)


def test_extra_without_an_output_layer_key_is_fine():
    logits = torch.randn(BATCH, LENGTH, 9)
    out = (logits, {'attn': None, 'inner_states': []})
    tokens, _, _ = generate_step_with_prob(out)
    assert torch.equal(tokens, F.softmax(logits, dim=-1).argmax(dim=-1))


def test_an_unstructured_head_in_extra_is_not_mistaken_for_a_crf(dictionary):
    """An MLP or shared-embedding head has no sequence-level decode step."""
    embed_tokens = nn.Embedding(len(dictionary), DIM, dictionary.pad())
    plain = SharedEmbeddingOutputLayer(embed_tokens=embed_tokens)
    logits = torch.randn(BATCH, LENGTH, len(dictionary))
    tokens, confidence, _ = generate_step_with_prob((logits, {'output_layer': plain}))
    assert torch.equal(tokens, F.softmax(logits, dim=-1).argmax(dim=-1))


# --- the structured path ----------------------------------------------------

@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_structured_output_is_decoded_by_the_crf(dictionary, mode):
    crf = build_crf(dictionary, mode)
    features = torch.randn(BATCH, LENGTH, DIM)
    mask = torch.ones(BATCH, LENGTH, dtype=torch.bool)
    out = (crf(features), {
        'features': features, 'output_layer': crf, 'output_mask': mask})

    tokens, confidence, probs = generate_step_with_prob(out)
    expected = crf.decode(features, mask)
    assert torch.equal(tokens, expected[0])
    torch.testing.assert_close(confidence, expected[1])
    torch.testing.assert_close(probs, expected[2])


def test_structured_decoding_differs_from_the_emission_argmax(dictionary):
    """Otherwise the CRF would not be influencing generation at all."""
    differences = 0
    for seed in range(20):
        torch.manual_seed(seed)
        crf = build_crf(dictionary)
        features = torch.randn(BATCH, LENGTH, DIM)
        mask = torch.ones(BATCH, LENGTH, dtype=torch.bool)
        emissions = crf(features)
        out = (emissions, {
            'features': features, 'output_layer': crf, 'output_mask': mask})
        tokens, _, _ = generate_step_with_prob(out)
        if not torch.equal(tokens, emissions.argmax(dim=-1)):
            differences += 1
    assert differences > 0


def test_structured_output_falls_back_when_features_are_absent(dictionary):
    """Defensive: a head present without its features must not crash decoding."""
    crf = build_crf(dictionary)
    logits = torch.randn(BATCH, LENGTH, len(dictionary))
    tokens, _, _ = generate_step_with_prob((logits, {'output_layer': crf}))
    assert torch.equal(tokens, F.softmax(logits, dim=-1).argmax(dim=-1))


# --- end to end through mask-predict ----------------------------------------

def masked_target(dictionary):
    tokens = torch.randint(dictionary.nspecial, len(dictionary), (BATCH, LENGTH))
    tokens[1, -1] = dictionary.pad()
    tokens[2, -2:] = dictionary.pad()
    return tokens


@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_mask_predict_runs_with_a_structured_head(dictionary, mode):
    strategy = MaskPredict(argparse.Namespace(decoding_iterations=5))
    model = StubModel(StubDecoder(dictionary, build_crf(dictionary, mode)))
    tokens = masked_target(dictionary)

    with torch.no_grad():
        output, lprobs = strategy.generate(model, None, tokens, dictionary)

    assert output.shape == (BATCH, LENGTH)
    assert lprobs.shape == (BATCH,)
    assert bool((output >= 0).all()) and bool((output < len(dictionary)).all())


@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_mask_predict_preserves_padding_with_a_structured_head(dictionary, mode):
    strategy = MaskPredict(argparse.Namespace(decoding_iterations=5))
    model = StubModel(StubDecoder(dictionary, build_crf(dictionary, mode)))
    tokens = masked_target(dictionary)
    pad_mask = tokens.eq(dictionary.pad())

    with torch.no_grad():
        output, _ = strategy.generate(model, None, tokens.clone(), dictionary)

    # Padding survives; the converse does not hold, since nothing stops an
    # untrained model predicting the pad symbol at a real position.
    assert bool(output[pad_mask].eq(dictionary.pad()).all())


def test_mask_predict_still_runs_with_an_unstructured_head(dictionary):
    """The baseline path must be unaffected by the CRF-awareness."""
    strategy = MaskPredict(argparse.Namespace(decoding_iterations=5))
    model = StubModel(StubDecoder(dictionary, output_layer=None))
    tokens = masked_target(dictionary)

    with torch.no_grad():
        output, lprobs = strategy.generate(model, None, tokens, dictionary)

    assert output.shape == (BATCH, LENGTH)
    assert lprobs.shape == (BATCH,)


@pytest.mark.parametrize('strategy_name', ['easy_first', 'left_to_right'])
def test_the_beam_search_strategies_still_run_with_a_structured_head(
        dictionary, strategy_name):
    """These two do their own beam search over the full vocabulary straight from
    ``decoder_out[0]`` rather than going through ``generate_step_with_prob``, so
    they read the CRF's emission scores and ignore its transitions. Keeping
    ``forward`` equal to the emission projection is what lets them keep working.
    """
    from fairseq.strategies import STRATEGY_REGISTRY

    strategy = STRATEGY_REGISTRY[strategy_name](argparse.Namespace(beam=2))
    model = StubModel(StubDecoder(dictionary, build_crf(dictionary)))
    tokens = masked_target(dictionary)
    tokens[:, 1] = dictionary.mask()

    encoder_out = {
        'encoder_out': torch.randn(4, BATCH, DIM),
        'encoder_padding_mask': None,
        'predicted_lengths': None,
    }
    with torch.no_grad():
        output, lprobs = strategy.generate(model, encoder_out, tokens, dictionary)

    assert output.shape == (BATCH, LENGTH)
    assert lprobs.shape == (BATCH,)


def test_structured_confidences_never_select_padding_for_remasking(dictionary):
    """``select_worst`` picks the least confident positions; padding is given
    confidence 1.0 so it is never among them."""
    crf = build_crf(dictionary)
    features = torch.randn(1, 5, DIM)
    mask = torch.tensor([[True, True, True, False, False]])
    out = (crf(features), {
        'features': features, 'output_layer': crf, 'output_mask': mask})

    _, confidence, _ = generate_step_with_prob(out)
    assert bool((confidence[:, 3:] == 1.0).all())
    strategy = MaskPredict(argparse.Namespace(decoding_iterations=2))
    chosen = strategy.select_worst(confidence, torch.tensor([2]))
    assert bool((chosen < 3).all())
