# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""End-to-end checks on the real model: build, forward, loss, backward, decode.

Everything else in this suite exercises the pieces in isolation; this file wires
the actual ``bert_transformer_seq2seq`` model to the actual criterion and
decoding strategy for each of the three output heads.
"""

import argparse

import pytest
import torch

from fairseq.criterions.label_smoothed_length_cross_entropy import (
    LabelSmoothedLengthCrossEntropyCriterion,
)
from fairseq.data import Dictionary
from fairseq.models.bert_seq2seq import Transformer_nonautoregressive
from fairseq.modules.dynamic_crf_output_layer import CRF_INFERENCE_CHOICES
from fairseq.strategies.mask_predict import MaskPredict

HEADS = ['shared_embed', 'mlp', 'crf']
DIM, BATCH, SRC_LEN, TGT_LEN = 16, 2, 5, 6


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


@pytest.fixture
def dictionary():
    d = Dictionary()
    for token in ['the', 'cat', 'sat', 'on', 'a', 'mat', 'and', 'then', 'slept']:
        d.add_symbol(token)
    return d


class StubTask(object):
    """The slice of a fairseq task that build_model and the criterion touch."""

    def __init__(self, dictionary):
        self.source_dictionary = dictionary
        self.target_dictionary = dictionary


def make_args(**overrides):
    args = argparse.Namespace(
        arch='bert_transformer_seq2seq',
        encoder_embed_dim=DIM,
        decoder_embed_dim=DIM,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        max_source_positions=64,
        max_target_positions=64,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        share_all_embeddings=True,
        label_smoothing=0.1,
        tie_adaptive_weights=False,
        left_pad_source=False,
        left_pad_target=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def build(dictionary, **overrides):
    args = make_args(**overrides)
    model = Transformer_nonautoregressive.build_model(args, StubTask(dictionary))
    return args, model


def make_sample(dictionary):
    """A batch shaped like language_pair_self_dataset_mask produces.

    ``prev_output_tokens`` is the reference with some positions replaced by
    <mask>; ``target`` is padding except at exactly those positions.
    """
    pad, mask_idx = dictionary.pad(), dictionary.mask()
    reference = torch.randint(dictionary.nspecial, len(dictionary), (BATCH, TGT_LEN))
    reference[1, -1] = pad                     # a shorter second sentence

    prev_output_tokens = reference.clone()
    target = torch.full_like(reference, pad)
    for row, positions in enumerate([[0, 2], [1, 3]]):
        for position in positions:
            target[row, position] = reference[row, position]
            prev_output_tokens[row, position] = mask_idx

    src_tokens = torch.randint(dictionary.nspecial, len(dictionary), (BATCH, SRC_LEN))
    return reference, {
        'target': target,
        'ntokens': int(target.ne(pad).sum()),
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': torch.full((BATCH,), SRC_LEN, dtype=torch.long),
            'prev_output_tokens': prev_output_tokens,
        },
    }


# --- build ------------------------------------------------------------------

@pytest.mark.parametrize('head', HEADS)
def test_model_builds_for_every_head(dictionary, head):
    _, model = build(dictionary, decoder_output_layer=head)
    assert model.decoder.output_layer_type == head
    assert model.decoder.has_structured_output == (head == 'crf')


@pytest.mark.parametrize('head', HEADS)
def test_forward_produces_vocabulary_logits(dictionary, head):
    _, model = build(dictionary, decoder_output_layer=head)
    _, sample = make_sample(dictionary)
    logits, extra = model(**sample['net_input'])
    assert logits.shape == (BATCH, TGT_LEN, len(dictionary))
    assert extra['predicted_lengths'].shape == (BATCH, 64)


def test_the_crf_head_adds_parameters_over_the_default(dictionary):
    _, default = build(dictionary, decoder_output_layer='shared_embed')
    _, crf = build(dictionary, decoder_output_layer='crf')
    assert sum(p.numel() for p in crf.parameters()) > \
        sum(p.numel() for p in default.parameters())


def test_the_default_head_adds_nothing_over_the_original_model(dictionary):
    """No new parameters, so a released checkpoint still loads as-is."""
    args = make_args()   # as restored from a checkpoint predating the flag
    assert not hasattr(args, 'decoder_output_layer')
    legacy = Transformer_nonautoregressive.build_model(args, StubTask(dictionary))
    _, explicit = build(dictionary, decoder_output_layer='shared_embed')
    assert set(legacy.state_dict()) == set(explicit.state_dict())
    explicit.load_state_dict(legacy.state_dict(), strict=True)


# --- training ---------------------------------------------------------------

@pytest.mark.parametrize('head', HEADS)
def test_loss_and_backward_run_for_every_head(dictionary, head):
    args, model = build(dictionary, decoder_output_layer=head)
    criterion = LabelSmoothedLengthCrossEntropyCriterion(args, StubTask(dictionary))
    _, sample = make_sample(dictionary)

    loss, sample_size, logging = criterion(model, sample)

    assert torch.isfinite(loss)
    assert sample_size == sample['ntokens']
    loss.backward()
    trained = [n for n, p in model.named_parameters() if p.grad is not None
               and p.grad.abs().sum() > 0]
    assert trained


def test_only_the_crf_head_reports_a_crf_loss(dictionary):
    _, sample = make_sample(dictionary)
    for head in HEADS:
        args, model = build(dictionary, decoder_output_layer=head)
        criterion = LabelSmoothedLengthCrossEntropyCriterion(args, StubTask(dictionary))
        logging = criterion(model, sample)[2]
        assert ('crf_loss' in logging) == (head == 'crf')


def test_the_crf_loss_is_a_genuine_negative_log_likelihood(dictionary):
    """Non-negative on the real model, with the real vocabulary and beam."""
    args, model = build(dictionary, decoder_output_layer='crf', crf_beam_size=3)
    criterion = LabelSmoothedLengthCrossEntropyCriterion(args, StubTask(dictionary))
    _, sample = make_sample(dictionary)
    assert criterion(model, sample)[2]['crf_loss'] >= 0


def test_the_crf_transition_parameters_actually_train(dictionary):
    args, model = build(dictionary, decoder_output_layer='crf')
    criterion = LabelSmoothedLengthCrossEntropyCriterion(args, StubTask(dictionary))
    _, sample = make_sample(dictionary)

    criterion(model, sample)[0].backward()
    head = model.decoder.output_projection
    assert head.transition_source.weight.grad.abs().sum() > 0
    assert head.transition_target.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in head.dynamic_transition.parameters())


def test_the_reference_the_crf_scores_is_the_full_target_sequence(dictionary):
    """Guards the recombination of the CMLM's split reference."""
    reference, sample = make_sample(dictionary)
    recovered = Transformer_nonautoregressive.reference_tokens(
        sample['net_input']['prev_output_tokens'], sample['target'], dictionary.pad())
    assert torch.equal(recovered, reference)


def test_an_mlp_emission_head_trains_inside_the_crf(dictionary):
    args, model = build(dictionary, decoder_output_layer='crf', crf_emission_layer='mlp')
    criterion = LabelSmoothedLengthCrossEntropyCriterion(args, StubTask(dictionary))
    _, sample = make_sample(dictionary)

    criterion(model, sample)[0].backward()
    emission = model.decoder.output_projection.emission_layer
    assert emission.projection.weight.grad.abs().sum() > 0


# --- generation -------------------------------------------------------------

@pytest.mark.parametrize('head', HEADS)
def test_mask_predict_generates_with_every_head(dictionary, head):
    _, model = build(dictionary, decoder_output_layer=head)
    _, sample = make_sample(dictionary)
    strategy = MaskPredict(argparse.Namespace(decoding_iterations=4))

    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(
            sample['net_input']['src_tokens'], sample['net_input']['src_lengths'])
        tokens = sample['net_input']['prev_output_tokens'].clone()
        pad_mask = tokens.eq(dictionary.pad())
        output, lprobs = strategy.generate(model, encoder_out, tokens, dictionary)

    assert output.shape == (BATCH, TGT_LEN)
    assert lprobs.shape == (BATCH,)
    # Padding must survive decoding. The converse does not hold: an untrained
    # model can predict the pad symbol at a real position, since nothing excludes
    # it from the vocabulary -- true of the original head too.
    assert bool(output[pad_mask].eq(dictionary.pad()).all())
    assert bool((output < len(dictionary)).all())


@pytest.mark.parametrize('mode', CRF_INFERENCE_CHOICES)
def test_every_crf_inference_mode_generates(dictionary, mode):
    _, model = build(dictionary, decoder_output_layer='crf', crf_inference=mode)
    _, sample = make_sample(dictionary)
    strategy = MaskPredict(argparse.Namespace(decoding_iterations=4))

    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(
            sample['net_input']['src_tokens'], sample['net_input']['src_lengths'])
        output, lprobs = strategy.generate(
            model, encoder_out, sample['net_input']['prev_output_tokens'].clone(),
            dictionary)

    assert output.shape == (BATCH, TGT_LEN)
    assert torch.isfinite(lprobs).all()


def test_model_overrides_can_switch_the_inference_mode(dictionary):
    """How ``--model-overrides`` at generation time reaches the head: the
    overrides are applied to ``args`` before the model is built."""
    args = make_args(decoder_output_layer='crf')
    for key, value in {'crf_inference': 'marginal'}.items():
        setattr(args, key, value)
    model = Transformer_nonautoregressive.build_model(args, StubTask(dictionary))
    assert model.decoder.output_projection.inference == 'marginal'


# --- fp16 -------------------------------------------------------------------

def test_a_half_precision_crf_model_still_trains(dictionary):
    """``--fp16`` is in the documented training command, and the CRF's logsumexp
    chains have to survive it."""
    args, model = build(dictionary, decoder_output_layer='crf')
    criterion = LabelSmoothedLengthCrossEntropyCriterion(args, StubTask(dictionary))
    _, sample = make_sample(dictionary)

    model.half()
    loss = criterion(model, sample)[0]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.decoder.output_projection.transition_source.weight.grad is not None
