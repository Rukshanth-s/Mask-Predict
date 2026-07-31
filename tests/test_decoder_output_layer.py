# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""Wiring of the selected output head into ``SelfTransformerDecoder``."""

import argparse

import pytest
import torch
import torch.nn as nn

from fairseq.data import Dictionary
from fairseq.models.bert_seq2seq import SelfTransformerDecoder, base_architecture
from fairseq.modules.dynamic_crf_output_layer import DynamicCRFOutputLayer
from fairseq.modules.output_layer import MLPOutputLayer, SharedEmbeddingOutputLayer

DIM, LAYERS = 8, 1
BATCH, LENGTH = 2, 4


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


@pytest.fixture
def dictionary():
    d = Dictionary()
    for token in ['the', 'cat', 'sat', 'on', 'a', 'mat']:
        d.add_symbol(token)
    return d


def make_args(**overrides):
    args = argparse.Namespace(
        encoder_embed_dim=DIM,
        decoder_embed_dim=DIM,
        decoder_layers=LAYERS,
        decoder_ffn_embed_dim=DIM * 2,
        decoder_attention_heads=2,
        max_target_positions=64,
        max_source_positions=64,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        share_decoder_input_output_embed=True,
        no_dec_token_positional_embeddings=False,
        tie_adaptive_weights=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    base_architecture(args)
    return args


def build_decoder(dictionary, **overrides):
    args = make_args(**overrides)
    embed_tokens = nn.Embedding(len(dictionary), DIM, dictionary.pad())
    return SelfTransformerDecoder(args, dictionary, embed_tokens)


def fake_encoder_out(dictionary, src_len=3):
    return {
        'encoder_out': torch.randn(src_len, BATCH, DIM),
        'encoder_padding_mask': None,
        'predicted_lengths': torch.log_softmax(torch.randn(BATCH, 64), dim=-1),
    }


def prev_output_tokens(dictionary):
    tokens = torch.randint(dictionary.nspecial, len(dictionary), (BATCH, LENGTH))
    tokens[1, -1] = dictionary.pad()   # a shorter second sequence
    return tokens


# --- checkpoint compatibility of the default -------------------------------

def test_shared_embed_decoder_state_dict_is_exactly_the_original(dictionary):
    """The guarantee that the released pre-trained checkpoints still load.

    Routing the default head through a module must not add, rename or remove a
    single state-dict key.
    """
    decoder = build_decoder(dictionary)
    assert set(decoder.state_dict()) == {
        'version',
        'embed_tokens.weight',
        'embed_positions.weight',
        'layers.0.self_attn.in_proj_weight',
        'layers.0.self_attn.in_proj_bias',
        'layers.0.self_attn.out_proj.weight',
        'layers.0.self_attn.out_proj.bias',
        'layers.0.self_attn_layer_norm.gamma',
        'layers.0.self_attn_layer_norm.beta',
        'layers.0.encoder_attn.in_proj_weight',
        'layers.0.encoder_attn.in_proj_bias',
        'layers.0.encoder_attn.out_proj.weight',
        'layers.0.encoder_attn.out_proj.bias',
        'layers.0.encoder_attn_layer_norm.gamma',
        'layers.0.encoder_attn_layer_norm.beta',
        'layers.0.fc1.weight',
        'layers.0.fc1.bias',
        'layers.0.fc2.weight',
        'layers.0.fc2.bias',
        'layers.0.final_layer_norm.gamma',
        'layers.0.final_layer_norm.beta',
    }


def test_shared_embed_head_contributes_no_state(dictionary):
    decoder = build_decoder(dictionary)
    assert isinstance(decoder.output_projection, SharedEmbeddingOutputLayer)
    assert not any(key.startswith('output_projection') for key in decoder.state_dict())


def test_untied_shared_embed_keeps_the_original_embed_out_name(dictionary):
    decoder = build_decoder(dictionary, share_decoder_input_output_embed=False)
    assert 'embed_out' in decoder.state_dict()
    assert not any(key.startswith('output_projection') for key in decoder.state_dict())


def test_args_without_any_of_the_new_flags_still_build_the_original_head(dictionary):
    """An ``args`` restored from a checkpoint predating all of these flags."""
    args = make_args()
    for attr in ['decoder_output_layer', 'decoder_output_mlp_hidden_dim',
                 'crf_emission_layer', 'crf_low_rank_dim', 'crf_beam_size',
                 'crf_no_dynamic_transition', 'crf_dynamic_hidden_dim',
                 'crf_nar_loss_weight', 'crf_inference']:
        delattr(args, attr)

    embed_tokens = nn.Embedding(len(dictionary), DIM, dictionary.pad())
    decoder = SelfTransformerDecoder(args, dictionary, embed_tokens)
    assert isinstance(decoder.output_projection, SharedEmbeddingOutputLayer)
    assert not decoder.has_structured_output


def test_a_shared_embed_checkpoint_loads_strictly(dictionary):
    source = build_decoder(dictionary)
    destination = build_decoder(dictionary)
    destination.load_state_dict(source.state_dict(), strict=True)


# --- the other two heads ----------------------------------------------------

def test_mlp_head_is_built_and_owns_its_weights(dictionary):
    decoder = build_decoder(dictionary, decoder_output_layer='mlp')
    assert isinstance(decoder.output_projection, MLPOutputLayer)
    keys = {k for k in decoder.state_dict() if k.startswith('output_projection')}
    assert keys == {
        'output_projection.dense.weight', 'output_projection.dense.bias',
        'output_projection.layer_norm.gamma', 'output_projection.layer_norm.beta',
        'output_projection.projection.weight', 'output_projection.projection.bias',
    }


def test_crf_head_is_built_with_its_transition_parameters(dictionary):
    decoder = build_decoder(dictionary, decoder_output_layer='crf', crf_low_rank_dim=4)
    assert isinstance(decoder.output_projection, DynamicCRFOutputLayer)
    keys = {k for k in decoder.state_dict() if k.startswith('output_projection')}
    assert 'output_projection.transition_source.weight' in keys
    assert 'output_projection.transition_target.weight' in keys
    assert any('dynamic_transition' in key for key in keys)
    assert decoder.output_projection.transition_source.weight.shape == (len(dictionary), 4)


def test_crf_head_can_use_the_mlp_as_its_emission_scorer(dictionary):
    decoder = build_decoder(
        dictionary, decoder_output_layer='crf', crf_emission_layer='mlp')
    assert isinstance(decoder.output_projection.emission_layer, MLPOutputLayer)


def test_crf_flags_reach_the_head(dictionary):
    decoder = build_decoder(
        dictionary, decoder_output_layer='crf', crf_beam_size=7,
        crf_low_rank_dim=5, crf_inference='marginal',
        crf_no_dynamic_transition=True)
    head = decoder.output_projection
    assert head.beam_size == 7
    assert head.low_rank_dim == 5
    assert head.inference == 'marginal'
    assert head.dynamic_transition is None


# --- forward ----------------------------------------------------------------

@pytest.mark.parametrize('head', ['shared_embed', 'mlp', 'crf'])
def test_forward_returns_vocabulary_logits(dictionary, head):
    decoder = build_decoder(dictionary, decoder_output_layer=head)
    tokens = prev_output_tokens(dictionary)
    logits, extra = decoder(tokens, fake_encoder_out(dictionary))
    assert logits.shape == (BATCH, LENGTH, len(dictionary))
    assert 'predicted_lengths' in extra
    assert 'inner_states' in extra


@pytest.mark.parametrize('head', ['shared_embed', 'mlp'])
def test_unstructured_heads_expose_no_structured_extras(dictionary, head):
    decoder = build_decoder(dictionary, decoder_output_layer=head)
    _, extra = decoder(prev_output_tokens(dictionary), fake_encoder_out(dictionary))
    assert extra.get('output_layer') is None
    assert extra.get('features') is None


def test_crf_head_exposes_features_and_itself_for_the_loss_and_decoding(dictionary):
    decoder = build_decoder(dictionary, decoder_output_layer='crf')
    tokens = prev_output_tokens(dictionary)
    logits, extra = decoder(tokens, fake_encoder_out(dictionary))

    assert extra['output_layer'] is decoder.output_projection
    assert extra['features'].shape == (BATCH, LENGTH, DIM)
    # The chain mask marks real tokens, which is the inverse of fairseq's
    # padding masks; padding is what the CRF must skip over.
    assert torch.equal(extra['output_mask'], tokens.ne(dictionary.pad()))
    # out[0] stays the emission logits so existing consumers keep working.
    torch.testing.assert_close(logits, decoder.output_projection(extra['features']))


@pytest.mark.parametrize('head', ['shared_embed', 'mlp', 'crf'])
def test_forward_is_differentiable(dictionary, head):
    decoder = build_decoder(dictionary, decoder_output_layer=head)
    logits, _ = decoder(prev_output_tokens(dictionary), fake_encoder_out(dictionary))
    logits.sum().backward()
    assert any(p.grad is not None for p in decoder.parameters())


# --- interaction with the pre-existing head options -------------------------

def test_adaptive_softmax_conflicts_with_a_non_default_head(dictionary):
    """Adaptive softmax replaces the output layer wholesale, so combining the
    two would silently ignore the flag."""
    with pytest.raises(ValueError):
        build_decoder(dictionary, decoder_output_layer='mlp',
                      adaptive_softmax_cutoff='2,4', adaptive_softmax_dropout=0,
                      adaptive_softmax_factor=4, tie_adaptive_proj=False)


def test_adaptive_softmax_still_works_with_the_default_head(dictionary):
    decoder = build_decoder(dictionary, adaptive_softmax_cutoff='2,4',
                            adaptive_softmax_dropout=0, adaptive_softmax_factor=4,
                            tie_adaptive_proj=False)
    assert decoder.adaptive_softmax is not None
    assert decoder.output_projection is None


@pytest.mark.parametrize('head', ['shared_embed', 'mlp', 'crf'])
def test_remove_head_builds_no_output_projection(dictionary, head):
    decoder = build_decoder(dictionary, decoder_output_layer=head, remove_head=True)
    assert decoder.output_projection is None
