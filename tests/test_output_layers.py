# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""Tests for the plain (non-structured) decoder output heads."""

import argparse

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules.bert_layer_norm import BertLayerNorm
from fairseq.modules.output_layer import (
    MLPOutputLayer, SharedEmbeddingOutputLayer, build_output_layer,
)

VOCAB, DIM, BATCH, LENGTH = 11, 8, 3, 5


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


@pytest.fixture
def embed_tokens():
    return nn.Embedding(VOCAB, DIM, padding_idx=1)


@pytest.fixture
def features():
    return torch.randn(BATCH, LENGTH, DIM)


def head_args(**overrides):
    args = argparse.Namespace(
        decoder_output_layer='shared_embed',
        decoder_output_dim=DIM,
        decoder_output_mlp_hidden_dim=DIM,
        share_decoder_input_output_embed=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# --- shared_embed -----------------------------------------------------------

def test_shared_embed_tied_matches_the_original_expression(embed_tokens, features):
    """Must reproduce ``F.linear(x, self.embed_tokens.weight)`` exactly."""
    layer = SharedEmbeddingOutputLayer(embed_tokens=embed_tokens)
    torch.testing.assert_close(layer(features), F.linear(features, embed_tokens.weight))


def test_shared_embed_untied_matches_the_original_expression(features):
    """Must reproduce ``F.linear(x, self.embed_out)`` exactly."""
    embed_out = nn.Parameter(torch.randn(VOCAB, DIM))
    layer = SharedEmbeddingOutputLayer(embed_out=embed_out)
    torch.testing.assert_close(layer(features), F.linear(features, embed_out))


def test_shared_embed_output_shape(embed_tokens, features):
    layer = SharedEmbeddingOutputLayer(embed_tokens=embed_tokens)
    assert layer(features).shape == (BATCH, LENGTH, VOCAB)


def test_shared_embed_owns_no_parameters(embed_tokens, features):
    """The checkpoint-compatibility guarantee.

    The weights it projects with are owned by the decoder, so this head must add
    neither parameters nor state-dict keys; otherwise wrapping the existing
    behaviour in a module would rename keys and break the released checkpoints.
    """
    layer = SharedEmbeddingOutputLayer(embed_tokens=embed_tokens)
    assert list(layer.parameters()) == []
    assert layer.state_dict() == {}
    assert list(layer.buffers()) == []

    embed_out = nn.Parameter(torch.randn(VOCAB, DIM))
    untied = SharedEmbeddingOutputLayer(embed_out=embed_out)
    assert list(untied.parameters()) == []
    assert untied.state_dict() == {}


def test_shared_embed_still_propagates_gradient_to_the_decoder_weights(embed_tokens, features):
    """Not owning the weight must not stop gradients reaching it."""
    layer = SharedEmbeddingOutputLayer(embed_tokens=embed_tokens)
    layer(features).sum().backward()
    assert embed_tokens.weight.grad is not None
    assert embed_tokens.weight.grad.abs().sum() > 0


def test_shared_embed_tracks_later_weight_mutation(embed_tokens, features):
    """It must read the live weight, not a snapshot taken at construction."""
    layer = SharedEmbeddingOutputLayer(embed_tokens=embed_tokens)
    before = layer(features)
    with torch.no_grad():
        embed_tokens.weight.mul_(2.0)
    torch.testing.assert_close(layer(features), before * 2.0)


def test_shared_embed_requires_exactly_one_weight_source(embed_tokens):
    with pytest.raises(ValueError):
        SharedEmbeddingOutputLayer()
    with pytest.raises(ValueError):
        SharedEmbeddingOutputLayer(
            embed_tokens=embed_tokens, embed_out=nn.Parameter(torch.randn(VOCAB, DIM)))


# --- mlp --------------------------------------------------------------------

def test_mlp_output_shape(features):
    layer = MLPOutputLayer(VOCAB, DIM, hidden_dim=DIM)
    assert layer(features).shape == (BATCH, LENGTH, VOCAB)


def test_mlp_has_the_expected_bert_style_structure(features):
    layer = MLPOutputLayer(VOCAB, DIM, hidden_dim=16)
    assert isinstance(layer.dense, nn.Linear)
    assert layer.dense.weight.shape == (16, DIM)
    assert isinstance(layer.layer_norm, BertLayerNorm)
    assert isinstance(layer.projection, nn.Linear)
    assert layer.projection.weight.shape == (VOCAB, 16)
    assert layer.projection.bias is not None


def test_mlp_hidden_dim_is_honoured(features):
    layer = MLPOutputLayer(VOCAB, DIM, hidden_dim=32)
    assert layer.dense.out_features == 32
    assert layer.projection.in_features == 32
    assert layer(features).shape == (BATCH, LENGTH, VOCAB)


def test_mlp_state_dict_keys(features):
    layer = MLPOutputLayer(VOCAB, DIM, hidden_dim=16)
    assert set(layer.state_dict()) == {
        'dense.weight', 'dense.bias',
        'layer_norm.gamma', 'layer_norm.beta',
        'projection.weight', 'projection.bias',
    }


def test_mlp_parameters_are_trained(features):
    layer = MLPOutputLayer(VOCAB, DIM, hidden_dim=16)
    layer(features).sum().backward()
    for name, param in layer.named_parameters():
        assert param.requires_grad, name
        assert param.grad is not None, name
    # A pure linear map would leave the vocabulary projection's gradient
    # independent of the hidden nonlinearity; check the hidden block trains too.
    assert layer.dense.weight.grad.abs().sum() > 0


def test_mlp_is_not_a_plain_linear_map(features):
    """The gelu + layer-norm block must make the head genuinely nonlinear."""
    layer = MLPOutputLayer(VOCAB, DIM, hidden_dim=16)
    with torch.no_grad():
        doubled = layer(features * 2.0)
        single = layer(features)
    assert not torch.allclose(doubled, single * 2.0, atol=1e-3)


def test_mlp_output_weight_is_untied_from_the_embeddings(embed_tokens, features):
    layer = MLPOutputLayer(VOCAB, DIM, hidden_dim=DIM)
    assert layer.projection.weight is not embed_tokens.weight
    assert layer.projection.weight.shape == embed_tokens.weight.shape
    assert not torch.equal(layer.projection.weight, embed_tokens.weight)


# --- factory ----------------------------------------------------------------

def test_build_output_layer_shared_embed_tied(embed_tokens, features):
    layer = build_output_layer('shared_embed', VOCAB, DIM, embed_tokens=embed_tokens)
    assert isinstance(layer, SharedEmbeddingOutputLayer)
    torch.testing.assert_close(layer(features), F.linear(features, embed_tokens.weight))


def test_build_output_layer_shared_embed_untied(embed_tokens, features):
    embed_out = nn.Parameter(torch.randn(VOCAB, DIM))
    layer = build_output_layer(
        'shared_embed', VOCAB, DIM, embed_tokens=embed_tokens, embed_out=embed_out,
        share_input_output_embed=False)
    torch.testing.assert_close(layer(features), F.linear(features, embed_out))


def test_build_output_layer_mlp_uses_the_configured_hidden_dim(embed_tokens, features):
    layer = build_output_layer(
        'mlp', VOCAB, DIM, embed_tokens=embed_tokens, mlp_hidden_dim=24)
    assert isinstance(layer, MLPOutputLayer)
    assert layer.dense.out_features == 24


def test_build_output_layer_mlp_ignores_share_all_embeddings(embed_tokens, features):
    """``--share-all-embeddings`` forces share_decoder_input_output_embed=True;
    the MLP head must still get its own trained output weights."""
    layer = build_output_layer(
        'mlp', VOCAB, DIM, embed_tokens=embed_tokens, share_input_output_embed=True)
    assert isinstance(layer, MLPOutputLayer)
    assert layer.projection.weight is not embed_tokens.weight


def test_build_output_layer_defaults_need_no_configuration(embed_tokens, features):
    """Every optional setting may be omitted; None means "use the default"."""
    layer = build_output_layer('mlp', VOCAB, DIM, embed_tokens=embed_tokens)
    assert layer.dense.out_features == DIM


def test_build_output_layer_rejects_unknown_head(embed_tokens):
    with pytest.raises(ValueError):
        build_output_layer('lstm', VOCAB, DIM, embed_tokens=embed_tokens)


# --- BertLayerNorm relocation ----------------------------------------------

def test_bert_layer_norm_keeps_its_parameter_names_and_formula():
    """Moving the class must not rename ``gamma``/``beta`` (checkpoint keys) nor
    change the eps-inside-sqrt formula."""
    norm = BertLayerNorm(DIM)
    assert set(norm.state_dict()) == {'gamma', 'beta'}

    x = torch.randn(BATCH, LENGTH, DIM)
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    expected = norm.gamma * ((x - u) / torch.sqrt(s + norm.variance_epsilon)) + norm.beta
    torch.testing.assert_close(norm(x), expected)


def test_bert_layer_norm_is_still_importable_from_the_model_module():
    from fairseq.models.bert_seq2seq import BertLayerNorm as Relocated
    assert Relocated is BertLayerNorm
