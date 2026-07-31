# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""Tests for the ``--decoder-output-layer`` family of training flags."""

import argparse

import pytest

from fairseq.models.bert_seq2seq import Transformer_nonautoregressive, base_architecture


def parse(*argv):
    # Mirror how fairseq itself parses model-specific args: with
    # ``argument_default=argparse.SUPPRESS`` (see fairseq/options.py:107), so
    # unspecified args are absent from the namespace rather than None and
    # ``base_architecture`` is the single source of defaults.
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    Transformer_nonautoregressive.add_args(parser)
    return parser.parse_args(list(argv))


def test_decoder_output_layer_defaults_to_shared_embed():
    args = parse()
    base_architecture(args)
    assert args.decoder_output_layer == 'shared_embed'


@pytest.mark.parametrize('choice', ['shared_embed', 'mlp', 'crf'])
def test_decoder_output_layer_accepts_each_choice(choice):
    args = parse('--decoder-output-layer', choice)
    assert args.decoder_output_layer == choice


def test_decoder_output_layer_rejects_unknown_choice():
    with pytest.raises(SystemExit):
        parse('--decoder-output-layer', 'lstm')


def test_crf_inference_defaults_and_choices():
    args = parse()
    base_architecture(args)
    assert args.crf_inference == 'viterbi_marginal'

    for choice in ['viterbi_marginal', 'marginal', 'viterbi_emission']:
        assert parse('--crf-inference', choice).crf_inference == choice

    with pytest.raises(SystemExit):
        parse('--crf-inference', 'beam')


def test_crf_emission_layer_choices():
    args = parse()
    base_architecture(args)
    assert args.crf_emission_layer == 'shared_embed'
    assert parse('--crf-emission-layer', 'mlp').crf_emission_layer == 'mlp'

    # The CRF emission head is one of the two plain heads, never a nested CRF.
    with pytest.raises(SystemExit):
        parse('--crf-emission-layer', 'crf')


def test_crf_numeric_defaults_match_the_paper():
    args = parse()
    base_architecture(args)
    assert args.crf_low_rank_dim == 32       # d_t
    assert args.crf_beam_size == 64          # k
    assert args.crf_nar_loss_weight == 0.5   # lambda


def test_dynamic_transition_is_on_by_default():
    args = parse()
    base_architecture(args)
    assert args.crf_no_dynamic_transition is False
    assert parse('--crf-no-dynamic-transition').crf_no_dynamic_transition is True


def test_hidden_dims_default_to_decoder_output_dim():
    args = parse('--decoder-embed-dim', '128')
    base_architecture(args)
    assert args.decoder_output_dim == 128
    assert args.decoder_output_mlp_hidden_dim == 128
    assert args.crf_dynamic_hidden_dim == 128


def test_hidden_dims_are_overridable():
    args = parse(
        '--decoder-embed-dim', '128',
        '--decoder-output-mlp-hidden-dim', '256',
        '--crf-dynamic-hidden-dim', '64',
    )
    base_architecture(args)
    assert args.decoder_output_mlp_hidden_dim == 256
    assert args.crf_dynamic_hidden_dim == 64


def test_base_architecture_fills_defaults_on_a_bare_namespace():
    """Old checkpoints carry an ``args`` without any of the new attributes."""
    args = argparse.Namespace()
    base_architecture(args)
    for attr, expected in [
        ('decoder_output_layer', 'shared_embed'),
        ('decoder_output_mlp_hidden_dim', 512),
        ('crf_emission_layer', 'shared_embed'),
        ('crf_low_rank_dim', 32),
        ('crf_beam_size', 64),
        ('crf_no_dynamic_transition', False),
        ('crf_dynamic_hidden_dim', 512),
        ('crf_nar_loss_weight', 0.5),
        ('crf_inference', 'viterbi_marginal'),
    ]:
        assert getattr(args, attr) == expected, attr
