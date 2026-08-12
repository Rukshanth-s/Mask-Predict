#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""Add a freshly initialised CRF head to a trained non-structured checkpoint.

The CRF output layer is slower to train than the unstructured heads because its
forward-backward recursions are sequential, so arXiv:1910.11555 initialises its
CRF models from a trained non-autoregressive counterpart. Passing such a
checkpoint straight to ``train.py --restore-file`` does not work: the trainer
loads with ``strict=True`` (fairseq/trainer.py) and the checkpoint has none of the
CRF's transition parameters.

This script writes a checkpoint that does load strictly -- every trained weight
is carried over untouched and only the missing transition parameters are added,
initialised the same way the model would initialise them.

    python scripts/init_crf_from_baseline.py \
        --input  ${model_dir}/shared_embed/checkpoint_best.pt \
        --output ${model_dir}/dcrf_init.pt

    python train.py ... --decoder-output-layer crf \
        --restore-file ${model_dir}/dcrf_init.pt \
        --reset-optimizer --reset-lr-scheduler --reset-meters
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

# Run as `python scripts/init_crf_from_baseline.py` from the repo root, Python puts
# scripts/ on sys.path rather than the working directory, so `fairseq` is not
# importable unless the package happens to be pip-installed. Add the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairseq.modules.dynamic_crf_output_layer import DynamicCRFOutputLayer

PREFIX = 'decoder.output_projection.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, metavar='FILE',
                        help='trained checkpoint using shared_embed or mlp')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='where to write the CRF-ready checkpoint')
    parser.add_argument('--crf-emission-layer', default=None,
                        choices=['shared_embed', 'mlp'],
                        help='emission head of the CRF (default: whatever head '
                             'the input checkpoint already uses)')
    parser.add_argument('--crf-low-rank-dim', type=int, default=32)
    parser.add_argument('--crf-beam-size', type=int, default=64)
    parser.add_argument('--crf-no-dynamic-transition', action='store_true')
    parser.add_argument('--crf-dynamic-hidden-dim', type=int, default=None)
    parser.add_argument('--crf-nar-loss-weight', type=float, default=0.5)
    parser.add_argument('--crf-inference', default='viterbi_marginal')
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location='cpu', weights_only=False)
    model_args, weights = checkpoint['args'], checkpoint['model']

    previous_head = getattr(model_args, 'decoder_output_layer', 'shared_embed')
    if previous_head == 'crf':
        raise SystemExit('{} already uses the CRF head'.format(args.input))
    if any(key.startswith(PREFIX + 'transition') for key in weights):
        raise SystemExit('{} already contains CRF transition parameters'.format(args.input))

    # A trained MLP head sits at "<PREFIX>dense.weight" on its own, but inside a
    # CRF the same head is a submodule at "<PREFIX>emission_layer.dense.weight".
    # A shared_embed head owns no parameters, so it needs no such move.
    moved = []
    for key in [k for k in weights if k.startswith(PREFIX)]:
        renamed = PREFIX + 'emission_layer.' + key[len(PREFIX):]
        weights[renamed] = weights.pop(key)
        moved.append((key, renamed))

    vocab_size = weights['decoder.embed_tokens.weight'].size(0)
    feature_dim = getattr(model_args, 'decoder_output_dim', None) \
        or model_args.decoder_embed_dim
    hidden_dim = args.crf_dynamic_hidden_dim or feature_dim

    # Only the transition parameters are needed, so the emission head is stubbed
    # out; the real one is rebuilt from the args at training time.
    head = DynamicCRFOutputLayer(
        emission_layer=nn.Identity(),
        vocab_size=vocab_size,
        feature_dim=feature_dim,
        low_rank_dim=args.crf_low_rank_dim,
        beam_size=args.crf_beam_size,
        dynamic_transition=not args.crf_no_dynamic_transition,
        dynamic_hidden_dim=hidden_dim,
        inference=args.crf_inference,
    )
    # Match Transformer_nonautoregressive.init_bert_weights.
    for module in head.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if getattr(module, 'bias', None) is not None:
                module.bias.data.zero_()

    added = []
    for name, tensor in head.state_dict().items():
        weights[PREFIX + name] = tensor
        added.append(PREFIX + name)

    model_args.decoder_output_layer = 'crf'
    model_args.crf_emission_layer = args.crf_emission_layer or (
        previous_head if previous_head in ('shared_embed', 'mlp') else 'shared_embed')
    model_args.crf_low_rank_dim = args.crf_low_rank_dim
    model_args.crf_beam_size = args.crf_beam_size
    model_args.crf_no_dynamic_transition = args.crf_no_dynamic_transition
    model_args.crf_dynamic_hidden_dim = hidden_dim
    model_args.crf_nar_loss_weight = args.crf_nar_loss_weight
    model_args.crf_inference = args.crf_inference

    # The optimiser state refers to the old parameter set, so drop it; training
    # must be resumed with --reset-optimizer anyway.
    checkpoint['last_optimizer_state'] = None

    torch.save(checkpoint, args.output)
    print('carried over {} trained tensors from {} ({} head)'.format(
        len(weights) - len(added), args.input, previous_head))
    if moved:
        print('re-homed {} emission-head tensors under the CRF:'.format(len(moved)))
        for before, after in moved:
            print('  {}  ->  {}'.format(before, after))
    print('added {} CRF tensors:'.format(len(added)))
    for name in added:
        print('  {}  {}'.format(name, tuple(weights[name].shape)))
    print('wrote {}'.format(args.output))


if __name__ == '__main__':
    main()
