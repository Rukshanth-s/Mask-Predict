# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""Selectable decoder output ("decoding") layers.

Every head shares one interface, ``forward(features) -> logits`` mapping
``B x T x C`` decoder states onto ``B x T x V`` vocabulary scores, so the choice
of head is invisible to the rest of the decoder.
"""

import torch.nn as nn
import torch.nn.functional as F

from .bert_layer_norm import BertLayerNorm
from .gelu import gelu

# The heads selectable through ``--decoder-output-layer``.
OUTPUT_LAYER_CHOICES = ['shared_embed', 'mlp', 'crf']

# The heads usable as the CRF's emission scorer (``--crf-emission-layer``).
# A CRF cannot be its own emission scorer, hence the smaller list.
EMISSION_LAYER_CHOICES = ['shared_embed', 'mlp']


class SharedEmbeddingOutputLayer(nn.Module):
    """Project onto the vocabulary with weights owned by the decoder.

    This is the original Mask-Predict head: a lookup against the (shared) token
    embedding matrix, or against the decoder's untied ``embed_out`` parameter.

    The weight belongs to the decoder, not to this module, and it is
    deliberately reached through a closure rather than stored as an attribute:
    assigning a ``Module`` or ``Parameter`` to an attribute would register it
    here and duplicate it under a new name in the state dict, which would break
    loading of the released pre-trained checkpoints. Autograd is unaffected --
    gradients still flow to whoever owns the weight.
    """

    def __init__(self, embed_tokens=None, embed_out=None):
        super().__init__()
        if (embed_tokens is None) == (embed_out is None):
            raise ValueError(
                'SharedEmbeddingOutputLayer needs exactly one weight source, '
                'either embed_tokens (tied) or embed_out (untied)')
        if embed_tokens is not None:
            self._weight = lambda: embed_tokens.weight
        else:
            self._weight = lambda: embed_out

    def forward(self, features):
        return F.linear(features, self._weight())


class MLPOutputLayer(nn.Module):
    """A dedicated, fully trained output head.

    ``Linear -> gelu -> BertLayerNorm -> Linear``, i.e. the BERT language-model
    head, matching the BERT-flavoured decoder this model already uses. Unlike
    :class:`SharedEmbeddingOutputLayer` the vocabulary projection has its own
    weights, so it is trained from scratch and stays untied from the token
    embeddings even under ``--share-all-embeddings``.
    """

    def __init__(self, vocab_size, feature_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = feature_dim if hidden_dim is None else hidden_dim
        self.dense = nn.Linear(feature_dim, hidden_dim)
        self.layer_norm = BertLayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features):
        x = self.layer_norm(gelu(self.dense(features)))
        return self.projection(x)


def build_plain_output_layer(name, vocab_size, feature_dim, embed_tokens=None,
                             embed_out=None, mlp_hidden_dim=None,
                             share_input_output_embed=True):
    """Build one of the two heads that map features straight onto logits."""
    if name == 'shared_embed':
        if share_input_output_embed:
            return SharedEmbeddingOutputLayer(embed_tokens=embed_tokens)
        return SharedEmbeddingOutputLayer(embed_out=embed_out)
    if name == 'mlp':
        return MLPOutputLayer(vocab_size, feature_dim, hidden_dim=mlp_hidden_dim)
    raise ValueError('unknown output layer {!r}, expected one of {}'.format(
        name, EMISSION_LAYER_CHOICES))


def build_output_layer(name, vocab_size, feature_dim, embed_tokens=None, embed_out=None,
                       share_input_output_embed=True, mlp_hidden_dim=None,
                       crf_emission_layer=None, crf_low_rank_dim=None, crf_beam_size=None,
                       crf_dynamic_transition=None, crf_dynamic_hidden_dim=None,
                       crf_inference=None):
    """Build the head named by ``--decoder-output-layer``.

    Takes plain values rather than an ``argparse`` namespace so the module layer
    stays independent of the CLI. Any ``None`` means "use the head's own
    default", which keeps every default in exactly one place.
    """
    if name not in OUTPUT_LAYER_CHOICES:
        raise ValueError('unknown output layer {!r}, expected one of {}'.format(
            name, OUTPUT_LAYER_CHOICES))

    if name != 'crf':
        return build_plain_output_layer(
            name, vocab_size, feature_dim, embed_tokens, embed_out,
            mlp_hidden_dim=mlp_hidden_dim,
            share_input_output_embed=share_input_output_embed,
        )

    # Imported lazily: the CRF head imports this module for its emission head.
    from .dynamic_crf_output_layer import DynamicCRFOutputLayer
    emission = build_plain_output_layer(
        'shared_embed' if crf_emission_layer is None else crf_emission_layer,
        vocab_size, feature_dim, embed_tokens, embed_out,
        mlp_hidden_dim=mlp_hidden_dim,
        share_input_output_embed=share_input_output_embed,
    )
    settings = {
        'low_rank_dim': crf_low_rank_dim,
        'beam_size': crf_beam_size,
        'dynamic_transition': crf_dynamic_transition,
        'dynamic_hidden_dim': crf_dynamic_hidden_dim,
        'inference': crf_inference,
    }
    return DynamicCRFOutputLayer(
        emission_layer=emission, vocab_size=vocab_size, feature_dim=feature_dim,
        **{key: value for key, value in settings.items() if value is not None})
