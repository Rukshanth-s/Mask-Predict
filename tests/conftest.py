# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""Shared test helpers for the decoder output layers."""

import itertools

import torch
import torch.nn as nn


class FixedEmission(nn.Module):
    """An emission head returning preset logits, ignoring its input features.

    Lets a test pin the emission scores exactly and reason about the CRF's
    transition behaviour in isolation.
    """

    def __init__(self, logits):
        super().__init__()
        self.register_buffer('logits', logits)

    def forward(self, features):
        return self.logits


def edge_score(crf, features, batch, position, source_token, target_token):
    """Transition score of one edge, computed the slow, obvious way."""
    e1 = crf.transition_source.weight[source_token]
    e2 = crf.transition_target.weight[target_token]
    if crf.dynamic_transition is None:
        return torch.dot(e1, e2)
    pair = torch.cat([features[batch, position], features[batch, position + 1]], dim=-1)
    dynamic = crf.dynamic_transition(pair).view(crf.low_rank_dim, crf.low_rank_dim)
    return e1 @ dynamic @ e2


def sequence_score(crf, features, emissions, batch, tokens):
    """Unnormalised CRF score of one full token sequence."""
    score = sum(emissions[batch, i, tokens[i]] for i in range(len(tokens)))
    for i in range(len(tokens) - 1):
        score = score + edge_score(crf, features, batch, i, tokens[i], tokens[i + 1])
    return score


def exact_log_partition(crf, features, emissions):
    """Log partition by enumerating every token sequence, with no beam truncation.

    Only usable for toy vocabularies -- it costs ``V ** T`` sequence scorings --
    but it is an oracle that shares no code with the dynamic programs.
    """
    batch_size, length, vocab = emissions.shape
    partitions = []
    for b in range(batch_size):
        scores = [
            sequence_score(crf, features, emissions, b, tokens)
            for tokens in itertools.product(range(vocab), repeat=length)
        ]
        partitions.append(torch.logsumexp(torch.stack(scores), dim=0))
    return torch.stack(partitions)
