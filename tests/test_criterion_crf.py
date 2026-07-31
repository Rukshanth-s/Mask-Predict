# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""The training objective: ``L = L_CRF + lambda * L_NAR + L_length``."""

import argparse

import pytest
import torch
import torch.nn as nn

from fairseq.criterions.label_smoothed_length_cross_entropy import (
    LabelSmoothedLengthCrossEntropyCriterion,
)

VOCAB, BATCH, LENGTH, MAX_LEN = 10, 2, 4, 16
PAD = 1


class StubTask(object):
    def __init__(self):
        self.target_dictionary = argparse.Namespace(pad=lambda: PAD)


class StubModel(nn.Module):
    """The smallest model the criterion can talk to.

    ``structured`` is the per-sentence CRF loss the real model would return from
    ``get_structured_loss``, or None for the unstructured heads.
    """

    def __init__(self, net_output, structured=None):
        super().__init__()
        self.net_output = net_output
        self.structured = structured

    def forward(self, **net_input):
        return self.net_output

    def get_normalized_probs(self, net_output, log_probs=True):
        return torch.log_softmax(net_output[0], dim=-1)

    def get_targets(self, sample, net_output):
        return sample['target']

    def get_structured_loss(self, net_output, sample):
        return self.structured


def make_criterion(label_smoothing=0.1, crf_nar_loss_weight=0.5):
    args = argparse.Namespace(
        label_smoothing=label_smoothing,
        crf_nar_loss_weight=crf_nar_loss_weight,
    )
    return LabelSmoothedLengthCrossEntropyCriterion(args, StubTask())


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    logits = torch.randn(BATCH, LENGTH, VOCAB)
    predicted_lengths = torch.log_softmax(torch.randn(BATCH, MAX_LEN), dim=-1)
    net_output = (logits, {'predicted_lengths': predicted_lengths})

    # As produced by language_pair_self_dataset_mask: the target is pad except at
    # the positions the CMLM was asked to predict.
    target = torch.full((BATCH, LENGTH), PAD, dtype=torch.long)
    target[0, 1] = 5
    target[0, 3] = 7
    target[1, 0] = 2
    prev_output_tokens = torch.randint(2, VOCAB, (BATCH, LENGTH))

    sample = {
        'target': target,
        'net_input': {'prev_output_tokens': prev_output_tokens},
    }
    return net_output, sample


def original_loss(net_output, sample, eps):
    """The criterion exactly as it was before the CRF head existed."""
    logits, extra = net_output
    lprobs = torch.log_softmax(logits, dim=-1).view(-1, VOCAB)
    target = sample['target'].view(-1, 1)
    non_pad_mask = target.ne(PAD)
    length_target = sample['net_input']['prev_output_tokens'].ne(PAD).sum(-1).unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask].sum()
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask].sum()
    length_loss = -extra['predicted_lengths'].gather(dim=-1, index=length_target).sum()

    eps_i = eps / VOCAB
    return (1. - eps) * nll_loss + eps_i * smooth_loss + length_loss, nll_loss, length_loss


# --- the unstructured heads must be untouched --------------------------------

def test_loss_is_numerically_unchanged_without_a_structured_head(batch):
    """Regression guard for options 1 and 2: adding the CRF branch must not move
    the loss of a model that does not use it."""
    net_output, sample = batch
    criterion = make_criterion(label_smoothing=0.1)
    loss, sample_size, logging = criterion(StubModel(net_output, structured=None), sample)

    expected, expected_nll, expected_length = original_loss(net_output, sample, 0.1)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.tensor(logging['nll_loss']), expected_nll.detach())
    torch.testing.assert_close(
        torch.tensor(logging['length_loss']), expected_length.detach())
    assert sample_size == int(sample['target'].ne(PAD).sum())


def test_no_crf_loss_is_logged_without_a_structured_head(batch):
    net_output, sample = batch
    _, _, logging = make_criterion()(StubModel(net_output, structured=None), sample)
    assert 'crf_loss' not in logging


def test_a_model_without_the_hook_at_all_still_works(batch):
    """Other models registered in this repo have no ``get_structured_loss``."""
    net_output, sample = batch

    class Bare(nn.Module):
        def forward(self, **net_input):
            return net_output

        def get_normalized_probs(self, net_output, log_probs=True):
            return torch.log_softmax(net_output[0], dim=-1)

        def get_targets(self, sample, net_output):
            return sample['target']

    model = Bare()
    assert not hasattr(model, 'get_structured_loss')

    loss, _, logging = make_criterion()(model, sample)
    expected, _, _ = original_loss(net_output, sample, 0.1)
    torch.testing.assert_close(loss, expected)
    assert 'crf_loss' not in logging


# --- the structured head -----------------------------------------------------

def test_crf_loss_is_added_and_the_nar_term_is_weighted(batch):
    """Paper Eq. 11: ``L = L_CRF + lambda * L_NAR``, plus this repo's length loss."""
    net_output, sample = batch
    structured = torch.tensor([3.0, 4.0])
    criterion = make_criterion(label_smoothing=0.1, crf_nar_loss_weight=0.5)

    loss, _, logging = criterion(StubModel(net_output, structured=structured), sample)

    baseline, _, length_loss = original_loss(net_output, sample, 0.1)
    nar = baseline - length_loss                       # the label-smoothed CE alone
    torch.testing.assert_close(loss, structured.sum() + 0.5 * nar + length_loss)
    torch.testing.assert_close(torch.tensor(logging['crf_loss']), torch.tensor(7.0))


@pytest.mark.parametrize('weight', [0.0, 0.5, 1.0, 2.0])
def test_the_nar_weight_is_honoured(batch, weight):
    net_output, sample = batch
    structured = torch.tensor([1.5, 2.5])
    criterion = make_criterion(label_smoothing=0.1, crf_nar_loss_weight=weight)

    loss, _, _ = criterion(StubModel(net_output, structured=structured), sample)

    baseline, _, length_loss = original_loss(net_output, sample, 0.1)
    nar = baseline - length_loss
    torch.testing.assert_close(loss, structured.sum() + weight * nar + length_loss)


def test_nll_and_length_logging_are_unaffected_by_the_crf_term(batch):
    """The reported nll must stay comparable across the three heads."""
    net_output, sample = batch
    plain = make_criterion()(StubModel(net_output, structured=None), sample)[2]
    crf = make_criterion()(StubModel(net_output, structured=torch.tensor([3.0, 4.0])), sample)[2]
    assert plain['nll_loss'] == crf['nll_loss']
    assert plain['length_loss'] == crf['length_loss']


def test_crf_loss_reaches_the_gradient(batch):
    net_output, sample = batch
    structured = torch.tensor([3.0, 4.0], requires_grad=True)
    loss, _, _ = make_criterion()(StubModel(net_output, structured=structured), sample)
    loss.backward()
    assert structured.grad is not None
    torch.testing.assert_close(structured.grad, torch.ones(2))


def test_sample_size_still_counts_target_tokens(batch):
    net_output, sample = batch
    _, sample_size, logging = make_criterion()(
        StubModel(net_output, structured=torch.tensor([1.0, 1.0])), sample)
    assert sample_size == int(sample['target'].ne(PAD).sum())
    assert logging['sample_size'] == sample_size


# --- logging aggregation -----------------------------------------------------

def test_aggregation_reports_crf_loss_per_sentence():
    aggregated = LabelSmoothedLengthCrossEntropyCriterion.aggregate_logging_outputs([
        {'loss': 10.0, 'nll_loss': 4.0, 'length_loss': 2.0, 'crf_loss': 8.0,
         'ntokens': 4, 'nsentences': 2, 'sample_size': 4},
        {'loss': 20.0, 'nll_loss': 8.0, 'length_loss': 4.0, 'crf_loss': 16.0,
         'ntokens': 8, 'nsentences': 4, 'sample_size': 8},
    ])
    import math
    torch.testing.assert_close(
        aggregated['crf_loss'], 24.0 / 6 / math.log(2))


def test_aggregation_omits_crf_loss_when_no_worker_reported_one():
    aggregated = LabelSmoothedLengthCrossEntropyCriterion.aggregate_logging_outputs([
        {'loss': 10.0, 'nll_loss': 4.0, 'length_loss': 2.0,
         'ntokens': 4, 'nsentences': 2, 'sample_size': 4},
    ])
    assert 'crf_loss' not in aggregated


def test_aggregation_keeps_the_pre_existing_fields():
    import math
    aggregated = LabelSmoothedLengthCrossEntropyCriterion.aggregate_logging_outputs([
        {'loss': 10.0, 'nll_loss': 4.0, 'length_loss': 2.0,
         'ntokens': 4, 'nsentences': 2, 'sample_size': 4},
    ])
    torch.testing.assert_close(aggregated['loss'], 10.0 / 4 / math.log(2))
    torch.testing.assert_close(aggregated['nll_loss'], 4.0 / 4 / math.log(2))
    torch.testing.assert_close(aggregated['length_loss'], 2.0 / 2 / math.log(2))
    assert aggregated['ntokens'] == 4
    assert aggregated['nsentences'] == 2
    assert aggregated['sample_size'] == 4


# --- reconstructing the reference sequence for the CRF ----------------------

def test_model_reconstructs_the_full_reference_sequence():
    """The CMLM target is pad outside the masked positions, so the CRF's gold
    sequence has to be recovered by combining it with the decoder input.

    See fairseq/data/language_pair_self_dataset_mask.py: ``dec_source`` is the
    reference with the selected positions replaced by <mask>, and ``dec_target``
    holds the reference at exactly those positions.
    """
    from fairseq.models.bert_seq2seq import Transformer_nonautoregressive

    prev_output_tokens = torch.tensor([[4, 3, 6, 3, PAD]])   # 3 stands in for <mask>
    target = torch.tensor([[PAD, 8, PAD, 9, PAD]])
    reference = Transformer_nonautoregressive.reference_tokens(
        prev_output_tokens, target, PAD)
    assert torch.equal(reference, torch.tensor([[4, 8, 6, 9, PAD]]))


def test_reference_reconstruction_handles_a_fully_masked_sequence():
    """When every position is masked, the target already is the full reference."""
    from fairseq.models.bert_seq2seq import Transformer_nonautoregressive

    prev_output_tokens = torch.tensor([[3, 3, 3, PAD]])
    target = torch.tensor([[4, 5, 6, PAD]])
    reference = Transformer_nonautoregressive.reference_tokens(
        prev_output_tokens, target, PAD)
    assert torch.equal(reference, torch.tensor([[4, 5, 6, PAD]]))
