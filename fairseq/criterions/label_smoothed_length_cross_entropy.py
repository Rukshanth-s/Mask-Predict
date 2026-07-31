# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('label_smoothed_length_cross_entropy')
class LabelSmoothedLengthCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        # lambda of "L = L_CRF + lambda * L_NAR" (arXiv:1910.11555 Eq. 11). It is
        # declared by the model, which is built before the criterion.
        self.crf_nar_loss_weight = getattr(args, 'crf_nar_loss_weight', 0.5)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, length_loss, crf_loss, ntokens = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = ntokens #TODO why not merge ntokens and sample_size? what is the difference?
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'length_loss': utils.item(length_loss.data) if reduce else length_loss.data,
            'ntokens': ntokens,
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if crf_loss is not None:
            logging_output['crf_loss'] = utils.item(crf_loss.data) if reduce else crf_loss.data
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        length_lprobs = net_output[1]['predicted_lengths']
        length_target = sample['net_input']['prev_output_tokens'].ne(self.padding_idx).sum(-1).unsqueeze(-1) #TODO doesn't work for dynamic length. change to eos-based method.
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        length_loss = -length_lprobs.gather(dim=-1, index=length_target)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            length_loss = length_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        nar_loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        # A structured output layer scores the target sequence as a whole. When
        # one is in use its loss leads and the per-token term is demoted to an
        # auxiliary objective (arXiv:1910.11555 Eq. 11); otherwise nothing here
        # changes.
        crf_loss = self.compute_structured_loss(model, net_output, sample, reduce=reduce)
        if crf_loss is None:
            loss = nar_loss + length_loss
        else:
            loss = crf_loss + self.crf_nar_loss_weight * nar_loss + length_loss
        return loss, nll_loss, length_loss, crf_loss, non_pad_mask.sum().data.item()

    def compute_structured_loss(self, model, net_output, sample, reduce=True):
        """Sequence-level loss of a structured output layer, or None."""
        get_structured_loss = getattr(model, 'get_structured_loss', None)
        if get_structured_loss is None:
            return None
        structured_loss = get_structured_loss(net_output, sample)
        if structured_loss is None:
            return None
        return structured_loss.sum() if reduce else structured_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        aggregated = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'length_loss': sum(log.get('length_loss', 0) for log in logging_outputs) / nsentences / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if any('crf_loss' in log for log in logging_outputs):
            # A sequence-level loss, so report it per sentence like length_loss.
            aggregated['crf_loss'] = sum(
                log.get('crf_loss', 0) for log in logging_outputs) / nsentences / math.log(2)
        return aggregated
