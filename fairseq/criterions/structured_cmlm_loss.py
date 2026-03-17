import math
import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion

@register_criterion('structured_cmlm_loss')
class StructuredCMLMLoss(FairseqCriterion):
    """
    CRF Loss for Structured CMLM with Beam Approximation.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.k = getattr(args, 'crf_beam_size', 64)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # This will allow configuring the beam size from command line.
        parser.add_argument('--crf-beam-size', default=64, type=int, metavar='D',
                            help='Number of candidates to keep for beam approximation')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample."""

        # 1. Model Forward
        net_output = model(**sample['net_input'])
        
        word_ins_out = net_output['word_ins_out'] # Emission logits: [batch_size, seq_len, vocab_size]
        crf_head = net_output['crf_head']
        targets = sample['target'] # [batch_size, seq_len]

        pad_idx = self.padding_idx
        # Boolean mask tracking which positions are padding.
        is_pad = (targets == pad_idx)

        batch_size, seq_len, vocab_size = word_ins_out.shape

        # 2. Beam Truncation
        # Get the top-k candidates and their emission scores.
        # topk_emissions shape: [batch_size, seq_len, k]
        # topk_indices shape: [batch_size, seq_len, k]
        topk_emissions, topk_indices = torch.topk(word_ins_out, self.k, dim=-1)

        # 3. Partition Function Z(X) (Denominator)
        # Initialize alpha with the first word's top-k emissions
        alpha = topk_emissions[:, 0, :] # [batch_size, k]

        # Forward Algorithm (Dynamic Programming)
        for t in range(1, seq_len):
            # Compute transition grid between previous candidates (t-1) and current candidates (t)
            # transitions shape: [batch_size, k, k]
            # crf_head handles the pairwise expansion: E1(prev)^T * E2(curr)
            transitions = crf_head(topk_indices[:, t-1], topk_indices[:, t], pairwise=True)
            
            # Broadcast alpha to match transition grid shape: [batch_size, k, 1]
            # Broadcast current emissions: [batch_size, 1, k]
            prev_alpha = alpha.unsqueeze(2) 
            curr_emissions = topk_emissions[:, t, :].unsqueeze(1)
            
            # The joint score to sum out is: alpha_{t-1}(u) + transition(u, v) + emission_t(v)
            scores = prev_alpha + transitions + curr_emissions # [batch_size, k, k]
            
            # LogSumExp over the previous states (dimension 1, size k)
            new_alpha = torch.logsumexp(scores, dim=1) # [batch_size, k]
            
            # 3.1 The Padding Fix
            # If the current token is a padding token, we should NOT accumulate transitions or emissions.
            # Instead, the alpha should just pass straight through unchanged.
            mask_t = is_pad[:, t].unsqueeze(1) # [batch_size, 1]
            alpha = torch.where(mask_t, alpha, new_alpha)

        # Final Log Partition Function Z(X)
        log_Z = torch.logsumexp(alpha, dim=1) # [batch_size]

        # 4. Numerator (Exact Path Score)
        # Calculate emissions of the exact ground truth token path
        # Gather token emissions using targets. Shape: [batch_size, seq_len]
        # target.unsqueeze(2) makes it [batch_size, seq_len, 1] allowing it to index into vocab_size (dim 2)
        exact_emissions = torch.gather(word_ins_out, dim=2, index=targets.unsqueeze(2)).squeeze(2)
        
        # Calculate transition scores along the exact ground truth token path
        # targets[:, :-1] are prev tokens, targets[:, 1:] are curr tokens
        # Exact transitions output shape: [batch_size, seq_len - 1]
        exact_transitions = crf_head(targets[:, :-1], targets[:, 1:], pairwise=False)

        # Accumulate exact sequence score sequence-wise
        # Start with the first token's emission
        path_score = exact_emissions[:, 0]
        
        for t in range(1, seq_len):
            # Current step adds exact emission and exact transition from t-1 to t
            step_score = exact_emissions[:, t] + exact_transitions[:, t-1]
            
            # The Padding Fix for Numerator
            # Ignore padding accumulations in the exact path
            mask_t = is_pad[:, t]
            path_score = torch.where(mask_t, path_score, path_score + step_score)

        # 5. Loss Calculation
        # CRF Negative Log Likelihood: log Z(X) - path_score
        loss_nll = log_Z - path_score
        
        # Sum batch losses (or average if specified)
        crf_loss = loss_nll.sum() if reduce else loss_nll

        # Length Loss Calculation
        try:
            length_logits = net_output['encoder_out']['predicted_lengths']
            targets_length = targets.ne(self.padding_idx).sum(dim=1)
            length_loss = F.cross_entropy(
                length_logits,
                targets_length,
                reduction='sum' if reduce else 'none'
            )
        except Exception as e:
            doc_id = sample.get('id', 'Unknown')
            raise RuntimeError(f"Length Loss error. Doc ID: {doc_id}. Error: {str(e)}")
            
        loss = crf_loss + length_loss

        # Return the dictionary format required by Fairseq
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        
        logging_output = {
            'loss': loss.data,
            'crf_loss': crf_loss.data,
            'length_loss': length_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        crf_loss_sum = sum(log.get('crf_loss', 0) for log in logging_outputs)
        length_loss_sum = sum(log.get('length_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'crf_loss': crf_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'length_loss': length_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': sum(log.get('nsentences', 0) for log in logging_outputs),
            'sample_size': sample_size,
        }
        return agg_output
