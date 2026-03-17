import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.models.bert_seq2seq import Transformer_nonautoregressive, base_architecture
from fairseq.models.low_rank_crf import LowRankCRFHead

@register_model('structured_cmlm')
class StructuredCMLM(Transformer_nonautoregressive):
    """
    A Structured CMLM that wraps a frozen Mask-Predict backbone and 
    fine-tunes a low-rank CRF transition head.
    """
    def __init__(self, encoder, decoder, vocab_size, d=32):
        super().__init__(encoder, decoder)
        # Instantiate the CRF head 
        self.crf_head = LowRankCRFHead(vocab_size=vocab_size, d=d)

    @classmethod
    def build_model(cls, args, task):
        """
        Builds the model similarly to the base Transformer_nonautoregressive, 
        but instantiates the StructuredCMLM wrapper with the vocabulary size.
        """
        # We can leverage the base class's build_model to get the encoder/decoder
        base_model = Transformer_nonautoregressive.build_model(args, task)
        
        # Grab vocab size from the task target dictionary
        vocab_size = len(task.target_dictionary)
        
        return cls(base_model.encoder, base_model.decoder, vocab_size=vocab_size, d=32)

    def freeze_backbone(self):
        """
        Freezes the CMLM backbone (encoder and decoder) by setting requires_grad = False,
        except for the parameters inside self.crf_head.
        """
        # First, freeze all parameters in the model
        for param in self.parameters():
            param.requires_grad = False
            
        # Then, unfreeze only the parameters in the CRF head
        for param in self.crf_head.parameters():
            param.requires_grad = True

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Overrides the forward method to return a dictionary containing original 
        encoder outputs, emission logits (word_ins_out), and the crf head.
        """
        # Run encoder
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        
        # Run decoder using the encoder outputs
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        
        # decoder_out is typically a tuple: (logits, extra_dictionary)
        word_ins_out = decoder_out[0]
        
        # Return the dictionary format requested by the custom criterion
        return {
            "encoder_out": encoder_out,
            "word_ins_out": word_ins_out,
            "decoder_out": decoder_out,
            "crf_head": self.crf_head
        }

@register_model_architecture('structured_cmlm', 'structured_cmlm')
def base_structured_cmlm_architecture(args):
    base_architecture(args)
