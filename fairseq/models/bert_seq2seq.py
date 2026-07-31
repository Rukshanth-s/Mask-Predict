# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import (
    FairseqDecoder, FairseqEncoder, FairseqLanguageModel,
    register_model, register_model_architecture,
    FairseqIncrementalDecoder, FairseqModel
)

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveSoftmax, BertLayerNorm, CharacterTokenEmbedder, MultiheadAttention,
    SimpleSinusoidalPositionalEmbedding, LearnedPositionalEmbedding
)
from fairseq.modules.dynamic_crf_output_layer import CRF_INFERENCE_CHOICES
from fairseq.modules.output_layer import (
    EMISSION_LAYER_CHOICES, OUTPUT_LAYER_CHOICES, build_output_layer,
)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def _arg_or_default(args, name, default):
    """Resolve an optional argument to its default.

    An unset argument is normally *absent* from the namespace (fairseq parses
    model args with ``argument_default=argparse.SUPPRESS``), and checkpoints
    predating the argument have no attribute either -- but a plain parser leaves
    it at ``None``. All three mean "use the default".
    """
    value = getattr(args, name, None)
    return default if value is None else value


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx):
    m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx)
    nn.init.normal_(m.weight, mean=0, std=0.02)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

@register_model('bert_transformer_seq2seq')
class Transformer_nonautoregressive(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.zero_()
            module.gamma.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    @staticmethod
    def reference_tokens(prev_output_tokens, target, padding_idx):
        """Recover the complete reference sequence from a CMLM training pair.

        The dataset (fairseq/data/language_pair_self_dataset_mask.py) hands the
        criterion a ``target`` that is padding everywhere except the positions the
        model was asked to predict, while ``prev_output_tokens`` holds the
        reference at every *other* position and <mask> at the predicted ones.
        A per-token loss only needs the former, but a structured loss scores the
        whole sequence, so the two have to be recombined.
        """
        return torch.where(target.ne(padding_idx), target, prev_output_tokens)

    def get_structured_loss(self, net_output, sample):
        """Sequence-level loss of the output layer, or None if it has none.

        Returns one value per sentence.
        """
        output_layer = net_output[1].get('output_layer', None)
        if output_layer is None or not hasattr(output_layer, 'crf_nll'):
            return None

        padding_idx = self.decoder.padding_idx
        prev_output_tokens = sample['net_input']['prev_output_tokens']
        reference = self.reference_tokens(
            prev_output_tokens, sample['target'], padding_idx)
        # The chain spans the whole sequence, not just the masked positions, so
        # the mask comes from the decoder input rather than from the target.
        return output_layer.crf_nll(
            net_output[1]['features'], reference, net_output[1]['output_mask'])

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-enc-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--no-dec-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--embedding-only', default=False, action='store_true',
                            help='if set, replaces the encoder with just token embeddings (could be complex e.g. bilm')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--bilm-model-dropout', default=0.1, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the model dropout for bilm')
        parser.add_argument('--bilm-attention-dropout', default=0.0, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the attention dropout for bilm')
        parser.add_argument('--bilm-relu-dropout', default=0.0, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the relu dropout for bilm')
        parser.add_argument('--bilm-mask-last-state', action='store_true',
                            help='if set, masks last state in bilm as is done during training')
        parser.add_argument('--bilm-add-bos', action='store_true',
                            help='if set, adds bos to input')
        parser.add_argument('--decoder-embed-scale', type=float,
                            help='scaling factor for embeddings used in decoder')
        parser.add_argument('--encoder-embed-scale', type=float,
                            help='scaling factor for embeddings used in encoder')
        parser.add_argument('--decoder-output-layer',
                            choices=OUTPUT_LAYER_CHOICES,
                            help='which head maps decoder states onto the vocabulary: '
                                 'a lookup against the (shared) token embeddings, a dedicated '
                                 'trained MLP, or a structured CRF layer (arXiv:1910.11555)')
        parser.add_argument('--decoder-output-mlp-hidden-dim', type=int, metavar='N',
                            help='hidden dimension of the MLP output head '
                                 '(default: decoder output dimension)')
        parser.add_argument('--crf-emission-layer',
                            choices=EMISSION_LAYER_CHOICES,
                            help='head producing the CRF emission (label) scores')
        parser.add_argument('--crf-low-rank-dim', type=int, metavar='N',
                            help='dimension d_t of the CRF transition embeddings E1/E2')
        parser.add_argument('--crf-beam-size', type=int, metavar='N',
                            help='number of candidates k kept per position by the CRF beam '
                                 'approximation')
        parser.add_argument('--crf-no-dynamic-transition', default=False, action='store_true',
                            help='use a static transition matrix E1 E2^T instead of one '
                                 'conditioned on the adjacent decoder states')
        parser.add_argument('--crf-dynamic-hidden-dim', type=int, metavar='N',
                            help='hidden dimension of the FFN producing the dynamic transition '
                                 '(default: decoder output dimension)')
        parser.add_argument('--crf-nar-loss-weight', type=float, metavar='D',
                            help='weight lambda of the per-token non-autoregressive loss added '
                                 'to the CRF loss')
        parser.add_argument('--crf-inference',
                            choices=CRF_INFERENCE_CHOICES,
                            help='how the CRF produces output tokens and their per-position '
                                 'confidence at generation time')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        #for ds in task.datasets.values():
        #    ds.target_is_source = True

        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, is_encoder, path=None):

            if path is not None:
                if path.startswith('elmo:'):
                    lm_path = path[5:]
                    task = LanguageModelingTask(args, dictionary, dictionary)
                    models, _ = utils.load_ensemble_for_inference([lm_path], task, {'remove_head': True})
                    assert len(models) == 1, 'ensembles are currently not supported for elmo embeddings'

                    embedder = ElmoTokenEmbedder(models[0], dictionary.eos(), dictionary.pad(), add_bos=is_encoder,
                                                 remove_bos=is_encoder, combine_tower_states=is_encoder,
                                                 projection_dim=embed_dim, add_final_predictive=is_encoder,
                                                 add_final_context=is_encoder)
                    return embedder, 1
                elif path.startswith('bilm:'):
                    lm_path = path[5:]
                    task = LanguageModelingTask(args, dictionary, dictionary)
                    models, _ = utils.load_ensemble_for_inference(
                        [lm_path],
                        task,
                        {'remove_head': True,
                         'dropout': args.bilm_model_dropout,
                         'attention_dropout': args.bilm_attention_dropout,
                         'relu_dropout': args.bilm_relu_dropout, })
                    assert len(models) == 1, 'ensembles are currently not supported for elmo embeddings'

                    return BILMEmbedder(models[0], args, args.encoder_embed_dim) if is_encoder \
                        else LMEmbedder(models[0], args.decoder_embed_dim)

            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, is_encoder=True, path=args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, is_encoder=True, path=args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, is_encoder=False, path=args.decoder_embed_path
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, args.encoder_embed_scale)
        decoder = SelfTransformerDecoder(args, tgt_dict, decoder_embed_tokens, args.decoder_embed_scale)
        return Transformer_nonautoregressive(encoder, decoder)



class SelfTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, args, dictionary, embed_tokens, embed_scale=None, no_encoder_attn=False, left_pad=False,
                 final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        self.embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(self.embed_dim) if embed_scale is None else embed_scale

        self.project_in_dim = nn.Linear(input_embed_dim, self.embed_dim,
                                     bias=False) if self.embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, self.embed_dim, self.padding_idx,
            #learned=args.decoder_learned_pos,
        ) if not args.no_dec_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = nn.Linear(self.embed_dim, output_embed_dim, bias=False) \
            if self.embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        self.load_softmax = not getattr(args, 'remove_head', False)

        self.output_layer_type = _arg_or_default(args, 'decoder_output_layer', 'shared_embed')
        self.output_projection = None
        self.embed_out = None

        if self.load_softmax:
            if args.adaptive_softmax_cutoff is not None:
                if self.output_layer_type != 'shared_embed':
                    raise ValueError(
                        '--adaptive-softmax-cutoff replaces the output layer, so it cannot '
                        'be combined with --decoder-output-layer {}'.format(self.output_layer_type))
                self.adaptive_softmax = AdaptiveSoftmax(
                    len(dictionary),
                    output_embed_dim,
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                    dropout=args.adaptive_softmax_dropout,
                    adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                    factor=args.adaptive_softmax_factor,
                    tie_proj=args.tie_adaptive_proj,
                )
            else:
                if not self.share_input_output_embed:
                    self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
                    #nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
                # The head selected by --decoder-output-layer. For 'shared_embed'
                # this wraps the weights above without owning them, so the state
                # dict is byte-for-byte what it was before the flag existed.
                self.output_projection = build_output_layer(
                    self.output_layer_type, len(dictionary), output_embed_dim,
                    embed_tokens=embed_tokens, embed_out=self.embed_out,
                    share_input_output_embed=self.share_input_output_embed,
                    mlp_hidden_dim=getattr(args, 'decoder_output_mlp_hidden_dim', None),
                    crf_emission_layer=getattr(args, 'crf_emission_layer', None),
                    crf_low_rank_dim=getattr(args, 'crf_low_rank_dim', None),
                    crf_beam_size=getattr(args, 'crf_beam_size', None),
                    crf_dynamic_transition=not getattr(
                        args, 'crf_no_dynamic_transition', False),
                    crf_dynamic_hidden_dim=getattr(args, 'crf_dynamic_hidden_dim', None),
                    crf_inference=getattr(args, 'crf_inference', None),
                )
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = BertLayerNorm(self.embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        incremental_state=None
        
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
        ) if self.embed_positions is not None else None

        # embed tokens and positions
        x = self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                decoder_padding_mask,
            )
            inner_states.append(x)
        if self.normalize:
            x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        features = x
        if self.output_projection is not None:
            # project back to size of vocabulary
            x = self.output_projection(features)

        extra = {
            'attn': attn,
            'inner_states': inner_states,
            'predicted_lengths': encoder_out['predicted_lengths'],
        }
        if self.has_structured_output:
            # A structured head scores whole sequences, so the loss and the
            # decoding strategies need the features and the head itself, not just
            # the emission logits above.
            extra['features'] = features
            extra['output_layer'] = self.output_projection
            extra['output_mask'] = ~decoder_padding_mask

        return x, extra

    @property
    def has_structured_output(self):
        """Whether the output head scores sequences rather than tokens."""
        return hasattr(self.output_projection, 'crf_nll')

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        #return min(self.max_target_positions, self.embed_positions.max_positions())
        return self.max_target_positions
    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self,
                       '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        pass


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.
    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, encoder_out, encoder_padding_mask, decoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=decoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        
        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = gelu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``True``
    """

    def __init__(self, args, dictionary, embed_tokens, embed_scale=None, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.eos_idx = dictionary.eos()

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(args.encoder_embed_dim) if embed_scale is None else embed_scale
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            #left_pad=left_pad,
            #learned=args.encoder_learned_pos,
        ) if not args.no_enc_token_positional_embeddings else None
        self.embed_lengths = nn.Embedding(args.max_target_positions, embed_dim)
        nn.init.normal_(self.embed_lengths.weight, mean=0, std=0.02)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = BertLayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_tokens(src_tokens)
        #assert (src_tokens.size(1) < self.embed_positions.weights.data.size(0))
        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)
        #len_tokens = self.embed_lengths(src_tokens.ne(self.padding_idx).sum(-1).unsqueeze(-1))   # If enabled, input of len token is src len
        len_tokens = self.embed_lengths(src_tokens.new(src_tokens.size(0), 1).fill_(0))
        x = torch.cat([len_tokens, x], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        encoder_padding_mask = torch.cat([encoder_padding_mask.new(src_tokens.size(0), 1).fill_(0), encoder_padding_mask], dim=1)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
        if self.normalize:
            x = self.layer_norm(x)
        
        predicted_lengths_logits = torch.matmul(x[0, :, :], self.embed_lengths.weight.transpose(0, 1)).float()
        predicted_lengths_logits[:, 0] += float('-inf')   # Cannot predict the len_token
        predicted_lengths = F.log_softmax(predicted_lengths_logits, dim=-1)
        x = x[1:, :, :]
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask[:, 1:]

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'predicted_lengths': predicted_lengths, # B x L
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['predicted_lengths'] is not None:
            encoder_out['predicted_lengths'] = \
                encoder_out['predicted_lengths'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return self.max_source_positions
        #return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if utils.item(state_dict.get('encoder.version', torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['encoder.version'] = torch.Tensor([1])
        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([BertLayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = gelu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

@register_model_architecture('bert_transformer_seq2seq', 'bert_transformer_seq2seq')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', args.encoder_embed_dim * 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', args.encoder_embed_dim // 64)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', args.encoder_attention_heads)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_enc_token_positional_embeddings = getattr(args, 'no_enc_token_positional_embeddings', False)
    args.no_dec_token_positional_embeddings = getattr(args, 'no_dec_token_positional_embeddings', False)
    args.embedding_only = getattr(args, 'embedding_only', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.decoder_embed_scale = getattr(args, 'decoder_embed_scale', None)
    args.encoder_embed_scale = getattr(args, 'encoder_embed_scale', None)

    # Output ("decoding") layer selection. Defaults reproduce the original
    # tied-embedding head, so checkpoints predating these flags load unchanged.
    args.decoder_output_layer = _arg_or_default(args, 'decoder_output_layer', 'shared_embed')
    args.decoder_output_mlp_hidden_dim = _arg_or_default(
        args, 'decoder_output_mlp_hidden_dim', args.decoder_output_dim)
    # CRF hyperparameters; numeric defaults are the ones used in arXiv:1910.11555.
    args.crf_emission_layer = _arg_or_default(args, 'crf_emission_layer', 'shared_embed')
    args.crf_low_rank_dim = _arg_or_default(args, 'crf_low_rank_dim', 32)
    args.crf_beam_size = _arg_or_default(args, 'crf_beam_size', 64)
    args.crf_no_dynamic_transition = _arg_or_default(args, 'crf_no_dynamic_transition', False)
    args.crf_dynamic_hidden_dim = _arg_or_default(
        args, 'crf_dynamic_hidden_dim', args.decoder_output_dim)
    args.crf_nar_loss_weight = _arg_or_default(args, 'crf_nar_loss_weight', 0.5)
    args.crf_inference = _arg_or_default(args, 'crf_inference', 'viterbi_marginal')

    args.bilm_mask_last_state = getattr(args, 'bilm_mask_last_state', False)
    args.bilm_add_bos = getattr(args, 'bilm_add_bos', False)



@register_model_architecture('bert_transformer_seq2seq', 'bert_transformer_seq2seq_big')
def bi_transformer_lm_big(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    base_architecture(args)
