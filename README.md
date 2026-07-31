# Mask-Predict


### Download model 
Description | Dataset | Model
---|---|---
MASK-PREDICT | [WMT14 English-German] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_en_de.tar.gz)
MASK-PREDICT | [WMT14 German-English] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_de_en.tar.gz)
MASK-PREDICT | [WMT16 English-Romanian] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_en_ro.tar.gz)
MASK-PREDICT | [WMT16 Romanian-English] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_ro_en.tar.gz)
MASK-PREDICT | [WMT17 English-Chinese] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_en_zh.tar.gz)
MASK-PREDICT | [WMT17 Chinese-English] | [download (.tar.bz2)](http://dl.fbaipublicfiles.com/fairseq/models/maskPredict_zh_en.tar.gz)

### Preprocess

text=PATH_YOUR_DATA

output_dir=PATH_YOUR_OUTPUT

src=source_language

tgt=target_language

model_path=PATH_TO_MASKPREDICT_MODEL_DIR

python preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref $text/train --validpref $text/valid --testpref $text/test  --destdir ${output_dir}/data-bin  --workers 60  --srcdict ${model_path}/maskPredict_${src}_${tgt}/dict.${src}.txt --tgtdict ${model_path}/maskPredict_${src}_${tgt}/dict.${tgt}.txt

### Train


model_dir=PLACE_TO_SAVE_YOUR_MODEL

python train.py ${output_dir}/data-bin --arch bert_transformer_seq2seq --share-all-embeddings --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_self --max-tokens 8192 --weight-decay 0.01 --dropout 0.3 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512  --fp16 --max-source-positions 10000 --max-target-positions 10000 --max-update 300000 --seed 0 --save-dir ${model_dir}

### Choosing the decoding layer

`--decoder-output-layer` selects how the decoder turns its hidden states into
vocabulary scores. It is a training-time flag, stored in the checkpoint, so
generation picks up the same choice automatically.

| Option | Flag | What it does |
| --- | --- | --- |
| 1 | `--decoder-output-layer shared_embed` | **Default.** A lookup against the (shared) token embedding matrix — the original Mask-Predict head. Adds no parameters, so released checkpoints load unchanged. |
| 2 | `--decoder-output-layer mlp` | A dedicated trained head: `Linear → gelu → LayerNorm → Linear`, with its own output weights (untied even under `--share-all-embeddings`). Width via `--decoder-output-mlp-hidden-dim` (default: the decoder output dimension). |
| 3 | `--decoder-output-layer crf` | The structured output layer of [Fast Structured Decoding for Sequence Models](https://arxiv.org/pdf/1910.11555) (Sun et al., NeurIPS 2019): a linear-chain CRF over adjacent target tokens, so tokens are scored jointly rather than independently. |

    # option 2
    python train.py ${output_dir}/data-bin --arch bert_transformer_seq2seq ... --decoder-output-layer mlp

    # option 3
    python train.py ${output_dir}/data-bin --arch bert_transformer_seq2seq ... --decoder-output-layer crf

#### CRF options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--crf-emission-layer` | `shared_embed` | Head producing the CRF's emission (label) scores; `shared_embed` or `mlp`. |
| `--crf-low-rank-dim` | `32` | Rank `d_t` of the factorised transition matrix `M = E1 E2^T`. A full `V x V` matrix is never built. |
| `--crf-beam-size` | `64` | Candidates `k` kept per position. The dynamic programs cost `O(T k^2)` instead of `O(T V^2)`. |
| `--crf-no-dynamic-transition` | off | Off gives the paper's NART-DCRF, whose transition matrix `M^i = E1 f([h_(i-1), h_i]) E2^T` depends on the adjacent decoder states. Set it for the static NART-CRF. |
| `--crf-dynamic-hidden-dim` | decoder output dim | Hidden width of the FFN `f`. |
| `--crf-nar-loss-weight` | `0.5` | `lambda` in `L = L_CRF + lambda * L_NAR`. |
| `--crf-inference` | `viterbi_marginal` | How tokens and per-position confidences are produced (below). |

The CRF trains with `L = L_CRF + lambda * L_NAR + L_length`, where `L_CRF` is the
sequence-level negative log-likelihood and `L_NAR` the existing per-token
label-smoothed cross-entropy. Training logs an extra `crf_loss` column.

Because mask-predict re-masks the least confident positions each iteration, the
CRF has to supply a per-position confidence alongside its tokens.
`--crf-inference` picks how:

* `viterbi_marginal` (default) — Viterbi over the beam chooses the tokens (the
  paper's decoding rule); the forward-backward posterior marginal of the chosen
  token is its confidence.
* `marginal` — the per-position posterior marginal chooses the token and is also
  its confidence.
* `viterbi_emission` — Viterbi chooses the tokens; the confidence is the plain
  emission softmax probability, skipping the backward pass.

This applies to `--decoding-strategy mask_predict`. The `easy_first` and
`left_to_right` strategies run their own beam search over the full vocabulary
rather than asking the head for a decision, so with a CRF model they still work
but use only its emission scores and ignore the transitions.

The mode is also overridable at generation time without retraining:

    python generate_cmlm.py ${output_dir}/data-bin --path ${model_dir}/checkpoint_best_average.pt --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 10 --decoding-strategy mask_predict --model-overrides "{'crf_inference': 'marginal'}"

Two practical notes. The CRF's forward-backward recursions are sequential, so
each update is slower than with the other two heads; the paper warms its CRF
models up from a trained non-structured checkpoint, which works here too since
`shared_embed` and `crf` share the emission head's parameter names. And CRF
confidences are normalised across the `k` beam candidates rather than the whole
vocabulary, so they read higher than emission softmax probabilities — harmless,
since only their relative order within a sentence matters.

### Tests

    python3 -m venv .venv
    .venv/bin/pip install torch "numpy<1.24" pytest
    .venv/bin/python -m pytest tests/

`numpy<1.24` is needed because this fairseq snapshot uses the removed `np.float`
alias.

### Evaluation


python generate_cmlm.py ${output_dir}/data-bin  --path ${model_dir}/checkpoint_best_average.pt  --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 10  --decoding-strategy mask_predict

# License
MASK-PREDICT is CC-BY-NC 4.0.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

```bibtex
@inproceedings{ghazvininejad2019MaskPredict,
  title = {Mask-Predict: Parallel Decoding of Conditional Masked Language Models},
  author = {Marjan Ghazvininejad, Omer Levy, Yinhan Liu, Luke Zettlemoyer},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year = {2019},
}
```
