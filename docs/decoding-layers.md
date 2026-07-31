# Decoding layers: training and generation reference

`--decoder-output-layer` selects how the decoder maps its hidden states onto
vocabulary scores. It is a **training-time** flag stored in the checkpoint, so
generation picks the same head up automatically without repeating it.

| Option | Value | Head |
| --- | --- | --- |
| 1 | `shared_embed` (default) | Lookup against the (shared) token embeddings — the original Mask-Predict head |
| 2 | `mlp` | Dedicated trained head: `Linear → gelu → LayerNorm → Linear` |
| 3 | `crf` | Linear-chain CRF over adjacent tokens ([arXiv:1910.11555](https://arxiv.org/pdf/1910.11555)) |

Every command below assumes these shell variables:

```bash
text=PATH_YOUR_DATA
output_dir=PATH_YOUR_OUTPUT
model_dir=PLACE_TO_SAVE_YOUR_MODEL
src=source_language
tgt=target_language
```

The shared part of every training command is collected once here so the
per-option sections stay readable. It must be a bash **array**, not a string: as a
string, `$COMMON` word-splits `--adam-betas '(0.9, 0.999)'` into `'(0.9,` and
`0.999)'` — quotes are not removed after expansion — and training fails on the
malformed argument.

```bash
COMMON=(
  --arch bert_transformer_seq2seq
  --share-all-embeddings
  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1
  --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt
  --warmup-updates 10000
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6
  --task translation_self --max-tokens 8192
  --weight-decay 0.01 --dropout 0.3
  --encoder-layers 6 --encoder-embed-dim 512
  --decoder-layers 6 --decoder-embed-dim 512
  --fp16 --max-source-positions 10000 --max-target-positions 10000
  --max-update 300000 --seed 0
)
```

Always expand it as `"${COMMON[@]}"`, with the quotes.

---

## Preprocess (identical for all three)

```bash
python preprocess.py \
  --source-lang ${src} --target-lang ${tgt} \
  --trainpref $text/train --validpref $text/valid --testpref $text/test \
  --destdir ${output_dir}/data-bin --workers 60 \
  --srcdict ${model_path}/maskPredict_${src}_${tgt}/dict.${src}.txt \
  --tgtdict ${model_path}/maskPredict_${src}_${tgt}/dict.${tgt}.txt
```

---

## Option 1 — shared embeddings (default, baseline)

The flag may be omitted entirely; this reproduces the original recipe exactly and
adds no parameters, so released checkpoints remain loadable.

```bash
python train.py ${output_dir}/data-bin "${COMMON[@]}" \
  --decoder-output-layer shared_embed \
  --save-dir ${model_dir}/shared_embed
```

---

## Option 2 — dedicated MLP head

Its vocabulary projection has its own trained weights, untied from the token
embeddings even under `--share-all-embeddings`.

```bash
# hidden width defaults to the decoder output dimension (512 here)
python train.py ${output_dir}/data-bin "${COMMON[@]}" \
  --decoder-output-layer mlp \
  --save-dir ${model_dir}/mlp
```

```bash
# wider head
python train.py ${output_dir}/data-bin "${COMMON[@]}" \
  --decoder-output-layer mlp \
  --decoder-output-mlp-hidden-dim 1024 \
  --save-dir ${model_dir}/mlp_h1024
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--decoder-output-mlp-hidden-dim` | decoder output dim | Hidden width of the head |

---

## Option 3 — structured CRF head

### 3a. NART-DCRF — dynamic transitions (the paper's best configuration)

Dynamic transitions are **on by default**, so this is the plain `crf` command.
Defaults match the paper: `d_t = 32`, `k = 64`, `lambda = 0.5`.

```bash
python train.py ${output_dir}/data-bin "${COMMON[@]}" \
  --decoder-output-layer crf \
  --save-dir ${model_dir}/dcrf
```

Equivalent with every CRF flag written out explicitly:

```bash
python train.py ${output_dir}/data-bin "${COMMON[@]}" \
  --decoder-output-layer crf \
  --crf-emission-layer shared_embed \
  --crf-low-rank-dim 32 \
  --crf-beam-size 64 \
  --crf-dynamic-hidden-dim 512 \
  --crf-nar-loss-weight 0.5 \
  --crf-inference viterbi_marginal \
  --save-dir ${model_dir}/dcrf
```

### 3b. NART-CRF — static transitions

Drops the FFN conditioning the transition matrix on adjacent decoder states, so
the transition matrix is a plain `M = E1 E2^T`. Fewer parameters, and the paper
reports it slightly weaker than the dynamic variant.

```bash
python train.py ${output_dir}/data-bin "${COMMON[@]}" \
  --decoder-output-layer crf \
  --crf-no-dynamic-transition \
  --save-dir ${model_dir}/crf_static
```

### 3c. CRF on top of an MLP emission head

Combines options 2 and 3: the CRF's emission (label) scores come from a trained
MLP instead of the embedding lookup.

```bash
python train.py ${output_dir}/data-bin "${COMMON[@]}" \
  --decoder-output-layer crf \
  --crf-emission-layer mlp \
  --decoder-output-mlp-hidden-dim 512 \
  --save-dir ${model_dir}/dcrf_mlp
```

### 3d. Warm-starting the CRF from a trained baseline (recommended)

The CRF's recursions are sequential, so each update is slower than with the other
two heads, and arXiv:1910.11555 initialises its CRF models from a trained
non-structured counterpart.

**`--restore-file` cannot do this on its own.** The trainer loads with
`strict=True` (`fairseq/trainer.py:152`) and a baseline checkpoint has none of the
CRF's transition parameters, so it aborts with six missing keys:

```
RuntimeError: Error(s) in loading state_dict for Transformer_nonautoregressive:
  Missing key(s) in state_dict: "decoder.output_projection.transition_source.weight",
  "decoder.output_projection.transition_target.weight",
  "decoder.output_projection.dynamic_transition.0.weight", ...
```

Convert the checkpoint first. `scripts/init_crf_from_baseline.py` carries every
trained tensor over untouched, adds only the transition parameters (initialised
exactly as the model would), and clears the stale optimiser state:

```bash
python scripts/init_crf_from_baseline.py \
  --input  ${model_dir}/shared_embed/checkpoint_best.pt \
  --output ${model_dir}/dcrf_init.pt

python train.py ${output_dir}/data-bin "${COMMON[@]}" \
  --decoder-output-layer crf \
  --restore-file ${model_dir}/dcrf_init.pt \
  --reset-optimizer --reset-lr-scheduler --reset-meters \
  --save-dir ${model_dir}/dcrf_warmstart
```

The `--reset-*` flags are required: the saved optimiser state refers to the old
parameter set. Pass the same CRF hyperparameters to both commands if you change
them from the defaults, e.g. `--crf-low-rank-dim 16 --crf-no-dynamic-transition`.

An `mlp` baseline works as a starting point too, and its trained head is reused
as the CRF's emission scorer (the script re-homes it from
`decoder.output_projection.*` to `decoder.output_projection.emission_layer.*`):

```bash
python scripts/init_crf_from_baseline.py \
  --input  ${model_dir}/mlp/checkpoint_best.pt \
  --output ${model_dir}/dcrf_mlp_init.pt
# --crf-emission-layer mlp is inferred from the input checkpoint
```

### CRF flag summary

| Flag | Default | Meaning |
| --- | --- | --- |
| `--crf-emission-layer` | `shared_embed` | Head producing emission scores; `shared_embed` or `mlp` |
| `--crf-low-rank-dim` | `32` | Rank `d_t` of `M = E1 E2^T`; no `V x V` matrix is built |
| `--crf-beam-size` | `64` | Candidates `k` per position; recursions cost `O(T k^2)` |
| `--crf-no-dynamic-transition` | off (dynamic on) | Set for static NART-CRF |
| `--crf-dynamic-hidden-dim` | decoder output dim | Hidden width of the FFN `f` |
| `--crf-nar-loss-weight` | `0.5` | `lambda` in `L = L_CRF + lambda * L_NAR` |
| `--crf-inference` | `viterbi_marginal` | Decoding rule; see below |

Training a CRF model logs an extra `crf_loss` column alongside `loss`,
`nll_loss` and `length_loss`. It is the sequence-level negative log-likelihood
per sentence and is always `>= 0`.

---

## Averaging checkpoints (all three, unchanged)

```bash
python scripts/average_checkpoints.py \
  --inputs ${model_dir}/dcrf \
  --num-epoch-checkpoints 5 \
  --output ${model_dir}/dcrf/checkpoint_best_average.pt
```

---

## Generation

The head is read from the checkpoint, so the same command serves all three:

```bash
python generate_cmlm.py ${output_dir}/data-bin \
  --path ${model_dir}/<run>/checkpoint_best_average.pt \
  --task translation_self --remove-bpe --max-sentences 20 \
  --decoding-iterations 10 --decoding-strategy mask_predict
```

### Switching the CRF decoding rule without retraining

`--crf-inference` is stored in the checkpoint but overridable at generation time,
because overrides are applied to `args` before the model is built.

```bash
# Viterbi tokens + forward-backward marginal confidence (default)
python generate_cmlm.py ${output_dir}/data-bin \
  --path ${model_dir}/dcrf/checkpoint_best_average.pt \
  --task translation_self --remove-bpe --max-sentences 20 \
  --decoding-iterations 10 --decoding-strategy mask_predict \
  --model-overrides "{'crf_inference': 'viterbi_marginal'}"
```

```bash
# argmax posterior marginal for both token and confidence
  --model-overrides "{'crf_inference': 'marginal'}"
```

```bash
# Viterbi tokens + plain emission softmax confidence (skips the backward pass)
  --model-overrides "{'crf_inference': 'viterbi_emission'}"
```

| Mode | Tokens chosen by | Confidence from |
| --- | --- | --- |
| `viterbi_marginal` | Viterbi over the beam (the paper's rule) | forward-backward marginal of the chosen token |
| `marginal` | argmax posterior marginal per position | that same marginal |
| `viterbi_emission` | Viterbi over the beam | emission softmax probability |

Mask-predict needs a per-position confidence to decide which positions to
re-mask each iteration, which is what these three modes supply. CRF confidences
are normalised across the `k` beam candidates rather than the whole vocabulary,
so they read higher than emission softmax probabilities — harmless, since only
their relative order within a sentence matters.

### Strategy compatibility

| `--decoding-strategy` | Uses CRF structure? |
| --- | --- |
| `mask_predict` | Yes — tokens and confidences come from the CRF |
| `easy_first`, `left_to_right` | No — they run their own beam search over the full vocabulary from the emission logits, so a CRF model still decodes but its transitions are ignored |

---

## Running the tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/
```

`numpy<1.24` is pinned in `requirements-dev.txt` because this fairseq snapshot
uses the `np.float` alias that newer numpy removed.
