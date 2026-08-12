# Training the CRF head on a remote GPU server

End-to-end runbook: SSH in, clone, build a conda environment, smoke-test at toy
scale, then scale up to the full configuration — with checkpointing and resume at
every stage.

This document is about *getting a run to work on a server*. For the meaning of
each `--decoder-output-layer` / `--crf-*` flag, see
[decoding-layers.md](decoding-layers.md).

**Contents**

1. [Fixes applied before this guide](#1-fixes-applied-before-this-guide)
2. [Survey the server](#2-survey-the-server)
3. [Clone and build the environment](#3-clone-and-build-the-environment)
4. [Verify the install](#4-verify-the-install)
5. [Get data — `get_data.sh` is broken, use `scripts/prepare_data.sh`](#5-get-data)
6. [Stage A — toy smoke test (minutes)](#stage-a--toy-smoke-test-minutes)
7. [Stage B — small real model (hours)](#stage-b--small-real-model-hours)
8. [Stage C — full size](#stage-c--full-size)
9. [Checkpoints: resume, retention, averaging, warm-start](#9-checkpoints-resume-retention-averaging-warm-start)
10. [Errors you will hit, and what they mean](#10-errors-you-will-hit-and-what-they-mean)
11. [Sizing `--max-tokens` for the CRF](#11-sizing---max-tokens-for-the-crf)

---

## 1. Fixes applied before this guide

The CRF code passed its 212 unit tests but had never been run end-to-end. Nine
things were broken or wasteful on the training/generation path; all are fixed in
the working tree. **Commit and push these before you clone on the server**, or
you will hit every one of them again.

Four of them (#1, #3, #8, #9) made documented commands fail outright, which is
worth noting about the test suite: it exercises the CRF *modules* thoroughly and
the *pipeline* not at all. Stage A below exists to close that gap — it is a
minutes-long CPU run, not a scale test.

| # | File | Problem | Fix |
| --- | --- | --- | --- |
| 1 | `fairseq/checkpoint_utils.py` | torch ≥ 2.6 flipped `torch.load(weights_only=)` to `True`. Every fairseq checkpoint pickles an `argparse.Namespace` under `'args'`, so **all** checkpoint loading raises `UnpicklingError` — generation, `--restore-file`, resume. Training from scratch works, which is why the tests missed it. | `weights_only=False` |
| 2 | `scripts/average_checkpoints.py` | Same `torch.load` failure, so checkpoint averaging was impossible. | `weights_only=False` |
| 3 | `generate_cmlm.py` | `use_cuda` was computed and then ignored: `models = [model.cuda() ...]` was unconditional, and `generate_batched_itr(..., cuda=True)` took its default. `--cpu` crashed with `Torch not compiled with CUDA enabled`. | honour `use_cuda` in both places |
| 4 | `fairseq/modules/dynamic_crf_output_layer.py` | `crf_nll` called `log_partition` and `gold_score`, each of which independently ran the emission layer — so the `B×T×V` float32 emission matrix was built **twice per step**. At `--max-tokens 8192` and a 32k vocabulary that is 1.0 GiB of avoidable activations. | emissions computed once in `crf_nll` and shared |
| 5 | same | The dynamic-transition FFN was likewise computed twice per step (~10% of the output layer's FLOPs). | computed once and threaded through |
| 6 | same | `build_transitions` documented itself as "always float32" but only cast its *result*: under `--fp16` the low-rank einsums ran in half, risking overflow to `inf`/`NaN` in the transition scores. | inputs upcast before the einsum, as `_prepare` already did for emissions |
| 7 | `fairseq/trainer.py` | A `load_state_dict` failure was re-raised as a bare "please ensure that the architectures match", discarding the missing-key list — i.e. discarding the only information that says *which* flag disagrees. This is exactly the CRF warm-start failure mode. | original exception chained into the message |
| 8 | `scripts/init_crf_from_baseline.py`, `scripts/average_checkpoints.py` | `ModuleNotFoundError: No module named 'fairseq'`. Running `python scripts/foo.py` puts `scripts/` on `sys.path`, not the working directory, so neither script could import `fairseq` — and `average_checkpoints` could not even unpickle a checkpoint. Both are documented in exactly this form in `README.md` and `decoding-layers.md`, so **both documented commands were broken** unless the package happened to be pip-installed. | repo root prepended to `sys.path` in both |
| 9 | `generate_cmlm.py` | `--quiet` crashed with `ValueError: not enough values to unpack (expected 2, got 0)`. Hypotheses were appended to `results` *inside* the `if not args.quiet` branch, so the quiet path scored an empty list — i.e. the one flag you want when you only care about the BLEU number was the one that could not produce it. | collection decoupled from printing; empty-result case reports instead of crashing |

Verification after the changes:

* 212/212 unit tests pass, unchanged. None of the CRF math was altered.
* Toy end-to-end train → checkpoint → resume → warm-start → average → generate →
  BLEU works for both `shared_embed` and `crf`. Resume reloaded at
  `epoch 2 @ 40 updates`; warm-start reloaded and trained on.
* fp16 CRF at `V=32768`: loss stays float32, finite, `>= 0`, max relative error
  `1.3e-4` vs. the float32 path.
* Emission layer and dynamic FFN now run exactly once per training step.
* The warm-start failure mode now reports its missing keys:
  `Missing key(s) in state_dict: "decoder.output_projection.transition_source.weight", ...`

```bash
git add -A && git commit -m "Fix CRF training/generation path for torch 2.x" && git push origin fast-decoding
```

---

## 2. Survey the server

Never write the environment spec before you know what the box is.

```bash
ssh USER@GPU_HOST
```

```bash
nvidia-smi                      # GPU model, VRAM, and the driver version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nproc && free -g                # CPU cores (for --num-workers) and host RAM
df -h ~ /scratch 2>/dev/null    # where is there room for checkpoints?
which conda mamba || ls -d ~/miniconda3 ~/anaconda3 /opt/conda 2>/dev/null
gcc --version                   # only needed if you `pip install -e .`
```

Write down two numbers, they decide everything below:

* **VRAM per GPU** → your `--max-tokens` ceiling ([§11](#11-sizing---max-tokens-for-the-crf)).
* **Driver version** → which CUDA wheel you may install. CUDA 12.x minor-version
  compatibility means any driver ≥ 525 will run the cu12x wheels; a driver in the
  4xx/5xx-early range needs a cu11x wheel instead. Pick the wheel to match the
  driver rather than upgrading the driver on a shared box.

If `conda` is missing:

```bash
curl -fsSLo /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/mc.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash && exec bash
```

---

## 3. Clone and build the environment

Everything below is one script. Save it as `bootstrap.sh` **on the server**, edit
the three variables at the top, then `bash bootstrap.sh`.

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO=https://github.com/YOUR_USER/Mask-Predict.git   # <-- your fork
BRANCH=fast-decoding
CUDA_WHEEL=cu121                                     # <-- match your driver (§2)

WORK=$HOME/maskpredict
mkdir -p "$WORK" && cd "$WORK"

# 1. code
[ -d Mask-Predict ] || git clone --branch "$BRANCH" "$REPO"
cd Mask-Predict

# 2. conda env. Python 3.10 because of the numpy cap below: numpy 1.23.x ships
#    wheels only up to cp311, so on 3.12 `pip install 'numpy<1.24'` fails with
#    "No matching distribution found". 3.11 also works; 3.12 does not.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env remove -n mp -y 2>/dev/null || true
conda create -n mp python=3.10 -y
conda activate mp

# 3. torch with CUDA. Install this FIRST and on its own, so the generic
#    dependency resolution below cannot pull the CPU-only wheel over it.
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"

# 4. the rest. numpy is capped: this snapshot uses the np.float alias that
#    numpy 1.24 removed, and it WILL crash at data-loading time without the cap.
pip install 'numpy<1.24' cffi sacrebleu tqdm pytest subword-nmt sacremoses

# 5. sanity
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA not visible from torch — wrong wheel or no GPU'
print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '|', torch.cuda.device_count(), 'GPU(s):',
      ', '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
PY
echo "environment ready: conda activate mp; cd $WORK/Mask-Predict"
```

Notes on the choices:

* **`pip`, not `conda`, for torch.** The conda `pytorch` channel packages lag and
  drag in their own numpy, which then fights the `<1.24` cap. Conda gives us the
  interpreter; pip gives us the packages.
* **`subword-nmt` and `sacremoses`** are needed by `scripts/prepare_data.sh`
  ([§5](#5-get-data)). Add them to step 4. `gdown` is *not* needed — the only
  thing that used it was `get_data.sh`, which no longer works.
* **Do not `pip install -e .`.** It is not needed — every entry point
  (`train.py`, `generate_cmlm.py`, `preprocess.py`) is run from the repo root and
  imports `fairseq` from the working tree. Installing only adds the `libbleu` C
  extension, which is used by `score.py`/`generate.py`; `generate_cmlm.py` scores
  with the pure-Python `fairseq/pybleu.py` instead. Skipping it removes the
  compiler from your dependency list. If you do want `score.py`, run
  `pip install -e .` and make sure `gcc` is present.
* `numpy<1.24` is not optional. It is the same cap `requirements-dev.txt` carries.

---

## 4. Verify the install

Two checks, both cheap, before you touch any real data.

```bash
conda activate mp && cd ~/maskpredict/Mask-Predict

# the CRF math, inference modes, criterion and arg plumbing
python -m pytest tests/ -q          # expect: 212 passed

# the GPU actually being usable for this model's ops
python - <<'PY'
import torch
x = torch.randn(64, 512, device='cuda', dtype=torch.half)
print('half matmul ok  :', torch.isfinite(x @ x.t()).all().item())
print('logsumexp fp32  :', torch.logsumexp(torch.randn(8, 64, 64, device='cuda'), 1).shape)
print('capability      :', torch.cuda.get_device_capability())
PY
```

Capability `>= (7, 0)` (Volta or newer) means `--fp16` is genuinely fast. Below
that, fairseq prints a warning and fp16 buys you memory but not speed.

---

## 5. Get data

### `get_data.sh` does not work. Do not spend time on it.

This is confirmed, not suspected. Two independent failures:

1. **`gdown` is not a declared dependency**, so the first thing you see is
   `get_data.sh: line 18: gdown: command not found`. Everything after that —
   `tar: Cannot open`, six `mv: cannot stat`, and four `preprocess.py`
   `FileNotFoundError: data/wmt16.en-ro/train.ro` tracebacks — is fallout from
   having no tarball. There is only one error; the rest is noise.
2. **Installing `gdown` does not save it.** Both Google Drive IDs are dead.
   Verified with gdown 6.1.0:

   ```
   1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl   (WMT16 en-ro) -> Cannot retrieve the public link of the file
   0B_bZck-ksdkpM25jRUN2X2UxMm8        (WMT14 en-de) -> Cannot retrieve the public link of the file
   ```

   The second is in the pre-2017 Drive ID format. Neither is coming back.

Two more red herrings in that output: `mkdir: cannot create directory 'data':
File exists` is just a leftover directory from an earlier attempt, and the
`line 10: ... No such file or directory` block at the top is the stray Python
`"""` docstring at the head of a bash file. Neither matters.

### Use `scripts/prepare_data.sh` instead

Added for this reason. It downloads, Moses-tokenises, length-filters, learns a
**joint** BPE, applies it, and binarises — all from sources verified live.

```bash
pip install subword-nmt sacremoses

# 29k pairs de-en. Minutes end to end. Start here.
bash scripts/prepare_data.sh multi30k

# ~400k pairs en-ro (Europarl v8 + newsdev/newstest2016), the paper's benchmark.
bash scripts/prepare_data.sh wmt16-enro
```

Verified locally for `multi30k`: 28332 train / 1014 valid / 1000 test sentences
(668 pairs dropped by the length filter), 9799-type joint BPE vocabulary, 0.0%
`<unk>` on train and 0.007% on test, and `data-bin/multi30k` trains with both
`shared_embed` and `crf`.

The joint BPE is not optional: `--share-all-embeddings` needs one shared
vocabulary across both sides, and `preprocess.py` must be given
`--joined-dictionary` to match. Every recipe here uses both.

### What *is* still alive: the released checkpoints

The `dl.fbaipublicfiles.com` model URLs in `README.md` work. Verified by
downloading `maskPredict_en_ro.tar.gz`:

```
maskPredict_en_ro/checkpoint_best.pt   0.93 GiB   6 layers, 512 dim, vocab 34984
maskPredict_en_ro/dict.en.txt
maskPredict_en_ro/dict.ro.txt
```

It loads cleanly with fix #1, and because it pre-dates `--decoder-output-layer`
that arg is simply absent from its `args`; `base_architecture` defaults it to
`shared_embed`, so the released model still builds against the new plumbing.
That is the backward-compatibility claim in `decoding-layers.md`, confirmed
against the real artifact.

**But the tarball contains no BPE codes** — only the dictionaries. So you cannot
reproduce the exact segmentation those dictionaries assume, which means you
cannot preprocess fresh text into that vocabulary, which means you cannot
warm-start the CRF from this released checkpoint. Train your own baseline on your
own BPE ([Stage C](#stage-c--full-size)) and warm-start from that instead.

The 0.93 GiB is also your calibration for
[retention](#9-checkpoints-resume-retention-averaging-warm-start): that is what
one Stage C checkpoint costs, and fairseq keeps one per epoch by default.

---

## Stage A — toy smoke test (minutes)

The point of this stage is to prove the whole pipeline runs — data → train →
checkpoint → resume → average → generate — before spending GPU hours. It uses
synthetic data and a model small enough to finish on a CPU.

The toy task is "copy the source, uppercased": trivially learnable, so a model
that *fails* to learn it has a bug, not a tuning problem.

```bash
mkdir -p toy && python - <<'PY'
import random
random.seed(0)
words = ['the','cat','sat','on','mat','dog','ran','fast','big','red','blue','house',
         'tree','sun','moon','sky','run','walk','eat','play']
for split, n in [('train', 2000), ('valid', 100), ('test', 100)]:
    src = [' '.join(random.choice(words) for _ in range(random.randint(3, 12))) for _ in range(n)]
    open(f'toy/{split}.en', 'w').write('\n'.join(src) + '\n')
    open(f'toy/{split}.de', 'w').write('\n'.join(s.upper() for s in src) + '\n')
print('wrote toy/')
PY

python preprocess.py --source-lang en --target-lang de \
  --trainpref toy/train --validpref toy/valid --testpref toy/test \
  --destdir data-bin/toy --joined-dictionary --workers 4
```

A tiny architecture. Note that overriding `--encoder-embed-dim` means you must
also override `--encoder-ffn-embed-dim` and `--encoder-attention-heads`:
`base_architecture` defaults them to the 512-dim values (2048 / 8), and 8 heads
do not divide a 64-dim model.

```bash
TINY=(
  --arch bert_transformer_seq2seq --share-all-embeddings
  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1
  --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9
  --lr-scheduler inverse_sqrt --warmup-updates 200
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6
  --task translation_self --max-tokens 2048
  --weight-decay 0.01 --dropout 0.1
  --encoder-layers 2 --encoder-embed-dim 64 --encoder-ffn-embed-dim 128 --encoder-attention-heads 2
  --decoder-layers 2 --decoder-embed-dim 64 --decoder-ffn-embed-dim 128 --decoder-attention-heads 2
  --max-source-positions 128 --max-target-positions 128
  --max-update 1500 --seed 0 --keep-last-epochs 5 --no-progress-bar --log-interval 100
)
```

Expand it as `"${TINY[@]}"`, **with the quotes** — as a bare string,
`--adam-betas '(0.9, 0.999)'` word-splits into `'(0.9,` and `0.999)'` and
training dies on the malformed argument.

Baseline first, so you have something to compare the CRF against:

```bash
python train.py data-bin/toy "${TINY[@]}" --save-dir ckpt/toy_base
python train.py data-bin/toy "${TINY[@]}" \
  --decoder-output-layer crf --crf-beam-size 8 --crf-low-rank-dim 8 \
  --save-dir ckpt/toy_crf
```

`--crf-beam-size 8` because the toy vocabulary is only ~48 types; the default
`64` would silently clamp to the vocabulary size (`build_beam` takes
`min(beam_size, V)`) and you would be testing the exact-CRF path rather than the
beam-approximated one that runs at real scale.

Then generate from both:

```bash
for run in toy_base toy_crf; do
  echo "=== $run ==="
  python generate_cmlm.py data-bin/toy \
    --path ckpt/$run/checkpoint_best.pt --task translation_self \
    --max-sentences 20 --decoding-iterations 4 \
    --decoding-strategy mask_predict --quiet
done
```

### What a healthy Stage A looks like

* A `crf_loss` column appears in the CRF run's log and **is never negative** —
  the gold path is forced into the beam precisely to guarantee that. A negative
  `crf_loss` means the forcing broke and is a real bug, not a tuning issue.
* `nll_loss` falls steadily in both runs.
* Both runs reach a high BLEU. Reference numbers measured on this setup (CPU,
  `mask_predict`, 4 decoding iterations):

  | Head | updates | BLEU4 |
  | --- | --- | --- |
  | `shared_embed` | 864 | 93.8 |
  | `crf` | 972 | 99.6 |

  Not a controlled comparison — the update counts differ — but it establishes
  that the CRF path trains, checkpoints, decodes and scores correctly, and is not
  *worse* than the baseline. If your CRF run lands far below the baseline, stop
  and investigate before scaling.

* **Do not read a speed penalty off this stage.** Measured here, 100 updates took
  25 s with `shared_embed` and 26 s with `crf` — about 4%. That is misleadingly
  cheap: the toy vocabulary is 48 types and `k=8`, so both the emission
  projection and the `O(T k²)` recursions are trivial.

  On real data (`multi30k`, `V=9799`, default `k=64`, CPU, 2×128 model) the same
  comparison gives:

  | Head | wps | wall for 20 updates |
  | --- | --- | --- |
  | `shared_embed` | 987 | 8 s |
  | `crf`, `k=64` | 654 | 13 s |
  | `crf`, `k=16` | 729 | 11 s |

  So budget roughly **1.5–1.6× the baseline's step time** for the CRF. Re-measure
  on your GPU at your vocabulary — this is CPU and a small model, so treat it as
  indicative of the ratio, not the absolute.

### If generation prints nothing but `<mask>`

This is expected from an undertrained model and is **not** a CRF bug. Unlike
`easy_first`, which explicitly zeroes the mask token's probability
(`easy_first.py`, `candidate_probs[:, :, mask] = 0`), the `mask_predict` strategy
never excludes `<mask>` from its own predictions. `<mask>` is very frequent in
the decoder *input* and, under `--share-all-embeddings`, shares its embedding
with the output projection — so an early model happily predicts it. It
disappears as training progresses. Seeing it after a full Stage A run *is* a
signal worth chasing.

---

## Stage B — small real model (hours)

Real data, real vocabulary, still small enough to iterate on. This is the stage
that tells you whether the CRF helps *on your data*, and it is where you
calibrate `--max-tokens` before committing to Stage C.

Run it on `multi30k` first — 29k pairs and a 9799-type vocabulary, so a baseline
and a CRF run both finish in well under an hour on one GPU. That is enough to see
a real BLEU difference between the two heads and to shake out anything the toy
stage could not. Then repeat on `wmt16.en-ro` with the same commands and a longer
`--max-update`.

```bash
DATA=data-bin/multi30k          # or data-bin/wmt16.en-ro
SMALL=(
  --arch bert_transformer_seq2seq --share-all-embeddings
  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1
  --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9
  --lr-scheduler inverse_sqrt --warmup-updates 4000
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6
  --task translation_self
  --weight-decay 0.01 --dropout 0.3
  --encoder-layers 4 --encoder-embed-dim 256 --encoder-ffn-embed-dim 1024 --encoder-attention-heads 4
  --decoder-layers 4 --decoder-embed-dim 256 --decoder-ffn-embed-dim 1024 --decoder-attention-heads 4
  --fp16 --max-source-positions 10000 --max-target-positions 10000
  --max-update 30000 --seed 0
  --max-tokens 4096
  --num-workers 4 --log-interval 100 --no-progress-bar
  --keep-last-epochs 5 --save-interval-updates 2000 --keep-interval-updates 3
)

nohup python train.py $DATA "${SMALL[@]}" \
  --decoder-output-layer crf \
  --save-dir ckpt/small_crf > logs/small_crf.log 2>&1 &
```

Watch it, and watch for the failure signatures too — not just progress:

```bash
mkdir -p logs
tail -f logs/small_crf.log | grep -E "num_updates|crf_loss|Traceback|out of memory|NaN|inf|overflow"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 10
```

Three things to confirm in the first ~15 minutes:

1. **Memory has headroom.** `nvidia-smi` should show meaningful free VRAM. If
   you are at 95%+, drop `--max-tokens` now — an OOM 6 hours in wastes 6 hours.
2. **The loss scaler settles.** Occasional `overflow detected, setting loss scale
   to ...` lines early on are normal fp16 behaviour. A scale that keeps halving
   toward `min_loss_scale` and then aborts is not — see
   [§10](#10-errors-you-will-hit-and-what-they-mean).
3. **`crf_loss` is decreasing and non-negative.**

### Multi-GPU

`train.py` spawns one process per GPU by itself; there is no `torchrun` involved.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py $DATA "${SMALL[@]}" \
  --decoder-output-layer crf --save-dir ckpt/small_crf
```

`--max-tokens` is **per GPU**, and the effective batch is
`max_tokens × n_gpus × update_freq`. When you change the GPU count, hold the
effective batch fixed with `--update-freq` — otherwise you have changed the
learning-rate schedule without meaning to, and the run is not comparable to the
one before it.

---

## Stage C — full size

The published recipe, plus the operational flags Stage B taught you.

```bash
FULL=(
  --arch bert_transformer_seq2seq --share-all-embeddings
  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1
  --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9
  --lr-scheduler inverse_sqrt --warmup-updates 10000
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6
  --task translation_self
  --weight-decay 0.01 --dropout 0.3
  --encoder-layers 6 --encoder-embed-dim 512
  --decoder-layers 6 --decoder-embed-dim 512
  --fp16 --max-source-positions 10000 --max-target-positions 10000
  --max-update 300000 --seed 0
  --max-tokens 4096 --update-freq 2
  --num-workers 8 --log-interval 200 --no-progress-bar
  --keep-last-epochs 10 --save-interval-updates 5000 --keep-interval-updates 5
)
```

At 512 dim the ffn/head defaults (2048 / 8) are already correct, so they are
omitted — this is the one size where you may leave them out.

**Train the baseline first, then warm-start the CRF from it.** This is what
arXiv:1910.11555 does, and it is the difference between a CRF run that converges
and one that spends its budget relearning what a plain softmax head already
knew.

```bash
# 1. non-structured baseline
nohup python train.py $DATA "${FULL[@]}" \
  --decoder-output-layer shared_embed \
  --save-dir ckpt/full_base > logs/full_base.log 2>&1 &

# 2. graft a fresh CRF head onto it (see §9 for why --restore-file alone fails)
python scripts/init_crf_from_baseline.py \
  --input  ckpt/full_base/checkpoint_best.pt \
  --output ckpt/full_dcrf_init.pt

# 3. CRF run, warm-started
nohup python train.py $DATA "${FULL[@]}" \
  --decoder-output-layer crf \
  --restore-file "$PWD/ckpt/full_dcrf_init.pt" \
  --reset-optimizer --reset-lr-scheduler --reset-meters --reset-dataloader \
  --save-dir ckpt/full_dcrf > logs/full_dcrf.log 2>&1 &
```

Then average and evaluate:

```bash
python scripts/average_checkpoints.py \
  --inputs ckpt/full_dcrf --num-epoch-checkpoints 5 \
  --output ckpt/full_dcrf/checkpoint_best_average.pt

python generate_cmlm.py $DATA \
  --path ckpt/full_dcrf/checkpoint_best_average.pt \
  --task translation_self --remove-bpe --max-sentences 20 \
  --decoding-iterations 10 --decoding-strategy mask_predict
```

Comparisons worth running once the CRF model exists — all three read the head
from the checkpoint, and `--crf-inference` is overridable without retraining:

```bash
for mode in viterbi_marginal marginal viterbi_emission; do
  echo -n "$mode  "
  python generate_cmlm.py $DATA \
    --path ckpt/full_dcrf/checkpoint_best_average.pt \
    --task translation_self --remove-bpe --max-sentences 20 \
    --decoding-iterations 10 --decoding-strategy mask_predict \
    --model-overrides "{'crf_inference': '$mode'}" --quiet | grep -o "BLEU4 = [0-9.]*"
done
```

On the Stage A toy model this sweep gave `viterbi_marginal` 99.6, `marginal`
99.6, `viterbi_emission` 82.8. The gap is the point: `viterbi_emission` picks the
same tokens as `viterbi_marginal` but ranks its confidences with the plain
emission softmax, and mask-predict uses exactly those confidences to choose which
positions to re-mask. Skipping the backward pass saves time and costs accuracy.
Worth re-measuring on your data before you trade it away.

---

## 9. Checkpoints: resume, retention, averaging, warm-start

### Resume after a crash or a preemption

`train.py` looks for `checkpoint_last.pt` inside `--save-dir` automatically.
**Re-run the identical command** — same flags, same `--save-dir` — and it picks
up the model, optimiser, LR schedule, epoch and dataloader position:

```bash
nohup python train.py $DATA "${FULL[@]}" --decoder-output-layer crf \
  --save-dir ckpt/full_dcrf >> logs/full_dcrf.log 2>&1 &
# | loaded checkpoint ckpt/full_dcrf/checkpoint_last.pt (epoch 12 @ 44000 updates)
```

There is no separate resume flag, and no `--restore-file` needed for this case.

### `--restore-file` is resolved relative to `--save-dir`

`checkpoint_utils.load_checkpoint` joins a non-absolute `--restore-file` onto
`--save-dir`. A relative path therefore silently looks in the wrong place and,
finding nothing, starts from scratch — a very expensive typo. **Always pass an
absolute path** (`"$PWD/..."` is enough).

### Retention — this will fill your disk

fairseq writes one checkpoint per epoch and keeps them all by default. At the
Stage C size each is roughly 1–2 GB, and a 300k-update run spans many epochs.

| Flag | Effect |
| --- | --- |
| `--keep-last-epochs 10` | keep only the 10 most recent epoch checkpoints |
| `--save-interval-updates 5000` | also snapshot every 5000 updates |
| `--keep-interval-updates 5` | keep only the 5 most recent of those |
| `--no-epoch-checkpoints` | epoch checkpoints off entirely — **do not use**, it leaves `average_checkpoints.py --num-epoch-checkpoints` nothing to average |
| `--no-save-optimizer-state` | smaller files, but they can no longer resume training |

`checkpoint_best.pt` and `checkpoint_last.pt` are always kept.

### Warm-starting the CRF from a non-structured baseline

`--restore-file` alone **cannot** do this. The trainer loads with `strict=True`,
and a `shared_embed`/`mlp` checkpoint has none of the CRF's transition
parameters, so it aborts on missing keys:

```
Missing key(s) in state_dict: "decoder.output_projection.transition_source.weight",
  "decoder.output_projection.transition_target.weight",
  "decoder.output_projection.dynamic_transition.0.weight", ...
```

(Thanks to fix #7 you now actually *see* that list instead of a bare
"architectures match" message.)

`scripts/init_crf_from_baseline.py` carries every trained tensor over untouched,
adds only the transition parameters — initialised exactly as the model would —
re-homes an `mlp` head to `output_projection.emission_layer.*`, and drops the
stale optimiser state. Then:

```bash
--restore-file "$PWD/ckpt/full_dcrf_init.pt" \
--reset-optimizer --reset-lr-scheduler --reset-meters --reset-dataloader
```

All four resets matter:

* `--reset-optimizer` — the saved optimiser state refers to the old parameter
  set. Required.
* `--reset-lr-scheduler` — otherwise you resume mid-decay and the new CRF
  parameters never get a warmup.
* `--reset-meters` — cosmetic; keeps the logged averages honest.
* `--reset-dataloader` — **easy to forget.** The init script preserves
  `extra_state`, so without this the warm-start run resumes at the baseline's
  epoch number and dataloader position instead of starting a fresh pass.

If you changed any CRF hyperparameter from its default, pass it to *both* the
init script and `train.py`, or the shapes will not match.

### Why warm-starting is worth the extra step

Measured on the Stage A toy setup, both at 20 updates — from scratch versus
warm-started from a trained `shared_embed` baseline:

| | `nll_loss` | `crf_loss` |
| --- | --- | --- |
| CRF from scratch | 5.65 | 27.6 |
| CRF warm-started | 0.21 | 2.5 |

The warm-started run begins where the baseline left off and spends its budget
learning the transitions, which are the only thing it does not already have. The
from-scratch run spends most of its early budget relearning the emission head
through a slower objective.

---

## 10. Errors you will hit, and what they mean

| Symptom | Cause | Action |
| --- | --- | --- |
| `UnpicklingError: Weights only load failed ... GLOBAL argparse.Namespace` | torch ≥ 2.6 default | fix #1/#2. If you cloned before committing them, apply `weights_only=False` in `fairseq/checkpoint_utils.py` and `scripts/average_checkpoints.py` |
| `AttributeError: module 'numpy' has no attribute 'float'` | numpy ≥ 1.24 | `pip install 'numpy<1.24'` |
| `ModuleNotFoundError: No module named 'fairseq'` from a `scripts/...` command | fix #8 | if unfixed, prefix with `PYTHONPATH=.` or use `python -m scripts.average_checkpoints` |
| `AssertionError: Torch not compiled with CUDA enabled` during generation | fix #3, or a CPU-only torch wheel | check `torch.version.cuda` is not `None`; reinstall from the CUDA index URL |
| `Cannot load model parameters from checkpoint ... Missing key(s) ... transition_source` | warm-starting a CRF run from a non-CRF checkpoint | run `scripts/init_crf_from_baseline.py` first ([§9](#9-checkpoints-resume-retention-averaging-warm-start)) |
| `Unexpected key(s) ... transition_source` | the reverse: a CRF checkpoint into a run without `--decoder-output-layer crf` | add the flag; it is a training-time flag stored in the checkpoint |
| `embed_dim must be divisible by num_heads` | you shrank `--*-embed-dim` but not `--*-attention-heads` | override ffn dim and head count together ([Stage A](#stage-a--toy-smoke-test-minutes)) |
| `--share-all-embeddings requires a joined dictionary` | preprocessed without `--joined-dictionary` | re-run `preprocess.py` with it |
| `CUDA out of memory` | `--max-tokens` too high for the CRF path | halve `--max-tokens`, restore the effective batch with `--update-freq` ([§11](#11-sizing---max-tokens-for-the-crf)) |
| `Minimum loss scale reached (0.0001)` then abort | fp16 loss-scale collapse | fix #6 addresses the CRF-side overflow. If it persists: `--fp16-scale-window 256`, or lower `--lr`, or drop `--fp16` for the CRF run |
| `crf_loss` is negative | gold path not inside the beam — a genuine bug | stop; do not tune around it |
| `crf_loss` climbing while `nll_loss` falls | `--crf-nar-loss-weight` too high, so the per-token term dominates | lower it toward 0.1–0.5 |
| Output is all `<mask>` | undertrained model, not a bug | train longer; see [Stage A](#stage-a--toy-smoke-test-minutes) |
| `ValueError: not enough values to unpack (expected 2, got 0)` from `generate_cmlm.py --quiet` | fix #9 | if unfixed, drop `--quiet` |
| `command not found` noise from `get_data.sh` | stray Python docstring at the top of a bash file | ignore |
| `gdown: command not found`, then `tar: Cannot open`, then `preprocess.py FileNotFoundError` | `get_data.sh`: `gdown` is undeclared, and installing it does not help — both Drive IDs are dead | use `bash scripts/prepare_data.sh multi30k` ([§5](#5-get-data)) |
| `Cannot retrieve the public link of the file` from `gdown` | the 2019 Drive IDs in `get_data.sh` no longer resolve | same as above |
| `mkdir: cannot create directory 'data': File exists` | leftover from a previous failed `get_data.sh` | ignore |
| Log says `training on 1 GPUs` under `--cpu` | cosmetic; `distributed_world_size` is printed regardless | ignore |
| CRF far slower per update than baseline | expected — the forward-backward recursions are sequential in `T` | warm-start ([§9](#9-checkpoints-resume-retention-averaging-warm-start)); lower `--crf-beam-size` |

### A note on CRF confidences

They are normalised across the `k` beam candidates, not the whole vocabulary, so
they read higher than emission softmax probabilities. This is harmless: mask-predict
only compares them *within* a sentence to choose which positions to re-mask.
Do not read them as calibrated probabilities.

---

## 11. Sizing `--max-tokens` for the CRF

The CRF's output layer is the memory bottleneck, and it scales with
`max_tokens × V`, independent of model depth. Per training step it holds, in
float32:

| Tensor | Size | At `max_tokens=8192`, `V=32768` |
| --- | --- | --- |
| emission matrix (`_prepare`) | `max_tokens × V × 4 B` | 1.0 GiB |
| criterion's `log_softmax` | `max_tokens × V × 4 B` | 1.0 GiB |
| decoder logits (fp16) | `max_tokens × V × 2 B` | 0.5 GiB |
| beam transitions + DP intermediates | `≈ 3 × max_tokens × k² × 4 B` | ~0.4 GiB at `k=64` |

Roughly `max_tokens × V × 10 bytes` plus the `k²` term, all retained for
backward — on top of the encoder/decoder activations and the optimiser state.
Before fix #4 the emission matrix was built twice, adding another 1.0 GiB.

Starting points, to confirm against `nvidia-smi` rather than trust:

| VRAM | `--max-tokens` (CRF) | `--update-freq` to match an 8192-token batch |
| --- | --- | --- |
| 16 GB | 2048 | 4 |
| 24 GB | 4096 | 2 |
| 40 GB | 8192 | 1 |
| 80 GB | 8192–16384 | 1 |

The non-CRF heads tolerate roughly 2× these values.

Other knobs, in the order worth trying:

1. **`--update-freq`** — the free one. Halve `--max-tokens`, double
   `--update-freq`, and the effective batch and LR schedule are unchanged.
2. **`--crf-beam-size`** — the DP costs `O(T k²)`, so this is a real *memory*
   lever. It is a weaker *speed* lever than the `k²` suggests, because the DP is
   only part of the step. Measured at `V=9799` (CPU, small model), dropping
   `k` 64 → 16 moved throughput 654 → 729 wps: about 15%, not 4×. Shrink `k` to
   fit memory, not to buy speed.
3. **`--crf-low-rank-dim`** — 32 → 16 shrinks the transition embeddings and the
   dynamic FFN's output. Cheap to try.
4. **`--crf-no-dynamic-transition`** — removes the FFN entirely (the paper's
   static NART-CRF). Reported slightly weaker than the dynamic variant.
5. **`--memory-efficient-fp16`** — last resort; slower.

Do not shrink `--max-tokens` far below ~1024. The length-prediction loss is
averaged per sentence, and very small batches make it noisy.
