#!/usr/bin/env bash
#
# IWSLT14 de-en, prepared by running fairseq's OWN script unmodified.
#
#   bash scripts/prepare_iwslt14.sh
#
# ~160k sentence pairs of TED talk transcripts, ~20 words per sentence. Five times
# Multi30k with much longer sentences, which is the regime where predicting every
# token independently starts to hurt -- i.e. where a CRF has something to fix.
#
# This wrapper does NOT reimplement the recipe. It runs
# scripts/third_party/prepare-iwslt14.sh, a verbatim copy of fairseq v0.12.2's
# script (see that directory's README), and only supplies what the script assumes
# is already present:
#
#   1. `python` on PATH        - the script calls `python learn_bpe.py`, and many
#                                systems (including this GPU server) have only
#                                python3. A shim directory is prepended to PATH.
#   2. shallow git clones      - the script does full clones of mosesdecoder and
#                                subword-nmt. mosesdecoder is ~130 MB of history
#                                for five files; --depth 1 transfers ~16 MB.
#                                Pre-creating the directories makes the script's
#                                own clone a harmless no-op.
#   3. the corpus              - pre-downloaded with curl and md5-checked, so the
#                                script's `wget` is not required.
#
# Then it binarises with this repo's preprocess.py using --joined-dictionary,
# which --share-all-embeddings requires.
#
# Requires: perl, git, curl, and a python3 with subword-nmt importable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRIPT=scripts/third_party/prepare-iwslt14.sh
WORK=data/iwslt14_official
OUT=data-bin/iwslt14.de-en
URL=http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz
MD5_EXPECTED=10936dfdf616504ddbd844e91b94ef4a

[ -f "$SCRIPT" ] || { echo "missing $SCRIPT"; exit 1; }
for c in perl git curl md5sum; do
  command -v "$c" >/dev/null || { echo "need '$c' on PATH"; exit 1; }
done

PY="${PYTHON:-$(command -v python3 || command -v python)}"
[ -n "$PY" ] || { echo "no python3 or python on PATH"; exit 1; }
"$PY" -c "import subword_nmt" 2>/dev/null || echo "note: subword-nmt not importable, but the script uses its own clone"
echo "interpreter: $PY ($("$PY" -V 2>&1))"

mkdir -p "$WORK"

# --- 1. `python` shim, so the script's `python learn_bpe.py` works ------------
SHIM="$ROOT/$WORK/.shim"
mkdir -p "$SHIM"
if [ ! -x "$SHIM/python" ]; then
  printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "$PY" > "$SHIM/python"
  chmod +x "$SHIM/python"
fi
export PATH="$SHIM:$PATH"
echo "python -> $(command -v python) ($(python -V 2>&1))"

cd "$WORK"

# --- 2. shallow clones, so the script's full clones become no-ops ------------
[ -d mosesdecoder ] || { echo "shallow-cloning mosesdecoder (~16 MB)..."; git clone --depth 1 -q https://github.com/moses-smt/mosesdecoder.git; }
[ -d subword-nmt  ] || { echo "shallow-cloning subword-nmt..."; git clone --depth 1 -q https://github.com/rsennrich/subword-nmt.git; }

# --- 3. corpus, md5-checked -------------------------------------------------
mkdir -p orig
if [ ! -f orig/de-en.tgz ]; then
  echo "downloading $URL ..."
  curl -fSL --progress-bar -o orig/de-en.tgz "$URL"
fi
MD5_ACTUAL=$(md5sum orig/de-en.tgz | cut -d' ' -f1)
if [ "$MD5_ACTUAL" = "$MD5_EXPECTED" ]; then
  echo "md5 OK: $MD5_ACTUAL (official fairseq IWSLT14 de-en distribution)"
else
  echo "WARNING: md5 is $MD5_ACTUAL, expected $MD5_EXPECTED -- upstream may have changed."
  echo "         Check the sentence counts printed at the end before using this data."
fi

# --- 4. run fairseq's script, unmodified ------------------------------------
# Its `git clone` and `wget` calls will complain that things already exist; that
# is expected and harmless, which is why this is not run under `set -e`.
echo
echo "=== running fairseq's prepare-iwslt14.sh (unmodified) ==="
set +e
bash "$ROOT/$SCRIPT"
set -e

PREP=iwslt14.tokenized.de-en
for f in train valid test; do
  for l in de en; do
    [ -s "$PREP/$f.$l" ] || { echo "FAILED: $WORK/$PREP/$f.$l is missing or empty"; exit 1; }
  done
done

echo
echo "=== sentence counts ==="
wc -l "$PREP"/{train,valid,test}.de | sed 's/^/  /'
echo "  published IWSLT14 de-en: train 160239  valid 7283  test 6750"
TRAIN_N=$(wc -l < "$PREP/train.de")
TEST_N=$(wc -l < "$PREP/test.de")
[ "$TEST_N" = "6750" ] && echo "  test size matches the published figure" \
                       || echo "  WARNING: test is $TEST_N, expected 6750"

# --- 5. binarise ------------------------------------------------------------
cd "$ROOT"
rm -rf "$OUT"
"$PY" preprocess.py \
  --source-lang de --target-lang en \
  --trainpref "$WORK/$PREP/train" \
  --validpref "$WORK/$PREP/valid" \
  --testpref  "$WORK/$PREP/test" \
  --destdir   "$OUT" \
  --joined-dictionary --workers 8

echo
echo "ready: $OUT  (src=de tgt=en, lowercased, joint 10k BPE)"
echo "text kept at $WORK/$PREP/ if you want to inspect it"
ls -la "$OUT"
