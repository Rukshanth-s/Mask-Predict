#!/usr/bin/env bash
#
# Download, tokenise, BPE and binarise a parallel corpus for translation_self.
#
# Replaces get_data.sh, whose Google Drive file IDs date from 2019 and no longer
# resolve ("Cannot retrieve the public link of the file") -- every download in it
# fails, and the preprocess steps then fail on the missing files. Everything here
# comes from live HTTPS sources instead.
#
#   bash scripts/prepare_data.sh multi30k     # 29k pairs de-en, minutes -- start here
#   bash scripts/prepare_data.sh wmt16-enro   # ~400k pairs en-ro, the paper's benchmark
#
# Output: data-bin/<name>/ ready for train.py, plus data/<name>/ with the text.
#
# Requires: pip install subword-nmt sacremoses
set -euo pipefail

DATASET="${1:-multi30k}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Some conda envs ship python3 without a `python` alias, so don't hardcode either.
# Override with PYTHON=/path/to/python if you need a specific interpreter.
PY="${PYTHON:-$(command -v python3 || command -v python)}"
[ -n "$PY" ] || { echo "no python3 or python on PATH"; exit 1; }
echo "using interpreter: $PY ($("$PY" -V 2>&1))"

command -v subword-nmt >/dev/null || { echo "need: pip install subword-nmt"; exit 1; }
"$PY" -c "import sacremoses" 2>/dev/null || { echo "need: pip install sacremoses"; exit 1; }

# ---------------------------------------------------------------------------
# Per-dataset fetch. Each leaves data/$NAME/{train,valid,test}.$SRC/$TGT as
# raw (untokenised) text, one sentence per line, and sets BPE_MERGES.
# ---------------------------------------------------------------------------
fetch_multi30k() {
  NAME=multi30k; SRC=de; TGT=en; BPE_MERGES=10000
  local base=https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw
  mkdir -p "data/$NAME" && pushd "data/$NAME" >/dev/null
  for lang in de en; do
    for pair in "train:train" "val:valid" "test_2016_flickr:test"; do
      local remote="${pair%%:*}" local_name="${pair##*:}"
      [ -f "$local_name.$lang" ] && continue
      curl -fsSL -o "$local_name.$lang.gz" "$base/$remote.$lang.gz"
      gunzip -f "$local_name.$lang.gz"
    done
  done
  popd >/dev/null
}

fetch_wmt16_enro() {
  NAME=wmt16.en-ro; SRC=en; TGT=ro; BPE_MERGES=40000
  local dl=https://data.statmt.org/wmt16/translation-task
  mkdir -p "data/$NAME/orig" && pushd "data/$NAME/orig" >/dev/null
  # Training: Europarl v8 is the ro-en parallel corpus WMT16 distributes directly.
  [ -f europarl-v8.ro-en.en ] || {
    curl -fsSL -O "$dl/training-parallel-ep-v8.tgz"
    tar -xzf training-parallel-ep-v8.tgz
    find . -name 'europarl-v8.ro-en.*' -exec mv {} . \; 2>/dev/null || true
  }
  # Dev/test: newsdev2016 and newstest2016, shipped as SGML inside dev.tgz/test.tgz.
  [ -f dev.tgz ]  || curl -fsSL -O "$dl/dev.tgz"
  [ -f test.tgz ] || curl -fsSL -O "$dl/test.tgz"
  tar -xzf dev.tgz  2>/dev/null || true
  tar -xzf test.tgz 2>/dev/null || true
  popd >/dev/null

  pushd "data/$NAME" >/dev/null
  cp orig/europarl-v8.ro-en.en train.en
  cp orig/europarl-v8.ro-en.ro train.ro
  # De-SGML the dev/test sets: keep <seg> contents and unescape entities.
  desgml() {
    grep '<seg' "$1" \
      | sed -e 's/<seg[^>]*>//' -e 's#</seg>##' -e 's/^\s*//' -e 's/\s*$//' \
            -e 's/&amp;/\&/g' -e 's/&lt;/</g' -e 's/&gt;/>/g' -e 's/&quot;/"/g' \
            -e "s/&apos;/'/g"
  }
  for lang in en ro; do
    desgml "$(find orig -name "newsdev2016-enro-*$lang.sgm" -o -name "newsdev2016-roen-*$lang.sgm" | head -1)"  > "valid.$lang"
    desgml "$(find orig -name "newstest2016-enro-*$lang.sgm" -o -name "newstest2016-roen-*$lang.sgm" | head -1)" > "test.$lang"
  done
  popd >/dev/null
}

case "$DATASET" in
  multi30k)    fetch_multi30k ;;
  wmt16-enro)  fetch_wmt16_enro ;;
  *) echo "unknown dataset '$DATASET' (multi30k | wmt16-enro)"; exit 1 ;;
esac

cd "data/$NAME"
echo "=== raw line counts ==="
wc -l "train.$SRC" "train.$TGT" "valid.$SRC" "valid.$TGT" "test.$SRC" "test.$TGT"

# ---------------------------------------------------------------------------
# Tokenise. Moses tokenisation, matching what the released dictionaries assume.
# ---------------------------------------------------------------------------
for split in train valid test; do
  for lang in "$SRC" "$TGT"; do
    [ -f "$split.tok.$lang" ] && continue
    sacremoses -l "$lang" -j 8 tokenize -x < "$split.$lang" > "$split.tok.$lang"
  done
done

# Drop empty and wildly mismatched training pairs; they destabilise the length
# predictor, which this model trains jointly with translation. Written to its own
# filename rather than over train.tok.*, so re-running this script is idempotent
# and each stage's input stays inspectable.
"$PY" - "$SRC" "$TGT" <<'PY'
import sys
src, tgt = sys.argv[1], sys.argv[2]
kept = dropped = 0
with open(f'train.tok.{src}') as fs, open(f'train.tok.{tgt}') as ft, \
     open(f'train.clean.{src}', 'w') as os_, open(f'train.clean.{tgt}', 'w') as ot:
    for s, t in zip(fs, ft):
        ls, lt = len(s.split()), len(t.split())
        if 1 <= ls <= 175 and 1 <= lt <= 175 and ls / max(lt, 1) <= 1.5 and lt / max(ls, 1) <= 1.5:
            os_.write(s); ot.write(t); kept += 1
        else:
            dropped += 1
print(f'cleaned train: kept {kept}, dropped {dropped}')
PY

# ---------------------------------------------------------------------------
# Joint BPE. Joint (not per-language) because --share-all-embeddings needs one
# shared vocabulary across both sides. Codes are learned on the cleaned training
# text only -- never on valid/test, which would leak.
# ---------------------------------------------------------------------------
if [ ! -f bpe.codes ]; then
  cat "train.clean.$SRC" "train.clean.$TGT" | subword-nmt learn-bpe -s "$BPE_MERGES" > bpe.codes
fi
for split in train valid test; do
  # train's BPE input is the cleaned text; valid/test are only tokenised.
  [ "$split" = train ] && stage=clean || stage=tok
  for lang in "$SRC" "$TGT"; do
    subword-nmt apply-bpe -c bpe.codes < "$split.$stage.$lang" > "$split.bpe.$lang"
  done
done

cd "$ROOT"
rm -rf "data-bin/$NAME"
"$PY" preprocess.py \
  --source-lang "$SRC" --target-lang "$TGT" \
  --trainpref "data/$NAME/train.bpe" \
  --validpref "data/$NAME/valid.bpe" \
  --testpref  "data/$NAME/test.bpe" \
  --destdir   "data-bin/$NAME" \
  --joined-dictionary --workers 8

echo
echo "ready: data-bin/$NAME  (src=$SRC tgt=$TGT)"
ls -la "data-bin/$NAME"