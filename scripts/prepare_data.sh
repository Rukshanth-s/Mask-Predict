#!/usr/bin/env bash
#
# Download, tokenise, BPE and binarise a parallel corpus for translation_self.
#
# Replaces get_data.sh, whose Google Drive file IDs date from 2019 and no longer
# resolve ("Cannot retrieve the public link of the file") -- every download in it
# fails, and the preprocess steps then fail on the missing files. Everything here
# comes from live HTTPS sources instead.
#
#   bash scripts/prepare_data.sh multi30k     # 29k pairs de-en, 13-word captions
#   bash scripts/prepare_data.sh iwslt14      # 160k pairs de-en, 19-word TED talks
#   bash scripts/prepare_data.sh wmt16-enro   # ~400k pairs en-ro, the paper's benchmark
#
# multi30k and iwslt14 are verified end to end. wmt16-enro is NOT: its URLs were
# checked but the 241 MB download and the SGML extraction have never been run.
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

fetch_iwslt14() {
  # IWSLT14 de-en: ~160k pairs of TED talk transcripts, ~20 words/sentence.
  # Five times Multi30k and much longer sentences, which is what the CRF needs to
  # have anything to fix -- but only 19 MB, versus 241 MB for WMT16.
  #
  # The original source (wit3.fbk.eu) now redirects to a Google login, so the
  # tarball comes from this repo's own history: it is committed on the
  # freeze_layer branch at data/iwslt14.tokenized.de-en/de-en.tgz.
  NAME=iwslt14.de-en; SRC=de; TGT=en; BPE_MERGES=10000
  local tgz=data/$NAME/de-en.tgz
  mkdir -p "data/$NAME"

  if [ ! -f "$tgz" ]; then
    echo "fetching de-en.tgz from git history (branch freeze_layer)..."
    git archive origin/freeze_layer data/iwslt14.tokenized.de-en/de-en.tgz 2>/dev/null \
      | tar -xO > "$tgz" || {
        rm -f "$tgz"
        cat <<'MSG'
Could not read de-en.tgz from origin/freeze_layer.

If you cloned with --single-branch, that branch is not present. Fetch it (~33 MB):

    git remote set-branches origin '*'
    git fetch --all

then re-run this script.
MSG
        exit 1; }
  fi
  [ -s "$tgz" ] || { echo "de-en.tgz is empty"; exit 1; }

  pushd "data/$NAME" >/dev/null
  [ -d de-en ] || tar -xzf de-en.tgz

  # Training text: drop the three metadata line types and untag title/description.
  # This is exactly fairseq's prepare-iwslt14.sh recipe. Verified on this tarball:
  # 178526 raw lines in both languages -> 174443 in both, i.e. 3 x 1361 talks
  # removed, so the two sides stay aligned. <speaker>/<doc>/<reviewer> do not
  # occur in this distribution, so do not filter for them.
  for lang in de en; do
    [ -f "train.raw.$lang" ] && continue
    grep -v '<url>' "de-en/train.tags.de-en.$lang" \
      | grep -v '<talkid>' | grep -v '<keywords>' \
      | sed -e 's/<title>//g' -e 's#</title>##g' \
            -e 's/<description>//g' -e 's#</description>##g' \
            -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
      > "train.raw.$lang"
  done

  # Dev/test come as SGML: keep the <seg> contents.
  desgml_seg() {
    grep '<seg id' "$1" \
      | sed -e 's#<seg id="[0-9]*">[[:space:]]*##g' -e 's#[[:space:]]*</seg>##g' \
            -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
  }
  # The canonical IWSLT14 de-en setup: test is the concatenation of dev2010,
  # tst2010, tst2011, tst2012 AND TEDX.dev2012 -- 887+1565+1433+1700+1165 = 6750,
  # which is the published test-set size. Omitting TEDX gives 5585 and would not
  # be comparable to the literature. valid = every 23rd training line held out.
  for lang in de en; do
    : > "test.$lang"
    for s in TED.dev2010 TED.tst2010 TED.tst2011 TED.tst2012 TEDX.dev2012; do
      desgml_seg "de-en/IWSLT14.$s.de-en.$lang.xml" >> "test.$lang"
    done
    awk 'NR % 23 == 0'  "train.raw.$lang" > "valid.$lang"
    awk 'NR % 23 != 0'  "train.raw.$lang" > "train.$lang"
  done
  popd >/dev/null

  echo "IWSLT14 de-en split: train $(wc -l < data/$NAME/train.de), valid $(wc -l < data/$NAME/valid.de), test $(wc -l < data/$NAME/test.de) (test should be 6750)"
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
  iwslt14)     fetch_iwslt14 ;;
  wmt16-enro)  fetch_wmt16_enro ;;
  *) echo "unknown dataset '$DATASET' (multi30k | iwslt14 | wmt16-enro)"; exit 1 ;;
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