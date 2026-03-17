#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses...'
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPE_CODE=32000
BPE_TOKENS=32000
FASTBPE=fastBPE/fast

URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ=de-en.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please clone mosesdecoder to $SCRIPTS"
    echo "git clone https://github.com/moses-smt/mosesdecoder.git"
    exit 1
fi

if [ ! -f "$FASTBPE" ]; then
    echo "Please compile fastBPE to $FASTBPE"
    echo "git clone https://github.com/glample/fastBPE.git && cd fastBPE && g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast"
    exit 1
fi

OUTDIR=data/iwslt14.tokenized.de-en

mkdir -p $OUTDIR
cd $OUTDIR

# Download if missing
if [ ! -f "$GZ" ]; then
    echo "Downloading data from ${URL}..."
    wget "$URL"
    if [ -f "$GZ" ]; then
        tar zxvf $GZ
        mv de-en/* .
    fi
fi
cd ../..

echo "Done"
