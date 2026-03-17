#!/usr/bin/env bash

# 1. Run the base preparation script to get the tokenized full dataset
bash examples/translation/prepare-iwslt14.sh

# 2. Create the micro dataset directory
mkdir -p data/micro_iwslt14

echo "Extracting 100 lines for the micro-dataset..."

# 3. Slice exactly the first 100 lines for train, valid, and test
for SPLIT in train valid test; do
    for LANG in de en; do
        head -n 100 data/iwslt14.tokenized.de-en/${SPLIT}.${LANG} > data/micro_iwslt14/${SPLIT}.${LANG}
    done
done

echo "Binarizing the micro-dataset with fairseq-preprocess..."

# 4. Run Fairseq Preprocess
python3 preprocess.py \
    --source-lang de --target-lang en \
    --trainpref data/micro_iwslt14/train \
    --validpref data/micro_iwslt14/valid \
    --testpref data/micro_iwslt14/test \
    --destdir data-bin/micro_iwslt14 \
    --joined-dictionary \
    --workers 4

echo "Micro-dataset preparation complete!"
