#!/usr/bin/env bash

DATA_DIR="../data"
EXPERIMENTS_DIR="../experiments"

for language in spanish portuguese
do
    echo "Running for $language" >&2

    for usize in 0.5 0.75 1
    do
        for epochs in 5 10
        do
            python run_cnn_char_emb.py \
                $DATA_DIR \
                $language \
                $EXPERIMENTS_DIR \
                --activation relu \
                --batch-size 4096 \
                --char-filter-count 64 \
                --char-filters-len 2 3 4 \
                --char-max-sequence-len 10 \
                --char-vector-size 32 \
                --drop-columns language pos split \
                --epochs $epochs \
                --optimizer nadam \
                --padding same \
                --unreliable-sampling $usize \
                --word-filter-count 128 \
                --word-filters-len 2 3 4 5 \
                --word-max-sequence-len 15
        done
    done
done
