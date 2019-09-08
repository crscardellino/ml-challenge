#!/usr/bin/env bash

DATA_DIR="../data"
RESULTS_DIR="../results"


for language in spanish portuguese
do
    echo "Running for $language" >&2

    python run_svm.py $DATA_DIR/meli/train_reliable.parquet \
        $DATA_DIR/meli/dev.parquet \
        $DATA_DIR/meli/test.parquet \
        $language $RESULTS_DIR

    python run_svm.py $DATA_DIR/meli/train_reliable.parquet \
        $DATA_DIR/meli/dev.parquet \
        $DATA_DIR/meli/test.parquet \
        $language $RESULTS_DIR -i

    python run_svm.py $DATA_DIR/meli/train_reliable.parquet \
        $DATA_DIR/meli/dev.parquet \
        $DATA_DIR/meli/test.parquet \
        $language $RESULTS_DIR -l

    python run_svm.py $DATA_DIR/meli/train_reliable.parquet \
        $DATA_DIR/meli/dev.parquet \
        $DATA_DIR/meli/test.parquet \
        $language $RESULTS_DIR -i -l
done