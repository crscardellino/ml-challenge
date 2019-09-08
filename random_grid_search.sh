#!/usr/bin/env bash

DATA_DIR="../data"
RESULTS_DIR="../results"

for i in $(seq 1 10)
do
    echo "******************* EXPLORING SETTING $i ***************************" >&2

    LAYER_SIZES=(512 768 1024)
    rand_layer_size=${LAYER_SIZES[$[$RANDOM % ${#LAYER_SIZES[@]}]]}
    echo "Layer size $rand_layer_size" >&2

    NETWORK_SIZES=(2 3 5)
    rand_network_size=${NETWORK_SIZES[$[$RANDOM % ${#NETWORK_SIZES[@]}]]}
    echo "Network size $rand_network_size" >&2

    DROPOUT=(0.1 0.2 0.3 0.4 0.5)
    rand_dropout=${DROPOUT[$[$RANDOM % ${#DROPOUT[@]}]]}
    echo "Dropout $rand_dropout" >&2

    OPTIMIZER=(adam rmsprop nadam adamax)
    rand_optimizer=${OPTIMIZER[$[$RANDOM % ${#OPTIMIZER[@]}]]}
    echo "Optimizer $rand_optimizer" >&2

    EPOCHS=(10 15 20)
    rand_epochs=${EPOCHS[$[$RANDOM % ${#EPOCHS[@]}]]}
    echo "Epochs $rand_epochs" >&2

    for language in spanish portuguese
    do
        echo "Running for $language" >&2

        python run_mlp.py $DATA_DIR/meli/train_reliable.parquet \
            $DATA_DIR/meli/dev.parquet \
            $DATA_DIR/meli/test.parquet \
            $language $DATA_DIR/$language/$language-word2vec.bin.gz \
            $RESULTS_DIR -l $rand_layer_size -n $rand_network_size -d $rand_dropout \
            -r 0 -o $rand_optimizer -b 1024 -e $rand_epochs
    done
done
