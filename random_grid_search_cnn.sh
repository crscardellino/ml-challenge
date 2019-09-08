#!/usr/bin/env bash

DATA_DIR="../data"
RESULTS_DIR="../results"

for i in 1
do
    echo "******************* EXPLORING SETTING $i ***************************" >&2

    # LAYER_SIZES=(512 1024)
    # rand_layer_size=${LAYER_SIZES[$[$RANDOM % ${#LAYER_SIZES[@]}]]}
    # echo "Layer size $rand_layer_size" >&2

    # NETWORK_SIZES=(0 1 2)
    # rand_network_size=${NETWORK_SIZES[$[$RANDOM % ${#NETWORK_SIZES[@]}]]}
    # echo "Network size $rand_network_size" >&2

    # MAX_SEQUENCE_LEN=(10 15 20)
    # rand_max_seq_len=${MAX_SEQUENCE_LEN[$[$RANDOM % ${#MAX_SEQUENCE_LEN[@]}]]}
    # echo "Max sequence length $rand_max_seq_len" >&2

    # FILTERS=("2 3 4" "2 3 5" "2 3 6" "2 3 5 10")
    # rand_filters=${FILTERS[$[$RANDOM % ${#FILTERS[@]}]]}
    # echo "Filters $rand_filters" >&2

    # FILTER_COUNT=(32 64 128 256)
    # rand_filter_count=${FILTER_COUNT[$[$RANDOM % ${#FILTER_COUNT[@]}]]}
    # echo "Filter count $rand_filter_count" >&2

    # PADDING=(valid same)
    # rand_padding=${PADDING[$[$RANDOM % ${#PADDING[@]}]]}
    # echo "Padding $rand_padding" >&2

    for language in spanish portuguese
    do
        echo "Running for $language" >&2

        python run_cnn.py $DATA_DIR/meli/train_reliable.parquet \
            $DATA_DIR/meli/dev.parquet \
            $DATA_DIR/meli/test.parquet \
            $language \
            $DATA_DIR/$language/$language-word2vec.bin.gz \
            $RESULTS_DIR \
            -a relu \
            -b 1024 \
            -e 15 \
            -f 2 3 5 \
            -c 128 \
            -l 128 \
            -m 20 \
            -n 1 \
            -o nadam \
            -p valid \
            -t $DATA_DIR/meli/train_unreliable.parquet \
            -u 0.5
    done
done
