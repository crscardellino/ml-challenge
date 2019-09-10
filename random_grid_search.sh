#!/usr/bin/env bash

DATA_DIR="../data"
EXPERIMENTS_DIR="../experiments"

for i in 1 # $(seq 1 5)
do
    echo "******************* EXPLORING SETTING $i ***************************" >&2

    MAX_SEQUENCE_LEN=(15)
    rand_max_seq_len=${MAX_SEQUENCE_LEN[$[$RANDOM % ${#MAX_SEQUENCE_LEN[@]}]]}
    echo "Max sequence length $rand_max_seq_len" >&2

    FILTERS=("2 3 4 5")
    rand_filters=${FILTERS[$[$RANDOM % ${#FILTERS[@]}]]}
    echo "Filters $rand_filters" >&2

    FILTER_COUNT=(128)
    rand_filter_count=${FILTER_COUNT[$[$RANDOM % ${#FILTER_COUNT[@]}]]}
    echo "Filter count $rand_filter_count" >&2

    PADDING=(same)
    rand_padding=${PADDING[$[$RANDOM % ${#PADDING[@]}]]}
    echo "Padding $rand_padding" >&2

    REGULARIZER=(0)
    rand_reg=${REGULARIZER[$[$RANDOM % ${#REGULARIZER[@]}]]}
    echo "Regularizer $rand_reg" >&2

    for language in spanish portuguese
    do
        echo "Running for $language" >&2

        python run_model.py \
            $DATA_DIR \
            $language \
            $EXPERIMENTS_DIR \
            -a relu \
            -b 2048 \
            -d split language pos \
            -e 10 \
            -f $rand_filters \
            -c $rand_filter_count \
            -s \
            -m $rand_max_seq_len \
            -o nadam \
            -p $rand_padding \
            -r $rand_reg \
            -u 0.5
    done
done
