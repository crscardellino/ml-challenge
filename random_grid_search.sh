#!/usr/bin/env bash

DATA_DIR="../data"
EXPERIMENTS_DIR="../experiments"

for i in $(seq(1 5))
do
    echo "******************* EXPLORING SETTING $i ***************************" >&2

    MAX_SEQUENCE_LEN=(10 15)
    rand_max_seq_len=${MAX_SEQUENCE_LEN[$[$RANDOM % ${#MAX_SEQUENCE_LEN[@]}]]}
    echo "Max sequence length $rand_max_seq_len" >&2

    FILTERS=("2 3 4" "2 3 4 5" "2 3 4 5 6")
    rand_filters=${FILTERS[$[$RANDOM % ${#FILTERS[@]}]]}
    echo "Filters $rand_filters" >&2

    FILTER_COUNT=(128 256)
    rand_filter_count=${FILTER_COUNT[$[$RANDOM % ${#FILTER_COUNT[@]}]]}
    echo "Filter count $rand_filter_count" >&2

    PADDING=(valid same)
    rand_padding=${PADDING[$[$RANDOM % ${#PADDING[@]}]]}
    echo "Padding $rand_padding" >&2

    REGULARIZER=(0 0.01)
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
            -b 1024 \
            -d split language words pos \
            -e 20 \
            -f $rand_filters \
            -c $rand_filter_count \
            -m $rand_max_seq_len \
            -o nadam \
            -p $rand_padding \
            -r $rand_reg \
            -u 0.25
    done
done
