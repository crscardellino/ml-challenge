# -*- coding: utf-8 -*-

import argparse
import gc
import logging
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import yaml

from collections import Counter
from datetime import datetime
from gensim.models.fasttext import load_facebook_model
from nltk.corpus import stopwords
from os import path
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from string import punctuation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv1D, Dense, Dropout,
                                     Embedding, GlobalMaxPooling1D, Input, TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

np.random.seed(42)
tf.compat.v1.random.set_random_seed(42)

# Logging
logger = logging.getLogger(__name__)


def load_data(base_path, language, drop_columns, unreliable_sampling):
    datasets = {}
    for ds in tqdm(["train_reliable", "train_unreliable", "dev", "test"], file=sys.stdout):
        if ds == "train_unreliable" and unreliable_sampling == 0:
            continue

        df = pd.read_parquet(
            path.join(base_path, f"{language}", f"{ds}.parquet")
        ).drop(drop_columns, axis=1, errors="ignore")

        if ds == "train_unreliable" and 0 < unreliable_sampling < 1:
            df = df.groupby(["category"]).apply(
                lambda cat: cat.sample(frac=unreliable_sampling)
            ).reset_index(drop=True)
        elif ds == "train_unreliable" and unreliable_sampling > 1:
            df = df.groupby(["category"]).apply(
                lambda cat: cat.sample(n=int(unreliable_sampling))
            ).reset_index(drop=True)

        if ds == "train_reliable":
            datasets["train"] = df
        elif ds == "train_unreliable":
            datasets["train"] = pd.concat([
                datasets["train"],
                df
            ], ignore_index=True)
        else:
            datasets[ds] = df

    w2v = load_facebook_model(
        path.join(base_path, f"{language}", "fasttext.bin")
    )

    return datasets, w2v


def label_encoder(*dfs):
    labels = pd.concat(dfs)["category"].tolist()
    lbl_enc = LabelEncoder().fit(labels)

    return lbl_enc


def remove_punctuation(datasets, punctuation, input_col="title", output_col="title"):
    for split in tqdm(datasets, file=sys.stdout):
        datasets[split][output_col] = datasets[split][input_col].apply(
            lambda words: [w for w in words if w not in punctuation]
        )
    return datasets


def remove_stopwords(datasets, stopwords, input_col="title", output_col="title"):
    for split in tqdm(datasets, file=sys.stdout):
        datasets[split][output_col] = datasets[split][input_col].apply(
            lambda words: [w for w in words if w not in stopwords]
        )
    return datasets


def words_to_idx(all_words, min_words=10):
    word_index = Counter([word for words in all_words for word in words])
    word_index = {w for w, c in word_index.items() if c >= min_words}
    word_index = {word: idx for idx, word in enumerate(sorted(word_index), start=1)}
    word_index["<NULL>"] = 0
    word_index["<UNK>"] = len(word_index)

    return word_index


def chars_to_idx(titles):
    char_index = {char for title in titles for char in title}
    char_index = {char: idx for idx, char in enumerate(sorted(char_index), start=1)}
    char_index["<NULL>"] = 0
    char_index["<UNK>"] = len(char_index)

    return char_index


def word_sequence_padding(series, word_index, max_len):
    return pad_sequences(
            series.apply(
                lambda words: [word_index.get(word, word_index["<UNK>"]) for word in words]
            ).tolist(), maxlen=max_len
        )


def char_sequence_padding(series, char_index, char_max_len, word_max_len):
    return pad_sequences(
        series.apply(
            lambda words: pad_sequences([
                [char_index.get(char, char_index["<UNK>"]) for char in word]
                for word in words], maxlen=char_max_len)
        ), maxlen=word_max_len, value=np.zeros(char_max_len))


def get_embedding_matrix(word_index, w2v):
    embedding_matrix = np.zeros((len(word_index), w2v.vector_size))

    for word, i in word_index.items():
        if word in w2v and word not in {"<NULL>", "<UNK>"}:
            embedding_matrix[i] = w2v[word]
        elif word == "<UNK>":
            embedding_matrix[i] = np.random.normal(size=(w2v.vector_size,))

    return embedding_matrix


def build_model(word_vocab_size, word_vector_size, word_embedding_matrix,
                char_vocab_size, char_vector_size, output_size,
                word_max_sequence_len, char_max_sequence_len,
                word_dropout, word_filters_len, word_filter_count,
                char_dropout, char_filters_len, char_filter_count,
                activation="relu", padding="same"):

    char_sequence_input = Input(shape=(word_max_sequence_len, char_max_sequence_len))
    word_sequence_input = Input(shape=(word_max_sequence_len,))

    char_embedded_sequences = TimeDistributed(
        Embedding(
            input_dim=char_vocab_size,
            output_dim=char_vector_size,
            embeddings_initializer="truncated_normal",  # TODO: Change this?
            trainable=True
        ))(char_sequence_input)
    char_embedded_sequences = Dropout(
        rate=char_dropout,
        noise_shape=(None, word_max_sequence_len, 1, char_vector_size),
        seed=42
    )(char_embedded_sequences)

    word_embedding_layer = Embedding(word_vocab_size, word_vector_size,
                                     weights=[word_embedding_matrix],
                                     input_length=word_max_sequence_len,
                                     trainable=False)
    word_embedded_sequences = word_embedding_layer(word_sequence_input)
    word_embedded_sequences = Dropout(
        rate=word_dropout,
        noise_shape=(None, 1, word_vector_size),
        seed=42
    )(word_embedded_sequences)

    char_layers = []
    for filter_len in char_filters_len:
        char_layer = TimeDistributed(
            Conv1D(
                char_filter_count,
                filter_len,
                padding=padding
            )
        )(char_embedded_sequences)
        # char_layer = TimeDistributed(
        #     Conv1D(
        #         char_filter_count,
        #         filter_len,
        #         padding=padding
        #     )
        # )(char_layer)
        char_layer = BatchNormalization(momentum=0.0)(char_layer)
        char_layers.append(TimeDistributed(GlobalMaxPooling1D())(char_layer))

    word_layer = Concatenate()([word_embedded_sequences] + char_layers)

    layers = []
    for filter_len in word_filters_len:
        layer = Conv1D(
            word_filter_count,
            filter_len,
            activation=activation,
            padding=padding
        )(word_layer)
        layer = BatchNormalization(momentum=0.0)(layer)
        layers.append(GlobalMaxPooling1D()(layer))

    layer = Concatenate()(layers)
    preds = Dense(output_size, activation="softmax")(layer)
    model = Model(inputs=[word_sequence_input, char_sequence_input], outputs=[preds])

    return model


def main(base_data_dir, language, output, activation, batch_size,
         char_dropout, char_filter_count, char_filters_len, char_max_sequence_len, char_vector_size,
         drop_columns, epochs, keep_punctuation, keep_stopwords,
         optimizer, padding, unreliable_sampling, use_normalized_char_tokens,
         word_dropout, word_filter_count, word_filters_len, word_max_sequence_len):
    # Setup logger
    experiment = datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + "_" + language
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(lineno)s] %(message)s")
    handler = logging.FileHandler(path.join(output, f"{experiment}.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    config_setup = yaml.dump({
        "EXPERIMENT": experiment,
        "LANGUAGE": language,
        "CHAR DROPOUT": char_dropout,
        "CHAR FILTER COUNT": char_filter_count,
        "CHAR FILTERS LEN": ", ".join(map(str, char_filters_len)),
        "CHAR MAX SEQUENCE LEN": char_max_sequence_len,
        "CHAR VECTOR SIZE": char_vector_size,
        "UNRELIABLE SAMPLING": unreliable_sampling,
        "WORD DROPOUT": word_dropout,
        "WORD FILTER COUNT": word_filter_count,
        "WORD FILTERS LEN": ", ".join(map(str, word_filters_len)),
        "WORD MAX SEQUENCE LEN": word_max_sequence_len,
    })
    logger.info(f"Beggining experiments with configuration:\n{config_setup.strip()}")

    logger.info("Loading data")
    datasets, w2v = load_data(base_data_dir, language, drop_columns, unreliable_sampling)

    logger.info("Getting labels")
    lbl_enc = label_encoder(datasets["train"], datasets["dev"])

    for split in ["train", "dev"]:
        datasets[split]["target"] = lbl_enc.transform(datasets[split]["category"])
        datasets[split].drop(["category"], axis=1, inplace=True)

    datasets["dev"]["original_title"] = datasets["dev"]["title"]
    for split in ["train", "dev", "test"]:
        datasets[split]["title"] = datasets[split]["words"]
        datasets[split].drop(["words"], axis=1, inplace=True)

    if not keep_punctuation:
        logger.info("Removing punctuation from titles")
        datasets = remove_punctuation(datasets, punctuation)

    if not keep_stopwords:
        logger.info("Removing stopwords from titles")
        datasets = remove_stopwords(datasets, set(stopwords.words(language)))

    logger.info("Gathering word to index")
    word_index = words_to_idx(
        pd.concat(list(datasets.values()), sort=False)["title"],
        10
    )
    logger.info(f"Vocab length: {len(word_index)}")

    logger.info("Gathering char to index")
    if use_normalized_char_tokens:
        char_base_col = "normalized_tokens"
    else:
        char_base_col = "title"
    char_index = chars_to_idx(
        pd.concat(
            list(datasets.values()),
            sort=False
        )[char_base_col].apply(lambda tokens: " ".join(tokens))
    )
    logger.info(f"Char vocab length: {len(char_index)}")

    logger.info("Padding word sequences")
    train_word_sequences = word_sequence_padding(
        datasets["train"]["normalized_tokens"], word_index, word_max_sequence_len
    )
    dev_word_sequences = word_sequence_padding(
        datasets["dev"]["normalized_tokens"], word_index, word_max_sequence_len
    )

    logger.info("Padding char sequences")
    train_char_sequences = char_sequence_padding(
        datasets["train"][char_base_col], char_index, char_max_sequence_len, word_max_sequence_len
    )
    dev_char_sequences = char_sequence_padding(
        datasets["dev"][char_base_col], char_index, char_max_sequence_len, word_max_sequence_len
    )
    test_word_sequences = word_sequence_padding(
        datasets["test"]["normalized_tokens"], word_index, word_max_sequence_len
    )
    test_char_sequences = char_sequence_padding(
        datasets["test"][char_base_col], char_index, char_max_sequence_len, word_max_sequence_len
    )

    logger.info("Encoding labels to one-hot")
    train_target = to_categorical(
        datasets["train"]["target"].tolist(),
        num_classes=lbl_enc.classes_.shape[0]
    )
    dev_target = to_categorical(
        datasets["dev"]["target"].tolist(),
        num_classes=lbl_enc.classes_.shape[0]
    )

    data_save_path = path.join(output, f"{experiment}_data.npz")
    logger.info(f"Saving data to {data_save_path}")
    np.savez_compressed(
        data_save_path,
        dev_word_sequences=dev_word_sequences,
        dev_char_sequences=dev_char_sequences,
        dev_target=datasets["dev"]["target"].values,
        test_word_sequences=test_word_sequences,
        test_char_sequences=test_char_sequences
    )

    logger.info("Getting word embedding matrix")
    word_embedding_matrix = get_embedding_matrix(word_index, w2v)

    logger.info("Building model")
    model = build_model(
        word_vocab_size=len(word_index),
        word_vector_size=w2v.vector_size,
        word_embedding_matrix=word_embedding_matrix,
        char_vocab_size=len(char_index),
        char_vector_size=char_vector_size,
        output_size=lbl_enc.classes_.shape[0],
        word_max_sequence_len=word_max_sequence_len,
        char_max_sequence_len=char_max_sequence_len,
        word_dropout=word_dropout,
        word_filters_len=word_filters_len,
        word_filter_count=word_filter_count,
        char_dropout=char_dropout,
        char_filters_len=char_filters_len,
        char_filter_count=char_filter_count,
        activation=activation,
        padding=padding
    )

    logger.info("Cleaning up data to save memory")
    del datasets["train"]
    del w2v
    del test_word_sequences
    del test_char_sequences
    gc.collect()

    logger.info("Compiling model")
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary(print_fn=logger.info)

    logger.info("Fitting model")
    model.fit(
        (train_word_sequences, train_char_sequences), train_target,
        validation_data=(
            (dev_word_sequences, dev_char_sequences),
            dev_target
        ),
        batch_size=batch_size, epochs=epochs,
        verbose=1, validation_split=0, validation_freq=1,
        callbacks=[
            ModelCheckpoint(
                path.join(output, f"{experiment}_best_model.h5"),
                monitor="val_accuracy",
                verbose=1,
                save_best_only=True,
                mode='max',
                period=1
            )
        ]
    )

    logger.info("Cleaning up data to save memory")
    del train_word_sequences
    del train_char_sequences
    del train_target
    gc.collect()

    logger.info("Model finished trainig. Getting final predictions.")

    logger.info("Getting dev data predictions")
    datasets["dev"]["predictions"] = model.predict(
        (dev_word_sequences, dev_char_sequences), batch_size=batch_size, verbose=0
    ).argmax(axis=1)

    logger.info("Saving eyeball dataset")
    eyeball_dataset = datasets["dev"][
        datasets["dev"]["target"] != datasets["dev"]["predictions"]
    ].head(100)
    eyeball_dataset["category"] = lbl_enc.inverse_transform(eyeball_dataset["target"])
    eyeball_dataset["pcategory"] = lbl_enc.inverse_transform(eyeball_dataset["predictions"])
    eyeball_dataset[
        ["original_title", "normalized_tokens", "label_quality", "category", "pcategory"]
    ].to_csv(
        path.join(output, f"{experiment}_eyeball.csv"), index=False
    )

    dev_acc = balanced_accuracy_score(datasets["dev"]["target"], datasets["dev"]["predictions"])
    logger.info(f"Balanced Accuracy Score for VALIDATION (TOTAL): {dev_acc}")

    logger.info("Getting test data predictions")
    test_word_sequences = word_sequence_padding(
        datasets["test"]["normalized_tokens"], word_index, word_max_sequence_len
    )
    test_char_sequences = char_sequence_padding(
        datasets["test"][char_base_col], char_index, char_max_sequence_len, word_max_sequence_len
    )
    datasets["test"]["predictions"] = model.predict(
        (test_word_sequences, test_char_sequences), batch_size=batch_size, verbose=0
    ).argmax(axis=1)

    logger.info("Writing test output predictions")
    datasets["test"]["category"] = lbl_enc.inverse_transform(datasets["test"]["predictions"])
    results_save_path = path.join(output, f"{experiment}_results.csv")
    datasets["test"][["id", "category"]].to_csv(results_save_path, index=False)

    model_save_path = path.join(output, f"{experiment}_model.h5")
    logger.info(f"Saving model to {model_save_path}")
    model.save(model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_data_dir")
    parser.add_argument("language")
    parser.add_argument("output")
    parser.add_argument("--activation", "-a", default="relu")
    parser.add_argument("--batch-size", "-b", default=4096, type=int)
    parser.add_argument("--char-dropout", default=0.0, type=float)
    parser.add_argument("--char-filter-count", "-c", default=32, type=int)
    parser.add_argument("--char-filters-len", "-f", default=[3, 4], type=int, nargs="+")
    parser.add_argument("--char-max-sequence-len", "-m", default=10, type=int)
    parser.add_argument("--char-vector-size", "-v", default=32, type=int)
    parser.add_argument("--drop-columns", "-d",
                        default=["split", "language", "pos"],
                        nargs="+")
    parser.add_argument("--epochs", "-e", default=10, type=int)
    parser.add_argument("--keep-punctuation", "-k", action="store_true")
    parser.add_argument("--keep-stopwords", "-s", action="store_true")
    parser.add_argument("--optimizer", "-o", default="nadam")
    parser.add_argument("--padding", "-p", default="same")
    parser.add_argument("--unreliable-sampling", "-u", default=0.5, type=float)
    parser.add_argument("--use-normalized-char-tokens", "-t", action="store_true")
    parser.add_argument("--word-dropout", default=0.0, type=float)
    parser.add_argument("--word-filter-count", "-w", default=128, type=int)
    parser.add_argument("--word-filters-len", "-x", default=[2, 3, 4, 5], type=int, nargs="+")
    parser.add_argument("--word-max-sequence-len", "-y", default=15, type=int)

    args = parser.parse_args()

    main(**vars(args))
