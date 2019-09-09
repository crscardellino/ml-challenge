# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np
import pandas as pd
import re
import sys
import tensorflow as tf

from datetime import datetime
from gensim.models import KeyedVectors
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from os import path
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from string import punctuation
from tensorflow.keras.layers import Concatenate, Conv1D, Dense, Embedding, GlobalMaxPooling1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from unidecode import unidecode

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

    w2v = KeyedVectors.load_word2vec_format(
        path.join(base_path, f"{language}", "word2vec.bin.gz"),
        binary=True
    )

    return datasets, w2v


def label_encoder(*dfs):
    labels = pd.concat(dfs)["category"].tolist()
    lbl_enc = LabelEncoder().fit(labels)

    return lbl_enc


def lowercase_titles(datasets):
    for split in tqdm(datasets, file=sys.stdout):
        datasets[split]["title"] = datasets[split]["title"].str.lower()
    return datasets


def tokenization(datasets, language):
    for split in tqdm(datasets, file=sys.stdout):
        datasets[split]["title"] = datasets[split]["title"].apply(
            lambda title: word_tokenize(title, language=language)
        )
    return datasets


def remove_punctuation(datasets, punctuation):
    for split in tqdm(datasets, file=sys.stdout):
        datasets[split]["title"] = datasets[split]["title"].apply(
            lambda words: [w for w in words if w not in punctuation]
        )
    return datasets


def remove_stopwords(datasets, stopwords):
    for split in tqdm(datasets, file=sys.stdout):
        datasets[split]["title"] = datasets[split]["title"].apply(
            lambda words: [w for w in words if w not in stopwords]
        )
    return datasets


def word_with_vector(word, w2v, stemmer):
    if word in w2v:
        return word
    elif word.capitalize() in w2v:
        return word.capitalize()
    elif word.upper() in w2v:
        return word.upper()
    elif unidecode(word) in w2v:
        return unidecode(word)
    elif unidecode(word.capitalize()) in w2v:
        return unidecode(word.capitalize())
    elif unidecode(word.upper()) in w2v:
        return unidecode(word.upper())
    elif stemmer.stem(word) in w2v:
        return stemmer.stem(word)
    elif re.search(r"\d+", word):
        return "<NUM>"
    else:
        return "<UNK>"
    # TODO: Lemmatization? Other normalizations?


def word_vectorize(datasets, language, w2v):
    stemmer = SnowballStemmer(language)
    for split in tqdm(datasets, file=sys.stdout):
        datasets[split]["title"] = datasets[split]["title"].apply(
            lambda words: [word_with_vector(w, w2v, stemmer) for w in words]
        )
    return datasets


def words_to_idx(all_words, w2v):
    word_index = {word for words in all_words for word in words if word in w2v}
    word_index = {word: idx for idx, word in enumerate(sorted(word_index), start=1)}
    word_index["<NULL>"] = 0
    word_index["<NUM>"] = len(word_index)
    word_index["<UNK>"] = len(word_index)

    return word_index


def sequence_padding(series, word_index, max_len):
    return pad_sequences(
            series.apply(
                lambda words: [word_index.get(word, word_index["<UNK>"]) for word in words]
            ).tolist(), maxlen=max_len
        )


def get_embedding_matrix(word_index, w2v):
    embedding_matrix = np.zeros((len(word_index), w2v.vector_size))

    for word, i in word_index.items():
        if word in w2v and word not in {"<NULL>", "<UNK>", "<NUM>"}:
            embedding_matrix[i] = w2v[word]
        elif word == "<UNK>" or word == "<NUM>":
            embedding_matrix[i] = np.random.normal(size=(w2v.vector_size,))

    return embedding_matrix


def build_model(word_vocab_size, word_vector_size, word_embedding_matrix,
                output_size, max_sequence_len, filters, filter_count, reg_lambda,
                activation, padding):
    word_embedding_layer = Embedding(word_vocab_size, word_vector_size,
                                     weights=[word_embedding_matrix],
                                     input_length=max_sequence_len,
                                     trainable=False)
    word_sequence_input = Input(shape=(max_sequence_len,))
    word_embedded_sequences = word_embedding_layer(word_sequence_input)

    layers = []
    for filter_size in filters:
        layer = Conv1D(
            filter_count,
            filter_size,
            activation=activation,
            padding=padding,
            kernel_regularizer=l2(reg_lambda) if reg_lambda > 0 else None
        )(word_embedded_sequences)
        layer = GlobalMaxPooling1D()(layer)
        layers.append(layer)

    layer = Concatenate()(layers)
    preds = Dense(output_size, activation="softmax")(layer)
    model = Model(word_sequence_input, preds)

    return model


def main(base_data_dir, language, output, activation, batch_size, drop_columns,
         epochs, filters, filter_count, keep_punctuation, keep_stopwords, max_sequence_len,
         optimizer, padding, reg_lambda, unreliable_sampling):
    # Setup logger
    experiment = datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + "_" + language
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(lineno)s] %(message)s")
    handler = logging.FileHandler(path.join(output, f"{experiment}.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    config_setup = f"BASE DIR: {base_data_dir}\n"
    config_setup += f"LANGUAGE: {language}\n"
    config_setup += f"ACTIVATION: {activation}\n"
    config_setup += f"BATCH SIZE: {batch_size}\n"
    config_setup += f"DROP COLUMNS: {drop_columns}\n"
    config_setup += f"EPOCHS: {epochs}\n"
    config_setup += f"FILTERS: {filters}\n"
    config_setup += f"FILTER COUNT: {filter_count}\n"
    config_setup += f"MAX SEQUENCE LENGTH: {max_sequence_len}\n"
    config_setup += f"PADDING: {padding}\n"
    config_setup += f"REG LAMBDA: {reg_lambda}\n"
    config_setup += f"OPTIMIZER: {optimizer}\n"
    config_setup += f"UNRELIABLE SAMPLING: {unreliable_sampling}"
    logger.info(f"Beggining experiments with configuration:\n{config_setup.strip()}")

    logger.info("Loading data")
    datasets, w2v = load_data(base_data_dir, language, drop_columns, unreliable_sampling)

    logger.info("Getting labels")
    lbl_enc = label_encoder(datasets["train"], datasets["dev"])

    for split in ["train", "dev"]:
        datasets[split]["target"] = lbl_enc.transform(datasets[split]["category"])
        datasets[split].drop(["category"], axis=1, inplace=True)

    datasets["dev"]["original_title"] = datasets["dev"]["title"]

    logger.info("Lowercasing titles")
    datasets = lowercase_titles(datasets)

    logger.info("Tokenizing titles")
    datasets = tokenization(datasets, language)

    if not keep_punctuation:
        logger.info("Removing punctuation from titles")
        datasets = remove_punctuation(datasets, punctuation)

    if not keep_stopwords:
        logger.info("Removing stopwords from titles")
        datasets = remove_stopwords(datasets, set(stopwords.words(language)))

    logger.info("Vectorizing words")
    datasets = word_vectorize(datasets, language, w2v)

    logger.info("Gathering word to index")
    word_index = words_to_idx(pd.concat(list(datasets.values()), sort=False)["title"], w2v)
    logger.info(f"Vocab length: {len(word_index)}")

    logger.info("Padding sequences")
    train_word_sequences = sequence_padding(
        datasets["train"]["title"], word_index, max_sequence_len
    )
    dev_word_sequences = sequence_padding(
        datasets["dev"]["title"], word_index, max_sequence_len
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

    logger.info("Getting embedding matrix")
    word_embedding_matrix = get_embedding_matrix(word_index, w2v)

    logger.info("Building model")
    model = build_model(
        word_vocab_size=len(word_index),
        word_vector_size=w2v.vector_size,
        word_embedding_matrix=word_embedding_matrix,
        output_size=lbl_enc.classes_.shape[0],
        max_sequence_len=max_sequence_len,
        filters=filters,
        filter_count=filter_count,
        reg_lambda=reg_lambda,
        activation=activation,
        padding=padding
    )

    logger.info("Cleaning up data to save memory")
    del datasets["train"]
    del w2v

    logger.info("Compiling model")
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary(print_fn=logger.info)

    logger.info("Fitting model")
    model.fit(
        train_word_sequences, train_target,
        validation_data=(dev_word_sequences, dev_target),
        batch_size=batch_size, epochs=epochs,
        verbose=1, validation_split=0, validation_freq=5
    )

    logger.info("Model finished trainig. Getting final predictions.")

    logger.info("Getting dev data predictions")
    datasets["dev"]["predictions"] = model.predict(
        dev_word_sequences, batch_size=batch_size, verbose=0
    ).argmax(axis=1)

    logger.info("Saving eyeball dataset")
    eyeball_dataset = datasets["dev"][
        datasets["dev"]["target"] != datasets["dev"]["predictions"]
    ].head(100)
    eyeball_dataset["category"] = lbl_enc.inverse_transform(eyeball_dataset["target"])
    eyeball_dataset["pcategory"] = lbl_enc.inverse_transform(eyeball_dataset["predictions"])
    eyeball_dataset[["original_title", "title", "label_quality", "category", "pcategory"]].to_csv(
        path.join(output, f"{experiment}_eyeball.csv"), index=False
    )

    dev_acc = balanced_accuracy_score(datasets["dev"]["target"], datasets["dev"]["predictions"])
    logger.info(f"Balanced Accuracy Score for VALIDATION (TOTAL): {dev_acc}")

    logger.info("Getting test data predictions")
    test_word_sequences = sequence_padding(
        datasets["test"]["title"], word_index, max_sequence_len
    )
    datasets["test"]["predictions"] = model.predict(
        test_word_sequences, batch_size=batch_size, verbose=0
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
    parser.add_argument("--batch-size", "-b", default=2048, type=int)
    parser.add_argument("--drop-columns", "-d",
                        default=["split", "language", "words", "pos"],
                        nargs="+")
    parser.add_argument("--epochs", "-e", default=20, type=int)
    parser.add_argument("--filters", "-f", default=[2, 3, 4, 5, 6], type=int, nargs="+")
    parser.add_argument("--filter-count", "-c", default=256, type=int)
    parser.add_argument("--keep-punctuation", "-k", action="store_true")
    parser.add_argument("--keep-stopwords", "-s", action="store_true")
    parser.add_argument("--max-sequence-len", "-m", default=10, type=int)
    parser.add_argument("--optimizer", "-o", default="nadam")
    parser.add_argument("--padding", "-p", default="valid")
    parser.add_argument("--reg-lambda", "-r", default=0, type=float)
    parser.add_argument("--unreliable-sampling", "-u", default=0.25, type=float)

    args = parser.parse_args()

    main(**vars(args))
