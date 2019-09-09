#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from gensim.models import KeyedVectors
from os import path
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate, Conv1D, Dense,\
    GlobalMaxPooling1D, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

np.random.seed(42)
tf.compat.v1.random.set_random_seed(42)

logger = logging.getLogger(__name__)


def build_cnn(config, vocab_size, vector_size, embedding_matrix, output_size):
    embedding_layer = Embedding(vocab_size, vector_size,
                                weights=[embedding_matrix],
                                input_length=config.max_sequence_len,
                                trainable=False)

    sequence_input = Input(shape=(config.max_sequence_len,))
    embedded_sequences = embedding_layer(sequence_input)

    layers = []
    for filter_size in config.filters:
        layer = Conv1D(
            config.filter_count,
            filter_size,
            activation=config.activation,
            padding=config.padding,
            # kernel_regularizer=regularizers.l2(config.reg_lambda)
        )(embedded_sequences)
        layer = GlobalMaxPooling1D()(layer)
        layers.append(layer)

    layer = Concatenate()(layers)

    for _ in range(config.network_size):
        layer = Dense(config.layer_size, activation=config.activation)(layer)

    preds = Dense(output_size, activation="softmax")(layer)
    model = Model(sequence_input, preds)

    return model


def get_logger(output, experiment):
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s")

    handler = logging.FileHandler(path.join(output, f"{experiment}.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_embedding_matrix(word_index, w2v):
    embedding_matrix = np.zeros((len(word_index), w2v.vector_size))

    for word, i in word_index.items():
        if word in w2v and word not in {"NULL", "UNK"}:
            embedding_matrix[i] = w2v[word]
        if word == "UNK":
            embedding_matrix[i] = np.random.normal(size=(w2v.vector_size,))

    return embedding_matrix


def label_encoder(*dfs):
    labels = pd.concat(dfs)["category"].tolist()
    lbl_enc = LabelEncoder().fit(labels)

    return lbl_enc


def load_data(file_path, language):
    df = pd.read_parquet(file_path)
    return df[df["language"] == language]


def sequence_padding(df, word_index, max_len):
    return pad_sequences(
        df["words"].apply(
            lambda words: words_to_sequence(words, word_index)
        ).tolist(), maxlen=max_len
    )


def words_to_idx(all_words, w2v, null_token="NULL", unknown_token="UNK"):
    word_index = {word for words in all_words for word in words if word in w2v}
    word_index = {word: idx for idx, word in enumerate(sorted(word_index), start=1)}
    word_index[null_token] = 0
    word_index[unknown_token] = len(word_index)

    return word_index


def words_to_sequence(words, word_index, default_value="UNK"):
    return [word_index.get(word, word_index[default_value]) for word in words]


def main(config):
    experiment = datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + "-" + config.language
    logger = get_logger(config.output, experiment)
    logger.info(f"Beggining experiments with configuration:\n{vars(config)}")

    logger.info("Loading training dataset")
    train_df = load_data(config.train_reliable_data, config.language)

    logger.info("Loading development dataset")
    dev_df = load_data(config.dev_data, config.language)

    logger.info("Loading test dataset")
    test_df = load_data(config.test_data, config.language)

    if config.train_unreliable_data:
        logger.info("Loading unreliable training dataset")
        trainu_df = load_data(
            config.train_unreliable_data,
            config.language
        )

        logger.info("Sampling unreliable training dataset")
        if config.unreliable_data_size >= 1:
            trainu_df = trainu_df.groupby("category").apply(
                lambda cat: cat.sample(n=int(config.unreliable_data_size))
            ).reset_index(drop=True)
        elif 0 < config.unreliable_data_size < 1:
            trainu_df = trainu_df.groupby("category").apply(
                lambda cat: cat.sample(frac=config.unreliable_data_size)
            ).reset_index(drop=True)
    else:
        trainu_df = pd.DataFrame(columns=train_df.columns)

    train_df = pd.concat([train_df, trainu_df])

    logger.info("Loading word vectors")
    w2v = KeyedVectors.load_word2vec_format(config.word_vectors, binary=True)

    logger.info("Encoding labels")
    lbl_enc = label_encoder(train_df, dev_df)
    train_df["target"] = lbl_enc.transform(train_df["category"])
    dev_df["target"] = lbl_enc.transform(dev_df["category"])

    train_df = train_df[["words", "target"]]
    dev_df = dev_df[["words", "target"]]
    test_df = test_df[["id", "words"]]

    logger.info("Turning words to sequence of indices")
    word_index = words_to_idx(pd.concat([train_df, dev_df, test_df], sort=False)["words"], w2v)
    train_word_sequences = sequence_padding(train_df, word_index, config.max_sequence_len)
    dev_word_sequences = sequence_padding(dev_df, word_index, config.max_sequence_len)
    test_word_sequences = sequence_padding(test_df, word_index, config.max_sequence_len)

    logger.info("Getting one-hot encoded targets")
    train_target = to_categorical(
        train_df["target"].tolist(),
        num_classes=lbl_enc.classes_.shape[0]
    )
    dev_target = to_categorical(
        dev_df["target"].tolist(),
        num_classes=lbl_enc.classes_.shape[0]
    )

    logger.info("Setting up embedding matrix")
    embedding_matrix = get_embedding_matrix(word_index, w2v)

    logger.info("Building network")
    model = build_cnn(config, len(word_index), w2v.vector_size,
                      embedding_matrix, lbl_enc.classes_.shape[0])
    model.summary(print_fn=logger.info)

    logger.info("Cleaning up data to save memory")
    del train_df
    del w2v

    logger.info("Compiling model")
    model.compile(loss="categorical_crossentropy", optimizer=config.optimizer, metrics=["accuracy"])

    logger.info("Fitting model")
    model.fit(
        train_word_sequences, train_target,
        validation_data=(dev_word_sequences, dev_target),
        batch_size=config.batch_size, epochs=config.epochs,
        verbose=1, validation_split=0, validation_freq=5
    )

    logger.info("Model finished trainig. Getting final predictions.")
    # logger.info("Getting training data predictions")
    # train_df["predictions"] = model.predict(
    #     train_word_sequences, batch_size=config.batch_size, verbose=0
    # ).argmax(axis=1)

    logger.info("Getting dev data predictions")
    dev_df["predictions"] = model.predict(
        dev_word_sequences, batch_size=config.batch_size, verbose=0
    ).argmax(axis=1)

    logger.info("Getting test data predictions")
    test_df["predictions"] = model.predict(
        test_word_sequences, batch_size=config.batch_size, verbose=0
    ).argmax(axis=1)

    # train_acc = balanced_accuracy_score(train_df["target"], train_df["predictions"])
    # logger.info(f"Balanced Accuracy Score for TRAINING: {train_acc}")

    dev_acc = balanced_accuracy_score(dev_df["target"], dev_df["predictions"])
    logger.info(f"Balanced Accuracy Score for VALIDATION (TOTAL): {dev_acc}")

    dev_acc_reliable = balanced_accuracy_score(
        dev_df[dev_df["label_quality"] == "reliable"]["target"],
        dev_df[dev_df["label_quality"] == "reliable"]["predictions"]
    )
    logger.info(f"Balanced Accuracy Score for VALIDATION (RELIABLE): {dev_acc_reliable}")

    dev_acc_unreliable = balanced_accuracy_score(
        dev_df[dev_df["label_quality"] == "unreliable"]["target"],
        dev_df[dev_df["label_quality"] == "unreliable"]["predictions"]
    )
    logger.info(f"Balanced Accuracy Score for VALIDATION (UNRELIABLE): {dev_acc_unreliable}")

    model_save_path = path.join(config.output, f"{experiment}_model.h5")
    logger.info(f"Saving model to {model_save_path}")
    model.save(model_save_path)

    test_df["category"] = lbl_enc.inverse_transform(test_df["predictions"])
    results_save_path = path.join(config.output, f"{experiment}_results.csv")
    test_df[["id", "category"]].to_csv(results_save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_reliable_data")
    parser.add_argument("dev_data")
    parser.add_argument("test_data")
    parser.add_argument("language")
    parser.add_argument("word_vectors")
    parser.add_argument("output")
    parser.add_argument("--activation", "-a", default="relu")
    parser.add_argument("--batch-size", "-b", default=1024, type=int)
    parser.add_argument("--epochs", "-e", default=10, type=int)
    parser.add_argument("--filters", "-f", default=[2, 3, 5], type=int, nargs="+")
    parser.add_argument("--filter-count", "-c", default=128, type=int)
    parser.add_argument("--layer-size", "-l", default=1024, type=int)
    parser.add_argument("--max-sequence-len", "-m", default=15, type=int)
    parser.add_argument("--network-size", "-n", default=3, type=int)
    parser.add_argument("--padding", "-p", default="valid")
    parser.add_argument("--optimizer", "-o", default="adam")
    parser.add_argument("--train-unreliable-data", "-t", default=None)
    parser.add_argument("--unreliable-data-size", "-u", default=5, type=float)
    # parser.add_argument("--dropout-ratio", "-d", default=0.3, type=float)
    # parser.add_argument("--reg-lambda", "-r", default=0.01, type=float)

    config = parser.parse_args()

    main(config)
