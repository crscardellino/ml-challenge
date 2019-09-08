#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from nltk.corpus import stopwords
from os import path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

np.random.seed(42)

logger = logging.getLogger(__name__)


def get_logger(output, experiment):
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s")

    handler = logging.FileHandler(path.join(output, f"{experiment}.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def label_encoder(*dfs):
    labels = pd.concat(dfs)["category"].tolist()
    lbl_enc = LabelEncoder().fit(labels)

    return lbl_enc


def load_data(file_path, language):
    df = pd.read_parquet(file_path)
    return df[df["language"] == language]


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

    logger.info("Encoding labels")
    lbl_enc = label_encoder(train_df, dev_df)
    train_df["target"] = lbl_enc.transform(train_df["category"])
    dev_df["target"] = lbl_enc.transform(dev_df["category"])

    logger.info("Setting up normalized titles")
    train_df["normalized_title"] = train_df["words"].apply(lambda words: " ".join(words))
    dev_df["normalized_title"] = dev_df["words"].apply(lambda words: " ".join(words))
    test_df["normalized_title"] = test_df["words"].apply(lambda words: " ".join(words))

    logger.info("Setting up randomized search")

    pipeline = [['vectorizer', CountVectorizer()]]
    param_grid = {
        "vectorizer__analyzer": ["word", "char_wb"],
        "vectorizer__stop_words": [None, stopwords.words(config.language)],
        "vectorizer__strip_accents": ["unicode", None],
        "vectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "vectorizer__min_df": [2, 3],
        "vectorizer__max_features": [150000, 20000, 30000]
    }

    if config.tfidf:
        pipeline.append(['reweighting', TfidfTransformer()])

    if config.lsa:
        pipeline.append(['dim_reduce', TruncatedSVD()])
        param_grid.update({
            "dim_reduce__n_components": [100, 150, 200],
            "dim_reduce__algorithm": ["randomized", "arpack"],
            "dim_reduce__n_iter": [5, 10, 15],
            "dim_reduce__random_state": [42]
        })

    pipeline.append(['clf', LinearSVC()])
    param_grid.update({
        "clf__loss": ["hinge", "squared_hinge"],
        "clf__dual": [True],
        "clf__C": [2.0, 1.0, 0.5, 0.25],
        "clf__class_weight": [None, "balanced"],
        "clf__max_iter": [500, 1000, 1500, 2000],
        "clf__random_state": [42]
    })

    def fit_estimator(params):
        estimator = Pipeline(steps=pipeline)
        estimator.set_params(**params)
        estimator.fit(train_df["normalized_title"], train_df["target"])
        bacc = balanced_accuracy_score(
            dev_df["target"],
            estimator.predict(dev_df["normalized_title"])
        )

        return (estimator, bacc, params)

    logger.info("Fitting models")
    search = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(fit_estimator)(param)
        for param in ParameterSampler(param_grid, config.random_iterations, 42)
    )
    search = sorted(search, key=lambda est: est[1], reverse=True)

    logger.info("Model finished trainig. Getting final predictions.")
    logger.info(f"Best model score: {search[0][1]}")
    logger.info(f"Best model parameters: {search[0][2]}")

    logger.info("Getting training data predictions")
    train_df["predictions"] = search[0][0].predict(train_df["normalized_title"])

    logger.info("Getting dev data predictions")
    dev_df["predictions"] = search[0][0].predict(dev_df["normalized_title"])

    logger.info("Getting test data predictions")
    test_df["predictions"] = search[0][0].predict(test_df["normalized_title"])

    train_acc = balanced_accuracy_score(train_df["target"], train_df["predictions"])
    logger.info(f"Balanced Accuracy Score for TRAINING: {train_acc}")

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

    model_save_path = path.join(config.output, f"{experiment}_model.jb")
    logger.info(f"Saving model to {model_save_path}")
    joblib.dump(search[0][0], model_save_path)

    test_df["category"] = lbl_enc.inverse_transform(test_df["predictions"])
    results_save_path = path.join(config.output, f"{experiment}_results.csv")
    test_df[["id", "category"]].to_csv(results_save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_reliable_data")
    parser.add_argument("dev_data")
    parser.add_argument("test_data")
    parser.add_argument("language")
    parser.add_argument("output")
    parser.add_argument("--train-unreliable-data", "-t", default=None)
    parser.add_argument("--unreliable-data-size", "-u", default=5, type=float)
    parser.add_argument("--tfidf", "-i", action="store_true")
    parser.add_argument("--lsa", "-l", action="store_true")
    parser.add_argument("--random-iterations", "-r", default=10, type=int)

    config = parser.parse_args()

    main(config)
