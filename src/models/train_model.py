# -*- coding: utf-8 -*-
import datetime
import os
import logging

import click
from keras.preprocessing.text import Tokenizer, hashing_trick
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

from src.models.embeddings import get_embeddings
from src.models.zoo import *
from src.models.train_utils import train_folds


UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


@click.command()
@click.argument('train_file_path', type=click.Path(exists=True))
@click.argument('test_file_path', type=click.Path(exists=True))
@click.argument('embedding_path', type=click.Path(exists=True))
@click.option('--max-vocab', type=click.INT, default=15000)
@click.option('--embed-size', type=click.INT, default=100)
@click.option('--result-path', type=click.Path(), default='models')
@click.option('--char-model', is_flag=True)
@click.option('--batch-size', '-b', type=click.INT, default=256)
@click.option('--sentences-length', type=click.INT, default=192)
@click.option('--recurrent-units', type=click.INT, default=128)
@click.option('--dropout-rate', type=click.FLOAT, default=0.3)
@click.option('--dense-size', type=click.INT, default=32)
@click.option('--fold-count', '-f', type=click.INT, default=10)
def main(train_file_path, test_file_path, embedding_path, result_path, char_model,
         batch_size, sentences_length, recurrent_units, dropout_rate,
         dense_size, fold_count, max_vocab, embed_size):

    if fold_count <= 1:
        raise ValueError("fold-count should be more than 1")

    click.echo("Loading data...")
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    list_sentences_train = train_data["comment_text"].values.astype(str)
    list_sentences_test = test_data["comment_text"].values.astype(str)
    y_train = train_data[CLASSES].values.astype(float)

    click.echo("Preparing tokenizers...")
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(list(list_sentences_train)+list(list_sentences_test))

    if char_model:
        char_tokenizer = Tokenizer(num_words=52, char_level=True)
        char_tokenizer.fit_on_texts(list(list_sentences_train)+list(list_sentences_test))

    click.echo("Tokenizing sentences in train set...")
    tokenized_sentences_train = tokenizer.texts_to_sequences(list_sentences_train)
    X_train = pad_sequences(tokenized_sentences_train, maxlen=sentences_length)

    if char_model:
        click.echo("Preparing char-level input in train set...")
        char_tokenized_sentences_train = char_tokenizer.texts_to_sequences(list_sentences_train)
        X_train_char = pad_sequences(char_tokenized_sentences_train, maxlen=sentences_length*5)

    click.echo("Tokenizing sentences in test set...")
    tokenized_sentences_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_test = pad_sequences(tokenized_sentences_test, maxlen=sentences_length)

    if char_model:
        click.echo("Preparing char-level input in test set...")
        char_tokenized_sentences_test = char_tokenizer.texts_to_sequences(list_sentences_test)
        X_test_char = pad_sequences(char_tokenized_sentences_test, maxlen=sentences_length*5)

    click.echo("Loading embeddings...")
    embedding_index, emb_mean, emb_std = get_embeddings(embedding_path, embed_size)

    click.echo("Preparing data...")
    word_index = tokenizer.word_index
    nb_words = min(max_vocab, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():
        if i >= max_vocab: continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    get_model_func = lambda: LSTMCNN(
        embedding_matrix,
        sentences_length,
        dropout_rate,
        recurrent_units,
        dense_size)

    # get_model_func = lambda: MixedCharWordModel(
    #     embedding_matrix,
    #     len(char_tokenizer.word_index),
    #     sentences_length,
    #     dropout_rate
    # )

    # get_model_func = lambda: CharLSTMCNN(
    #     sentences_length*5,
    #     len(char_tokenizer.word_index),
    #     dropout_rate,
    #     recurrent_units,
    #     dense_size
    # )

    click.echo('Model Parameters:')

    click.echo("Starting to train models...")
    if char_model:
        models = train_folds(X_train_char, y_train, fold_count, batch_size, get_model_func)
        # models = train_folds([X_train, X_train_char], y_train, fold_count, batch_size, get_model_func)
    else:
        models = train_folds(X_train, y_train, fold_count, batch_size, get_model_func)

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    click.echo("Predicting results...")
    test_predicts_list = []
    ts = datetime.datetime.now().isoformat()
    for fold_id, model in enumerate(models):
        dir_model = os.path.join(result_path, ts, str(fold_id))
        os.makedirs(dir_model, exist_ok=True)
        model.save(dir_model)

        test_predicts_path = os.path.join(dir_model, "test_predicts.npy")
        if char_model:
            test_predicts = model.predict(X_test_char, batch_size=batch_size)
            # test_predicts = model.predict([X_test, X_test_char], batch_size=batch_size)
        else:
            test_predicts = model.predict(X_test, batch_size=batch_size)
        test_predicts_list.append(test_predicts)
        np.save(test_predicts_path, test_predicts)

    test_predicts = np.ones(test_predicts_list[0].shape)
    for fold_predict in test_predicts_list:
        test_predicts *= fold_predict

    test_predicts **= (1. / len(test_predicts_list))

    test_ids = test_data["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + CLASSES]
    submit_path = os.path.join(result_path, ts, "submission.csv")
    test_predicts.to_csv(submit_path, index=False)


if __name__ == "__main__":
    main()
