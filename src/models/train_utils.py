""" K-Fold Cross-Validation for Keras Models

Inspired by PavelOstyakov
https://github.com/PavelOstyakov/toxic/blob/master/toxic/train_utils.py

"""
import numpy as np


def train_folds(X, y, fold_count, batch_size, get_model_func):
    fold_size = len(X[0]) // fold_count
    models = []

    for fold_id in range(0, fold_count):
        print('===== FOLD {} ====='.format(fold_id))
        model = get_model_func()
        model.compile()
        model.fit(
            X, y, validation_split=1/fold_count,
            batch_size=batch_size, epochs=10, shuffle=True
        )
        models.append(model)

    return models
