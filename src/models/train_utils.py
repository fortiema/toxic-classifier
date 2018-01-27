""" K-Fold Cross-Validation for Keras Models

Inspired by PavelOstyakov
https://github.com/PavelOstyakov/toxic/blob/master/toxic/train_utils.py

"""
import numpy as np


def train_folds(X, y, fold_count, batch_size, get_model_func):
    fold_size = len(X) // fold_count
    models = []

    for fold_id in range(0, fold_count):
        print('===== FOLD {} ====='.format(fold_id))
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = X[fold_start:fold_end]
        val_y = y[fold_start:fold_end]

        model = get_model_func()
        model.compile()
        model.fit(
            train_x, train_y, validation_data=(val_x, val_y),
            batch_size=batch_size, epochs=10, shuffle=True
        )
        models.append(model)

    return models
