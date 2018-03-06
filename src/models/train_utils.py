from keras.callbacks import Callback
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def train_folds(X, y, fold_count, batch_size, get_model_func):
    """ K-Fold Cross-Validation for Keras Models

    Inspired by PavelOstyakov
    https://github.com/PavelOstyakov/toxic/blob/master/toxic/train_utils.py

    """
    fold_size = len(X[0]) // fold_count
    models = []

    for fold_id in range(0, fold_count):
        print('===== FOLD {} ====='.format(fold_id+1))
        model = get_model_func()
        model.compile()
        RocAuc = RocAucEvaluation()
        RocAuc.set_model(model)
        model.fit(
            X, y, validation_split=max(1/fold_count, 0.15),
            batch_size=batch_size, epochs=20, shuffle=True,
            add_callbacks=[RocAuc], verbose=1
        )
        models.append(model)

    return models


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=None, interval=1):
        super(RocAucEvaluation, self).__init__()

        self.interval = interval
        # self.X_val, self.y_val = validation_data or (None, None)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.model.validation_data[0], verbose=0)
            score = roc_auc_score(self.model.validation_data[1], y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))