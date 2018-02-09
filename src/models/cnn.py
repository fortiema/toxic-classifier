from keras.layers import Dense, Embedding, Input, Activation, BatchNormalization, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.models import Model
from keras.optimizers import RMSprop

from .base import ClassificationModel


class BaseConv2D(ClassificationModel):
    """Basic CNN with 4 filters to analyze 3-6 ngrams
    """

    def __init__(self, embedding_matrix, sequence_length, dropout_rate, conv_units=50, dense_size=32):
        input_layer = Input(shape=(1, sequence_length,))
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                    weights=[embedding_matrix], trainable=True)(input_layer)


        x1 = Conv2D(conv_units, (3, embedding_matrix.shape[1]), data_format='channels_first')(embedding_layer)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling2D((int(int(x1.shape[2])  / 1.5), 1), data_format='channels_first')(x1)
        x1 = Flatten()(x1)

        x2 = Conv2D(conv_units, (4, embedding_matrix.shape[1]), data_format='channels_first')(embedding_layer)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = MaxPooling2D((int(int(x2.shape[2])  / 1.5), 1), data_format='channels_first')(x2)
        x2 = Flatten()(x2)

        x3 = Conv2D(conv_units, (5, embedding_matrix.shape[1]), data_format='channels_first')(embedding_layer)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = MaxPooling2D((int(int(x3.shape[2])  / 1.5), 1), data_format='channels_first')(x3)
        x3 = Flatten()(x3)

        x4 = Conv2D(conv_units, (6, embedding_matrix.shape[1]), data_format='channels_first')(embedding_layer)
        x4 = BatchNormalization()(x4)
        x4 = Activation('relu')(x4)
        x4 = MaxPooling2D((int(int(x4.shape[2])  / 1.5), 1), data_format='channels_first')(x4)
        x4 = Flatten()(x4)

        x = concatenate([x1, x2, x3, x4])
        x = Dense(dense_size, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(6, activation="sigmoid")(x)

        self._model = Model(inputs=input_layer, outputs=x)

    def compile(self):
        self._model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(clipvalue=3, clipnorm=2),
            metrics=['accuracy']
        )
