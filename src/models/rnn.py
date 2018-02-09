from keras.layers import Dense, Embedding, Input, Activation, BatchNormalization
from keras.layers import Bidirectional, Dropout, CuDNNGRU, TimeDistributed, SpatialDropout1D
from keras.models import Model
from keras.optimizers import RMSprop

from .base import ClassificationModel
from .layers import Attention


class BiGRUSimple(ClassificationModel):
    """Simple Bi-GRU model with Dense penultimate layer

    Cell implementation: CuDNNGRU
    """

    def __init__(self, embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
        """Initialize/Build model

        Args:
          - embedding_matrix
          - sequence_length
          - dropout_rate
          - recurrent_units
          - dense_size
        """
        input_layer = Input(shape=(sequence_length,))
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                    weights=[embedding_matrix], trainable=False)(input_layer)
        x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
        x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_size, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        output_layer = Dense(6, activation="sigmoid")(x)

        self._model = Model(inputs=input_layer, outputs=output_layer)


    def compile(self):
        self._model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(clipvalue=1, clipnorm=1),
            metrics=['accuracy']
        )


class BiGRUAtt(ClassificationModel):
    """Simple Bi-GRU model with Dense penultimate layer

    Cell implementation: CuDNNGRU
    """

    def __init__(self, embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
        """Initialize/Build model

        Args:
          - embedding_matrix
          - sequence_length
          - dropout_rate
          - recurrent_units
          - dense_size
        """
        input_layer = Input(shape=(sequence_length,))
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                    # weights=[embedding_matrix], trainable=True)(input_layer)
                                    trainable=True)(input_layer)
        x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
        x = SpatialDropout1D(dropout_rate)(x)
        x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
        x = Attention()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_size, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        output_layer = Dense(6, activation="sigmoid")(x)

        self._model = Model(inputs=input_layer, outputs=output_layer)


    def compile(self):
        self._model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(rho=0.75, clipvalue=1, clipnorm=1),
            metrics=['accuracy']
        )
