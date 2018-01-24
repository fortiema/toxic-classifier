from keras.layers import Dense, Embedding, Input
from keras.layers import Bidirectional, Dropout, CuDNNGRU
from keras.models import Model
from keras.optimizers import RMSprop

from .base import ClassificationModel


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
        x = Dropout(dropout_rate)(x)
        x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(x)
        x = Dense(dense_size, activation="relu")(x)
        output_layer = Dense(6, activation="sigmoid")(x)

        self.model = Model(inputs=input_layer, outputs=output_layer)


    def compile(self):
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(clipvalue=1, clipnorm=1),
            metrics=['accuracy']
        )

    @property
    def model(self):
        return self.model
