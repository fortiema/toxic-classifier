from keras.layers import Dense, Embedding, Input, Activation, BatchNormalization, Dropout, Flatten, Reshape
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, concatenate, TimeDistributed, GRU, Bidirectional
from keras.models import Model
from keras.optimizers import RMSprop

from .base import ClassificationModel
from .layers import Attention


class MixedCharWordModel(ClassificationModel):
    """ Bi-RNN model using both word-level and char-level embeddings
    """

    def __init__(self, embedding_matrix, char_vocab, sequence_length, dropout_rate=0.3, recurrent_units=64, conv_units=64, dense_size=32):
        # Char-level Embeddings
        char_to_word_factor = 5
        char_input = Input(shape=(sequence_length*char_to_word_factor, char_vocab), name='char_input')
        char_embed_layer = Embedding(char_vocab, 30, trainable=True)(char_input)
        
        x1 = Conv2D(conv_units, (1, 3), padding='valid', activation='tanh', data_format="channels_last")(char_embed_layer)
        x1 = MaxPooling2D((1, char_to_word_factor - 3 + 1), data_format="channels_last")(x1)
        x1 = Reshape((sequence_length*char_to_word_factor, conv_units))(x1)

        x2 = Conv2D(conv_units, (1, 4), padding='valid', activation='tanh', data_format="channels_last")(char_embed_layer)
        x2 = MaxPooling2D((1, char_to_word_factor - 4 + 1), data_format="channels_last")(x2)
        x2 = Reshape((sequence_length*char_to_word_factor, conv_units))(x2)

        x3 = Conv2D(conv_units, (1, 5), padding='valid', activation='tanh', data_format="channels_last")(char_embed_layer)
        x3 = MaxPooling2D((1, char_to_word_factor - 5 + 1), data_format="channels_last")(x3)
        x3 = Reshape((sequence_length*char_to_word_factor, conv_units))(x3)

        x_c = concatenate([x1, x2, x3], axis=1)
        

        # Word-Level Embeddings (Use pretrained by default)
        word_input = Input(shape=(sequence_length,), name='word_input')
        embedding_layer = Embedding(
            embedding_matrix.shape[0],
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=True)(word_input)

        x = concatenate([x_c, embedding_layer])
        x = BatchNormalization()(x)
        x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))(x)
        x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))(x)
        x = Attention()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_size, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        output_layer = Dense(6, activation="sigmoid")(x)

        self._model = Model(inputs=[word_input, char_input], outputs=output_layer)

    def compile(self):
        self._model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=RMSprop(clipvalue=1, clipnorm=1),
            metrics=['accuracy']
        )
