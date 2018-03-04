from keras.layers import Dense, Embedding, Input, Activation, BatchNormalization, Dropout, Flatten, Reshape, Dropout, SpatialDropout1D
from keras.layers import Conv1D, Conv2D, GlobalMaxPooling1D, GlobalMaxPool1D, MaxPooling1D, MaxPooling2D, concatenate, TimeDistributed, GRU, Bidirectional, CuDNNGRU
from keras.layers import CuDNNGRU
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
        char_embedding_dim = 30
        char_input = Input(shape=(1, sequence_length*char_to_word_factor, ), name='char_input')
        char_embed_layer = Embedding(char_vocab, char_embedding_dim, trainable=True)(char_input)
        
        x1 = Conv2D(conv_units, (3, char_embedding_dim), padding='same', activation='tanh', data_format='channels_first')(char_embed_layer)
        # x1 = MaxPooling2D(((char_to_word_factor - 3 + 1), 1), data_format='channels_first')(x1)
        x1 = MaxPooling2D((int(int(x1.shape[2])  / 1.5), 1), data_format='channels_first')(x1)
        # # x1 = Dropout(dropout_rate)(x1)
        # # x1 = Reshape((sequence_length, char_embedding_dim))(x1)
        # # x1 = Flatten()(x1)

        x2 = Conv2D(conv_units, (4, char_embedding_dim), padding='same', activation='tanh', data_format='channels_first')(char_embed_layer)
        # x2 = MaxPooling2D(((char_to_word_factor - 4 + 1), 1), data_format='channels_first')(x2)
        x2 = MaxPooling2D((int(int(x2.shape[2])  / 1.5), 1), data_format='channels_first')(x2)
        # # x2 = Dropout(dropout_rate)(x2)
        # # x2 = Reshape((sequence_length, char_embedding_dim))(x2)
        # # x2 = Flatten()(x2)

        x3 = Conv2D(conv_units, (5, char_embedding_dim), padding='same', activation='tanh', data_format='channels_first')(char_embed_layer)
        # x3 = MaxPooling2D(((char_to_word_factor - 5 + 1), 1), data_format='channels_first')(x3)
        x3 = MaxPooling2D((int(int(x3.shape[2])  / 1.5), 1), data_format='channels_first')(x3)
        # x3 = Dropout(dropout_rate)(x3)
        # x3 = Reshape((sequence_length, char_embedding_dim))(x3)
        # x3 = Flatten()(x3)

        x_c = concatenate([x1, x2, x3])
        x_c = Reshape((sequence_length, char_embedding_dim))(x_c)
        

        # Word-Level Embeddings (Use pretrained by default)
        word_input = Input(shape=(sequence_length,), name='word_input')
        embedding_layer = Embedding(
            embedding_matrix.shape[0],
            embedding_matrix.shape[1],
            # weights=[embedding_matrix],
            trainable=True)(word_input)

        # x = concatenate([x_c, embedding_layer])
        x = BatchNormalization()(x_c)
        # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
        x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))(x)
        x = GlobalMaxPooling1D()(x)
        # x = Attention()(x)
        # x = Dropout(dropout_rate)(x)
        x = Dense(dense_size*2, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_size, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        output_layer = Dense(6, activation="sigmoid")(x)

        self._model = Model(inputs=[word_input, char_input], outputs=output_layer)

    def compile(self):
        self._model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(clipvalue=1, clipnorm=1),
            metrics=['accuracy']
        )


class CharCNNLSTM(ClassificationModel):
    """Char-level CNN + LSTM
    """

    def __init__(self, sequence_length, char_vocab, dropout_rate=0.25, recurrent_units=64, conv_units=64, dense_size=32):
        input_layer = Input(shape=(sequence_length,))
        embedding_layer = Embedding(char_vocab, 128)(input_layer)

        x = Conv1D(conv_units, 4, padding='same', activation='relu')(embedding_layer)
        x = MaxPooling1D(pool_size=4)(x)

        x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))(x)
        x = Attention()(x)
        # x = GlobalMaxPool1D()(x)

        x = BatchNormalization()(x)
        x = Dense(dense_size, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(6, activation="sigmoid")(x)

        self._model = Model(inputs=input_layer, outputs=x)

    def compile(self):
        self._model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )


class CharLSTMCNN(ClassificationModel):
    """Char-level CNN + LSTM
    """

    def __init__(self, sequence_length, char_vocab, dropout_rate=0.25, recurrent_units=64, conv_units=64, dense_size=32):
        input_layer = Input(shape=(sequence_length,))
        embedding_layer = Embedding(char_vocab, 128)(input_layer)   

        x = SpatialDropout1D(0.2)(embedding_layer)

        x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
        # x = Attention()(x)
        # x = GlobalMaxPool1D()(x)

        x = Conv1D(conv_units, 4, padding='same', activation='relu')(embedding_layer)
        x = MaxPooling1D(pool_size=4)(x)

        x = BatchNormalization()(x)
        x = Dense(dense_size, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(6, activation="sigmoid")(x)

        self._model = Model(inputs=input_layer, outputs=x)

    def compile(self):
        self._model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )