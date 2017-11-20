import numpy as np
import pandas as pd
import keras
import keras.optimizers as optimizers
from keras.layers import Input, Dense, Conv1D, MaxPool1D, Dropout, Reshape, Flatten, BatchNormalization, Activation
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from pylab import rcParams
import keras.backend as K
from sklearn import preprocessing


class Autoencoder_model:
    """
    class to build and check autoencoder model
    """

    def __init__(self, input_dim, encoding_dim, mid_acivation):
        """
        function to build and check autoencoder model
        params:
            input_dim = number of dimensions in input
            encoding_dim = number of dimensions to encode
            mid_acivation = the activation function for the middle layers
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.mid_activation = mid_acivation
        self.lam = 1e-4

    def contractive_loss(self, y_pred, y_true):
        model = self.build_autoencoder()
        # mse = K.mean(K.square(y_true - y_pred), axis=1)
        mse = K.mean(K.square(y_true - y_pred), axis=1)

        W = K.variable(value=model.get_layer(
            'encoded').get_weights())  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N

        h = model.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = self.lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return mse + contractive

    # def build_autoencoder(self):
    #     """
    #     defines the autoencoder keras model
    #     """
    #
    #     input_layer = Input(shape=(self.input_dim,))
    #     encoder = Activation('tanh')
    #     encoder = Dense(self.encoding_dim, activation=self.mid_activation)(input_layer)
    #     encoder = BatchNormalization()(encoder)
    #     encoder = Dense(self.encoding_dim, activation=self.mid_activation)(encoder)
    #     encoder = Dropout(0.5)(encoder)
    #     decoder = Dense(self.input_dim, activation='tanh')(encoder)
    #     model =  Model(inputs=input_layer, outputs=decoder)
    #     #for naming convention get the number of intermediate layers(input and output layers not counted)
    #     self.num_layers = len(model.layers) - 2
    #     return model

    def build_autoencoder(self):
        input_size = 8
        hidden_size = 6
        code_size = 4

        input_img = Input(shape=(input_size,))
        hidden_1 = Dense(
            hidden_size, activation=self.mid_activation)(input_img)
        code = Dense(code_size, activation=self.mid_activation)(hidden_1)
        hidden_2 = Dense(hidden_size, activation=self.mid_activation)(code)
        output_img = Dense(input_size)(hidden_2)

        autoencoder = Model(input_img, output_img)
        self.num_layers = len(autoencoder.layers) - 2
        return autoencoder


def scale_data(data):
    """
    scales the given data to similar values
    """

    min_max_scalar = preprocessing.MinMaxScaler()
    x_scaled = min_max_scalar.fit_transform(data)
    x_scaled = x_scaled * 100
    return x_scaled


def get_model():
    activation = 'tanh'
    return Autoencoder_model(input_dim=8, encoding_dim=12, mid_acivation=activation)


def main():
    # train_data = pd.read_csv('data/normal_data.csv')
    train_data = pd.read_csv("data/anomaly_data.csv")
    # test_data = pd.read_csv('data/test_normal_data.csv')
    test_data = pd.read_csv("data/test_anomaly_data.csv")

    min_max_scalar = preprocessing.MinMaxScaler()
    train_df = pd.DataFrame(train_data[train_data.keys()[1:-1].values])
    train_df_norm = min_max_scalar.fit_transform(train_df.values)
    test_df = pd.DataFrame(test_data[test_data.keys()[1:-1]].values)
    test_df_norm = min_max_scalar.fit_transform(test_df.values)

    train_data = train_df_norm
    test_data = test_df_norm

    nb_epoch = 10000
    batch_size = 100
    autoencoder = get_model()
    model = autoencoder.build_autoencoder()
    # model.compile(optimizer=optimizers.Adam(lr=0.0001),loss='mean_squared_error', metrics=['accuracy'])
    model.compile(optimizer='adadelta', loss='mean_squared_error')

    checkpointer = ModelCheckpoint(filepath="checkpoints/autoencoder_model.h5",
                                   verbose=1,
                                   save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)
    history = model.fit(x=train_data + (np.random.rand(train_data.shape[0], train_data.shape[1]) / 10.0), y=train_data,
                        epochs=nb_epoch, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[checkpointer, tensorboard]).history
    model.save_weights('weights/autoencoder_{}layer_{}_{}embed.h5'.format(
        autoencoder.num_layers, autoencoder.mid_activation, autoencoder.encoding_dim))


if __name__ == '__main__':
    main()
