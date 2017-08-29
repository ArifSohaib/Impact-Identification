import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Conv1D, MaxPool1D, Dropout, Reshape, Flatten, BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.models import Model 
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from pylab import rcParams
import keras.backend as K
from sklearn import preprocessing

class Post_Autoencoder:
    """
    class uses outputs from autoencoder model in a classifier
    """
    def build_model(self):
        input_layer = Input(shape=(8,))
        x = Dense(128,activation='elu')(input_layer)
        x = Dense(64, activation='elu')(x)
        x = Dense(2, activation='relu')(x)
        return Model(inputs=input_layer, outputs=x)

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

        W = K.variable(value=model.get_layer('encoded').get_weights())  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        
        h = model.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = self.lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return mse + contractive

    def build_autoencoder(self):
        """
        defines the autoencoder keras model
        """
        
        input_layer = Input(shape=(self.input_dim,))
        
        # encoder = Dense(self.encoding_dim, activation=self.mid_activation)(input_layer)
        encoder = Reshape((-1,1))(input_layer)
        encoder = Conv1D(filters=32,kernel_size=4)(encoder)
        
        # decoder = Conv1D(filters=32,kernel_size=1)(encoder)
        decoder = Flatten()(encoder)
        # decoder = Dense(self.encoding_dim,activation=self.mid_activation)(decoder)
        decoder = Dense(self.input_dim)(decoder)
        
        model =  Model(inputs=input_layer, outputs=decoder)
        #for naming convention get the number of intermediate layers(input and output layers not counted)
        self.num_layers = len(model.layers) - 2
        return model

def scale_data(data):
    """
    scales the given data to similar values
    """

    min_max_scalar = preprocessing.MinMaxScaler()
    x_scaled = min_max_scalar.fit_transform(data)
    return x_scaled

def get_model():
    activation = 'elu'
    return Autoencoder_model(input_dim=8, encoding_dim=6, mid_acivation=activation)

def main():
    data = pd.read_csv('data/normal_data.csv')
    df = data[data.keys()[1:-1]]
    df_norm = (df - df.mean()) / (df.max() - df.min())

    data = df_norm.values
    # data = scale_data(data)
    nb_epoch = 100
    batch_size = 128
    autoencoder = get_model()
    model = autoencoder.build_autoencoder()
    model.compile(optimizer='rmsprop',loss='mean_squared_error', metrics=['accuracy'])
    
    checkpointer = ModelCheckpoint(filepath="checkpoints/autoencoder_model.h5",
                               verbose=1,
                               save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
    history = model.fit(data, data, epochs=nb_epoch, batch_size=batch_size, shuffle=True, validation_data=(data, data), verbose=1, callbacks=[checkpointer, tensorboard]).history
    model.save_weights('weights/autoencoder_{}layer_{}_{}embed.h5'.format(autoencoder.num_layers, autoencoder.mid_activation, autoencoder.encoding_dim))


if __name__ == '__main__':
    main()