import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Dropout
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

        W = K.variable(value=model.get_layer('encoded').get_weights())  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        
        h = model.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = self.lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return mse + contractive

    # def build_autoencoder(self):
    #     inputs = Input(shape=(self.input_dim,))
    #     encoded = Dense(self.encoding_dim, activation='sigmoid', name='encoded')(inputs)
    #     outputs = Dense(self.input_dim, activation='linear')(encoded)

    #     model = Model(input=inputs, output=outputs)
    #     return model

    def build_autoencoder(self):
        """
        defines the autoencoder keras model
        """
        
        input_layer = Input(shape=(self.input_dim,))
        encoder = Dropout(0.3)(input_layer)
        encoder = Dense(self.encoding_dim, activation='elu', activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(self.encoding_dim*2), activation=self.mid_activation)(encoder)
        encoder = Dense(int(self.encoding_dim*4), activation=self.mid_activation)(encoder)
        encoder = Dense(int(self.encoding_dim*2), activation=self.mid_activation)(encoder)
        decoder = Dense(int(self.encoding_dim), activation='elu')(encoder)
        decoder = Dense(self.input_dim, activation='elu')(decoder)
        
        model =  Model(inputs=input_layer, outputs=decoder)
        #for naming convention get the number of intermediate layers(input and output layers not counted)
        self.num_layers = len(model.layers) - 2
        return model
        # return autoencoder

    def check_autoencoder(self):
        normal_data = pd.read_csv('data/normal_data.csv')
        normal_data = normal_data[normal_data.keys()[1:]].values #key 0 is index so it can be skipped for predictions
        anomaly_data = pd.read_csv('data/anomaly_data.csv')
        anomaly_data = anomaly_data[anomaly_data.keys()[1:]].values
        model = self.build_autoencoder()
        model.compile(optimizer='adam', loss='mean_squared_error')
        anomaly_preds = model.predict(anomaly_data, batch_size=128,verbose=1)
        normal_preds = model.predict(normal_data, batch_size=128,verbose=1)
        return normal_preds, anomaly_preds

def scale_data(data):
    """
    scales the given data to similar values
    """
    # x = data.values
    min_max_scalar = preprocessing.MinMaxScaler()
    x_scaled = min_max_scalar.fit_transform(data)
    # df = pd.DataFrame(x_scaled)
    return x_scaled

def main():
    data = pd.read_csv('data/normal_data.csv')
    data = data[data.keys()[1:-1]].values
    data = scale_data(data)
    nb_epoch = 100
    batch_size = 64
    autoencoder = Autoencoder_model(input_dim=9, encoding_dim=4, mid_acivation='elu')
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