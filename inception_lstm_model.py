from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers.recurrent import LSTM
import data_generator

def lstm():
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    # Model.
    model = Sequential()
    model.add(LSTM(2048, return_sequences=True, input_shape=(1,2048)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model

lstm_model = lstm()
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy')
lstm_model.fit_generator(generator=data_generator.generate_embeddings(), steps_per_epoch=1500)

lstm_model.save_weights('lstm_model_weights.h5')