from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers.recurrent import LSTM
import data_generator

def lstm():
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    # Model.
    model = Sequential()
    model.add(LSTM(4096, return_sequences=True, input_shape=(1,2048)))
    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model

def main():
    lstm_model = lstm()
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy')
    gen = data_generator.EmbeddingGenerator(embedding_file='period1',frame_file='frames_vid1')
    lstm_model.fit_generator(generator=gen.generate_embeddings_skip(batch_size=50, skip_prob=0.9), steps_per_epoch=3000)
    lstm_model.save_weights('lstm_model_weights.h5')

if __name__ == '__main__':
    main()

