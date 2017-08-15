"""
Extracts features to use in LSTM model
"""
import numpy as np 
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import imageio


def extract_embedding(filename):
    """
    extracts the avg_pool layer of the inception model as it runs prediction
    the resulting embedding can be used for LSTM network which was shown to have high accuracy
    """
    base_model = InceptionV3(weights='imagenet', include_top=True)

    model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('avg_pool').output)


    #load the images
    vid = imageio.get_reader(filename)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for _ in tqdm(range(vid.get_meta_data()['nframes'])):
        image = np.expand_dims(vid.get_next_data(), axis=0)
        features = model.predict(image)[0]
        sequence.append(features)

    # Save the sequence.
    np.savetxt("data/features/period{}".format(filename[-9]), sequence)

def main():
    extract_embedding('U18 vs Waterloo Period 1.mp4')

if __name__ == '__main__':
    main()