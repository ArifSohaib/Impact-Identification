"""
This script generates extracted features for each video, which other
models make use of.
You can change you sequence length and limit to a set number of classes
below.
class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np 
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import imageio

base_model = InceptionV3(weights='imagenet', include_top=True)

model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('avg_pool').output)


#load the images
vid = imageio.get_reader('U18 vs Waterloo Period 1 299.mp4')

# Now loop through and extract features to build the sequence.
sequence = []
for _ in tqdm(range(vid.get_meta_data()['nframes'])):
    image = np.expand_dims(vid.get_next_data(), axis=0)
    features = model.predict(image)
    sequence.append(features)

# Save the sequence.
np.savetxt("period1", sequence)

