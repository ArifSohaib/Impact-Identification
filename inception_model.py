
# start coding the inception model
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import data_generator
from sklearn.metrics import f1_score
import numpy as np

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 2 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy')
            
#2723 * 43 = 117089 which is the total number of frames
#this is a workaround to the fact that the last will always be null in the current generator
train_gen = data_generator.DataGenerator(video_file='./U18 vs Waterloo Period 2 smaller.mp4', impact_file='frames_vid2')
test_gen = data_generator.DataGenerator(video_file='./U18 vs Waterloo Period 1 smaller.mp4', impact_file='frames_vid1')

class Metrics(keras.callbacks.Callback):
    """
    class containing the f1 metric
    """
    def on_epoch_end(self, batch, logs={}):
        predict= np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        self.f1s = f1_score(targ, predict)
metrics = Metrics()

model.fit_generator(generator=train_gen.generate_data_onehot_skip( batch_size=100, skip_prob=0.8),
                    steps_per_epoch=1200, 
                    validation_data=test_gen.generate_data_onehot_skip(batch_size=100, skip_prob=0.5),
                    verbose=1,
                    validation_steps=120
                    # callbacks=[metrics]
                    )
model.save('weights.h5')

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='mean_squared_error')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# train_gen = data_generator.DataGenerator(video_file='./U18 vs Waterloo Period 2 smaller.mp4', impact_file='frames_vid2')
# test_gen = data_generator.DataGenerator(video_file='./U18 vs Waterloo Period 1 smaller.mp4', impact_file='franes_vid1')
model.fit_generator(generator=train_gen.generate_data_onehot_skip(batch_size=100, skip_prob=0.8),
                    steps_per_epoch=1200, 
                    # validation_data=test_gen.generate_data_onehot_skip(batch_size=100, skip_prob=0.5),
                    verbose=1,
                    # validation_steps=1200
                    )
model.save('inception_final.h5')