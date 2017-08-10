"""
module to load the video data to any model
"""
import numpy as np 
import imageio
import pandas as pd
#load the frame numbers

import pickle
def load_dict(name ):
    with open('frames/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
# frame_nums_train = load_dict('frames_vid1')
# frame_nums_val= load_dict('frames_vid2')

def get_impact_frames(file_name):
    """
    Returns a list containing the frames that have the impact in that second
    """
    frame_nums = load_dict(file_name)
    impact_frames = list()
    sorted_frames = sorted(frame_nums.values())
    for val in sorted_frames:
        #in the impact_frames, add 30 values before the frame and 30 after the frame
        for idx in range(val-30, val +30):       
            impact_frames.append(idx)
    return impact_frames

def make_data(video_file):
    """
    Makes a numpy file to store the compressed data
    """
    vid = imageio.get_reader(video_file, 'ffmpeg')
    vid_frame_count = 0
    impact_frame_count = 0
    features = np.zeros(shape=(vid.get_meta_data()['nframes'], vid.get_meta_data()['source_size'][1], vid.get_meta_data()['source_size'][0], 3))
    labels = np.zeros(shape=(vid.get_meta_data()['nframes'], 2))
    for data in vid.iter_data():
        if vid_frame_count == impact_frames[impact_frame_count]:
            labels[vid_frame_count] = [0,1]
        else:
            labels[vid_frame_count] = [1,0]
        features[vid_frame_count] = np.array(data)
    return features, labels

def generate_data(video_file= './U18 vs Waterloo Period 1 smaller.mp4', batch_size=100, impact_file = 'frames_vid1'):
    """
    generates video frames and labels to train/validate model(use different function to test)
    yields:
        x: video frame
        y: label
    args:
        video_file: the file containing the video to process
        frame_file: the frame containing the labeled impact
    """
    impact_frames = get_impact_frames(impact_file)
    vid = imageio.get_reader(video_file, 'ffmpeg')
    vid_frame_count = 0
    impact_frame_count = 0
    batch_count = 0
    batch_features = np.zeros((batch_size, vid.get_meta_data()['source_size'][1], vid.get_meta_data()['source_size'][0], 3))
    batch_labels = np.zeros((batch_size,1))
    while True:
        try:
            if batch_count < batch_size:
                if vid_frame_count == impact_frames[impact_frame_count]:
                    batch_labels[batch_count] = 1
                    impact_frame_count+=1
                    vid_frame_count+=1
                else:
                    batch_labels[batch_count] = 0
                    vid_frame_count += 1
                batch_features[batch_count] = np.array(vid.get_next_data())
                batch_count += 1
            else:
                batch_count = 0
                yield batch_features, batch_labels
        except IndexError:
            yield batch_features, batch_labels
            vid_frame_count = 0
            impact_frame_count = 0
            batch_count = 0

def generate_data_onehot(video_file= './U18 vs Waterloo Period 1 smaller.mp4', batch_size=100, impact_file = 'frames_vid1'):
    """
    generates video frames and labels to train/validate model(use different function to test)
    yields:
        x: video frame
        y: label
    args:
        video_file: the file containing the video to process
        frame_file: the frame containing the labeled impact
    """
    impact_frames = get_impact_frames(impact_file)
    vid = imageio.get_reader(video_file, 'ffmpeg')
    vid_frame_count = 0
    impact_frame_count = 0
    batch_count = 0
    batch_features = np.zeros((batch_size, vid.get_meta_data()['source_size'][1], vid.get_meta_data()['source_size'][0], 3))
    batch_labels = np.zeros((batch_size,2))
    while True:
        try:
            if batch_count < batch_size:
                if vid_frame_count == impact_frames[impact_frame_count]:
                    batch_labels[batch_count] = [1,0]
                    impact_frame_count+=1
                    vid_frame_count+=1
                else:
                    batch_labels[batch_count] = [0,1]
                    vid_frame_count += 1
                batch_features[batch_count] = np.array(vid.get_next_data())
                batch_count += 1
            else:
                batch_count = 0
                yield batch_features, batch_labels
        except IndexError:
            yield batch_features, batch_labels
            vid_frame_count = 0
            impact_frame_count = 0
            batch_count = 0

def generate_embeddings(embedding_file= 'period1', batch_size=100, impact_file = 'frames_vid1'):
    """
    generates video frames and labels to train/validate model(use different function to test)
    yields:
        x: embedding
        y: label
    args:
        video_file: the file containing the video to process
        frame_file: the frame containing the labeled impact
    """
    data = pd.read_csv(embedding_file, sep=" ", header=None)
    impact_frames = get_impact_frames(impact_file)
    vid_frame_count = 0
    impact_frame_count = 0
    batch_count = 0
    # batch_features = np.zeros((batch_size, 2048))
    batch_labels = np.zeros((batch_size,2))
    while True:
        try:
            if batch_count < batch_size:
                #if the current_frame is one of the frames containing the impact
                if vid_frame_count == impact_frames[impact_frame_count]:
                    batch_labels[batch_count] = [1,0]
                    impact_frame_count+=1
                    
                else:
                    batch_labels[batch_count] = [0,1]

                vid_frame_count += 1
                # batch_features[batch_count] = data[batch_count:vid_frame_count]
                batch_count += 1
            else:
                batch_count = 0
                yield np.expand_dims(data[vid_frame_count-batch_size:vid_frame_count], axis=1), batch_labels
        except IndexError:
            # yield batch_features, batch_labels
            yield np.expand_dims(data[vid_frame_count-batch_size:vid_frame_count], axis=1), batch_labels
            #restart 
            vid_frame_count = 0
            impact_frame_count = 0
            batch_count = 0

def test_generator(generator, iterations, print_data=False):
    """
    Tests the given generator for the specified number of iterations
    """
    count = 0
    for data in generator:
        while count < iterations:
            print(count)
            if print_data:
                print(data)
            count+=1
        break 

class DataGenerator():
    """
    class creaates a generator
    video_file: the video to generate images from
    impact_file: the file containing the labeled impact frames
    """
    def __init__(self, video_file, impact_file):
        self.vid = imageio.get_reader(video_file, 'ffmpeg')
        self.impact_frames = get_impact_frames(impact_file)

    def generate_data_onehot_skip(self, batch_size=100, skip_prob=0.5):
        """
        generates video frames and labels to train/validate model(use different function to test)
        yields:
            x: video frame
            y: label
        args:
            video_file: the file containing the video to process
            frame_file: the frame containing the labeled impact
            skip_prob: the probablity of skipping frames without impact
        """
        if skip_prob > 1.0 or skip_prob < 0.0:
            print("invalid skip probablity")
            return
        vid_frame_count = 0
        impact_frame_count = 0
        batch_count = 0
        batch_features = np.zeros((batch_size, self.vid.get_meta_data()['source_size'][1], self.vid.get_meta_data()['source_size'][0], 3))
        batch_labels = np.zeros((batch_size,2))
        while True:
            try:
                if batch_count < batch_size:
                    if vid_frame_count == self.impact_frames[impact_frame_count]:
                        #if there is an impact, set the label to [1,0], add the image to the batch, then go to next frame
                        batch_labels[batch_count] = [1,0]
                        batch_features[batch_count] = np.array(self.vid.get_next_data())
                        impact_frame_count+=1
                        vid_frame_count += 1
                    else:
                        #if there isn't an impact, generate a random number
                        rand = np.random.uniform(low=0.0, high=1.0)
                        
                        if (rand > skip_prob):
                            #if the random number is above the probablity, set the label to [0,1], add the image to batch then go to next frame
                            batch_labels[batch_count] = [0,1]
                            batch_features[batch_count] = np.array(self.vid.get_next_data())
                            vid_frame_count +=1
                        else:
                            #if the random number is lower, skip the frame and go to next frame
                            vid_frame_count +=1
                    batch_count += 1
                    
                else:
                    #id the batch is filled, then just yield it
                    batch_count = 0
                    yield batch_features, batch_labels
            except IndexError:
                yield batch_features, batch_labels
                vid_frame_count = 0
                impact_frame_count = 0
                batch_count = 0

class EmbeddingGenerator():
    def __init__(embedding_file, frame_file):
        this.embedding_file = embedding_file
        this.frame_file = frame_file

    def generate_embeddings(embedding_file= 'period1', batch_size=100, impact_file = 'frames_vid1'):
        """
        generates video frames and labels to train/validate model(use different function to test)
        yields:
            x: embedding
            y: label
        args:
            video_file: the file containing the video to process
            frame_file: the frame containing the labeled impact
        """
        data = pd.read_csv(self.embedding_file, sep=" ", header=None)
        impact_frames = get_impact_frames(self.impact_file)
        vid_frame_count = 0
        impact_frame_count = 0
        batch_count = 0
        # batch_features = np.zeros((batch_size, 2048))
        batch_labels = np.zeros((batch_size,2))
        while True:
            try:
                if batch_count < batch_size:
                    #if the current_frame is one of the frames containing the impact
                    if vid_frame_count == impact_frames[impact_frame_count]:
                        batch_labels[batch_count] = [1,0]
                        impact_frame_count+=1
                    
                    else:
                        batch_labels[batch_count] = [0,1]

                    vid_frame_count += 1
                    # batch_features[batch_count] = data[batch_count:vid_frame_count]
                    batch_count += 1
                else:
                    batch_count = 0
                    yield np.expand_dims(data[vid_frame_count-batch_size:vid_frame_count], axis=1), batch_labels
            except IndexError:
                # yield batch_features, batch_labels
                yield np.expand_dims(data[vid_frame_count-batch_size:vid_frame_count], axis=1), batch_labels
                #restart 
                vid_frame_count = 0
                impact_frame_count = 0
                batch_count = 0