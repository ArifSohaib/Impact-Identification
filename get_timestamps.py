"""
extracts timestamps from predicted data
"""

import numpy as np 
import imageio

pred_file = 'lstm_preds_vid3.npy'
vid_file = 'U18 vs Waterloo Period 3 299.mp4'
output_file = 'frames_period3.csv'

data = np.load(pred_file)

idx = np.where(data[:,0] > data[:,1])[0]

vid = imageio.get_reader(vid_file)

with open(output_file,'w') as out:
    for frame_no in idx:
        # frame_no = frame
        minute = (frame_no-1)//(vid.get_meta_data()['fps']*vid.get_meta_data()['fps'])
        second = (frame_no-1)%(vid.get_meta_data()['fps']*vid.get_meta_data()['fps']) // vid.get_meta_data()['fps']
        print(minute, second)
        out.write("{} minute, {} second\n".format(minute, second))