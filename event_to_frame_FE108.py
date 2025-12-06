import os
import pdb
import shutil

# import csv
import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from dv import AedatFile
import numpy as np
import math


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0")
    data_path = "/wangx/DATA/Dataset/fe108/test/"
    save_path = "/wangx/DATA/Dataset/fe108/test/"
    video_files = os.listdir(data_path)
    dvs_img_interval = 1

    for videoID in range(len(video_files)):
        foldName = video_files[videoID]
        if foldName != "test.txt" and foldName != "list.txt":
            print("==>> foldName: ", foldName)
            
            # save_path_aps = os.path.join(save_path, foldName, 'aps/')
            # if not os.path.exists(save_path_aps):
            #     os.makedirs(save_path_aps)
                
            save_path_dvs = os.path.join(save_path, foldName, 'dvs/')
            if not os.path.exists(save_path_dvs):
                os.makedirs(save_path_dvs)
            
            aedat4_file = 'events.aedat4'
            read_path = os.path.join(data_path, foldName, aedat4_file)

            # read aeda4;
            frame_all = []
            frame_exposure_time = []
            frame_interval_time = []
            match_file = "/wangx/DATA/Dataset/fe108/pair.txt"
            pair = {}
            with open(match_file, 'r') as f:
                for line in f.readlines():
                    file, start_frame = line.split()
                    pair[file] = int(start_frame) + 1
            start_frame = pair[foldName]
            img_path = os.path.join(data_path, foldName, 'img')

            frame_end = len(os.listdir(img_path))
            with AedatFile(read_path) as f:
                # print(f.names)
                for frame in f['frames']:
                    frame_all.append(frame.image)
                    frame_interval_time.append([frame.timestamp_start_of_frame,
                                                frame.timestamp_end_of_frame])  ## [1607928583387944, 1607928583410285]
            frame_timestamp = frame_interval_time
            frame_num = len(frame_timestamp)

            events = np.hstack([packet for packet in f['events'].numpy()])

            t_all = torch.tensor(events['timestamp']).unsqueeze(1).to(device)
            x_all = torch.tensor(events['x']).unsqueeze(1).to(device)
            y_all = torch.tensor(events['y']).unsqueeze(1).to(device)
            p_all = torch.tensor(events['polarity']).unsqueeze(1).to(device)

            begin_frame = start_frame-1
            end_frame = frame_end + start_frame-1
            for frame_no in range(begin_frame, end_frame):
                start_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][0])[0][0]
                end_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][1])[0][0]
                sub_event = events[start_idx:end_idx]
                
                # ## RGB image generate
                # this_frame = frame_all[frame_no]
                # height, width = f['events'].size
                # event_img = np.zeros((height, width))
                # cv2.imwrite(os.path.join(save_path_aps, 'frame{:04d}'.format(frame_no-begin_frame)+'.png'), this_frame)

                # event img generate
                height, width = f['events'].size
                event_frame = 255 * np.ones((height, width, 3), dtype=np.uint8)
                start_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][0])[0][0]
                end_idx = np.where(events['timestamp'] >= frame_timestamp[frame_no][1])[0][0]
                event = events[start_idx:end_idx]

                on_idx = np.where(event['polarity'] == 1)  ## (array([    3,     4,     5, ..., 10633, 10635, 10636]),)
                off_idx = np.where(event['polarity'] == 0)  ## (array([    0,     1,     2, ..., 10629, 10632, 10634]),)
                event_frame[event['y'][on_idx], event['x'][on_idx], :] = [30, 30, 220] * event['polarity'][on_idx][:, None]
                event_frame[event['y'][off_idx], event['x'][off_idx], :] = [200, 30, 30] * (event['polarity'][off_idx] + 1)[
                                                                                        :, None]
                cv2.imwrite(os.path.join(save_path_dvs, 'frame{:04d}'.format((frame_no-begin_frame) * dvs_img_interval) + '.png'),
                            event_frame)
