import csv
import os
import random
import numpy as np
import cv2
import skvideo.io
import torch
import torch.nn.init as init
import torch.nn as nn
from scipy import stats
import matplotlib.pyplot as plt

from opts import INPUT_LENGTH, INPUT_SIZE, CLIP_STRIDE


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_pred_label(csv_path, list1, list2):
    infos = zip(list1, list2)
    with open(csv_path, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for info in infos:
            writer.writerow(info)

def weigth_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def MOS_label(MOS, MOS_range):
    MOS_min, MOS_max = MOS_range
    label = (MOS - MOS_min) / (MOS_max - MOS_min)
    return label


def label_MOS(label, MOS_range):
    MOS_min, MOS_max = MOS_range
    MOS = label * (MOS_max - MOS_min) + MOS_min
    return MOS


def get_PLCC(y_pred, y_val):
    return stats.pearsonr(y_pred, y_val)[0]


def get_SROCC(y_pred, y_val):
    return stats.spearmanr(y_pred, y_val)[0]


def get_KROCC(y_pred, y_val):
    return stats.stats.kendalltau(y_pred, y_val)[0]


def get_RMSE(y_pred, y_val, MOS_range):
    y_p = label_MOS(y_pred, MOS_range)
    y_v = label_MOS(y_val, MOS_range)
    return np.sqrt(np.mean((y_p - y_v) ** 2))


def get_MSE(y_pred, y_val, MOS_range):
    y_p = label_MOS(y_pred, MOS_range)
    y_v = label_MOS(y_val, MOS_range)
    return np.mean((y_p - y_v) ** 2)


def mos_scatter(pred, mos, show_fig=False):
    fig = plt.figure()
    plt.scatter(mos, pred, s=5, c='g', alpha=0.5)
    plt.xlabel('MOS')
    plt.ylabel('PRED')
    plt.plot([0, 1], [0, 1], linewidth=0.5)
    if show_fig:
        plt.show()
    return fig


def read_video_npy(video_path, start=None, intput_length=None):
    video_frames = np.load(video_path, mmap_mode='r')
    if start and intput_length:
        video_frames = video_frames[start: start + intput_length]
    frames = torch.from_numpy(video_frames)
    frames = frames.permute([3, 0, 1, 2])
    frames = frames.float()
    frames = frames / 255
    return frames


def save_crop_video(video_path, video_save_dir, tail):
    if not os.path.exists(video_save_dir):
        os.mkdir(video_save_dir)
    video_name = os.path.basename(video_path)

    if tail == '.yuv':
        video_frames = skvideo.io.vread(video_path, 1080, 1920,
                                        inputdict={'-pix_fmt': 'yuvj420p'})
        length = video_frames.shape[0]
        H = video_frames.shape[1]
        W = video_frames.shape[2]

    else:
        cap = cv2.VideoCapture()
        cap.open(video_path)
        if not cap.isOpened():
            raise Exception("VideoCapture failed!")

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_frames = np.zeros((length, H, W, 3), dtype='uint8')

        for i in range(length):
            rval, frame = cap.read()
            if rval:
                video_frames[i] = frame
            else:
                raise Exception("VideoCapture failed!")
        cap.release()

    n_clips = count_clip(length, CLIP_STRIDE)

    print(video_frames.size, length, H, W, n_clips)

    for clip in range(n_clips):
        pos = int(clip * CLIP_STRIDE)
        hIdxMax = H - INPUT_SIZE
        wIdxMax = W - INPUT_SIZE

        hIdx = [INPUT_SIZE * i for i in range(0, hIdxMax // INPUT_SIZE + 1)]
        wIdx = [INPUT_SIZE * i for i in range(0, wIdxMax // INPUT_SIZE + 1)]
        if hIdxMax % INPUT_SIZE != 0:
            hIdx.append(hIdxMax)
        if wIdxMax % INPUT_SIZE != 0:
            wIdx.append(wIdxMax)
        for h in hIdx:
            for w in wIdx:
                video_save_path = os.path.join(video_save_dir, video_name + '_{}_{}_{}.npy'.format(pos, h, w))
                np.save(video_save_path,
                        video_frames[pos:pos + INPUT_LENGTH,
                        h:h + INPUT_SIZE, w:w + INPUT_SIZE, :])


def get_video_name(file_name, tail):
    video_name = os.path.basename(file_name)
    if not video_name.endswith('.npy'):
        return video_name
    else:
        return video_name.split(tail)[0] + tail


def read_label(csv_path, id1, id2, tail):
    dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        for row in reader:
            name = row[id1]
            if not name.endswith(tail):
                name = name + tail
            dict[name] = float(row[id2])
    return dict


def read_split(csv_path):
    info_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        for row in reader:  # name, class
            info_dict[row[0]] = row[1]
    return info_dict


def write_split(csv_path, infos, wtype):
    with open(csv_path, wtype, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if wtype == 'w':
            writer.writerow(['file_name', 'class'])
        for info in infos:
            writer.writerow(info)


def count_clip(n_frames, stride):
    tIdMax = n_frames - INPUT_LENGTH
    n_clips = tIdMax // stride + 1

    return int(n_clips)


def data_split(full_list, ratio, shuffle=True):
    nums_total = len(full_list)
    offset1 = int(nums_total * ratio[0])
    offset2 = int(nums_total * (ratio[0] + ratio[1]))

    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset1]
    sublist_2 = full_list[offset1:offset2]
    sublist_3 = full_list[offset2:]
    return sublist_1, sublist_2, sublist_3
