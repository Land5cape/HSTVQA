import argparse
import os
import time


KoNViD_1k_video_dir = 'E:/workplace/VQAdataset/KoNViD-1k'
KoNViD_1k_label_path = 'E:/workplace/VQAdataset/KoNViD-1k/KoNViD_1k_attributes.csv'
KoNViD_1k_split_path = './datas/KoNViD_1k_split.csv'
KoNViD_1k_MOS = [1.22, 4.64]

CVD2014_video_dir = 'E:/workplace/VQAdataset/CVD2014'
CVD2014_label_path = 'E:/workplace/VQAdataset/CVD2014/Realignment_MOS.csv'
CVD2014_split_path = './datas/CVD2014_split.csv'
CVD2014_MOS = [-6.50, 93.38]

LiveV_video_dir = 'E:/workplace/VQAdataset/Live-VQC'
LiveV_label_path = 'E:/workplace/VQAdataset/Live-VQC/LiveVQCData.csv'
LiveV_split_path = './datas/LiveV_split.csv'
LiveV_MOS = [6.2237, 94.2865]

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_LENGTH = 8
INPUT_SIZE = 256
CLIP_STRIDE = 8


ds = 4
if ds == 1:
    video_dir = KoNViD_1k_video_dir
    label_path = KoNViD_1k_label_path
    split_path = KoNViD_1k_split_path
    mos = KoNViD_1k_MOS
    tail = '.mp4'
elif ds == 2:
    video_dir = CVD2014_video_dir
    label_path = CVD2014_label_path
    split_path = CVD2014_split_path
    mos = CVD2014_MOS
    tail = '.avi'
else:
    video_dir = LiveV_video_dir
    label_path = LiveV_label_path
    split_path = LiveV_split_path
    mos = LiveV_MOS
    tail = '.mp4'

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', default='train', type=str, help='train, retrain or predict')

    parser.add_argument('--video_dir', default=video_dir, type=str, help='Path to input videos')
    parser.add_argument('--video_type', default=tail, type=str, help='Type of videos')

    parser.add_argument('--score_file_path', default=label_path, type=str,
                        help='Path to input subjective score')
    parser.add_argument('--info_file_path', default=split_path, type=str,
                        help='Path to input info')
    parser.add_argument('--MOS_min', default=mos[0], type=float, help='MOS min range')
    parser.add_argument('--MOS_max', default=mos[1], type=float, help='MOS max range')

    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='L2 regularization')
    parser.add_argument('--epoch_nums', default=20, type=int, help='epochs to train')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')

    start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    parser.add_argument('--start_time', default=start_time, type=str,
                        help='start time of this process')
    parser.add_argument('--save_model', default='./model-save/' + start_time, type=str,
                        help='path to save the model')
    parser.add_argument('--save_checkpoint', default=True, type=bool, help='')
    parser.add_argument('--load_model', default='', type=str,
                        help='path to load checkpoint')
    parser.add_argument('--writer_t_dir', default='./runs/' + start_time + '_train', type=str,
                        help='batch size to train')
    parser.add_argument('--writer_v_dir', default='./runs/' + start_time + '_val', type=str, help='batch size to train')

    args = parser.parse_args()

    if not os.path.isdir('./model-save/'):
        os.mkdir('./model-save/')
    if not os.path.isdir('./runs/'):
        os.mkdir('./runs/')

    return args


if __name__ == '__main__':
    args = parse_opts()
    print(args)
