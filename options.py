import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=210, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=70, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default=os.getcwd() + '/LFSOD_dataset/RGBD_for_train/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default=os.getcwd() + '/LFSOD_dataset/RGBD_for_train/depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default=os.getcwd() + '/LFSOD_dataset/RGBD_for_train/GT/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str, default=os.getcwd() + '/LFSOD_dataset/test_in_train/RGB/', help='the test rgb images root')
parser.add_argument('--test_depth_root', type=str, default=os.getcwd() + '/LFSOD_dataset/test_in_train/depth/', help='the test depth images root')
parser.add_argument('--test_gt_root', type=str, default=os.getcwd() + '/LFSOD_dataset/test_in_train/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default=os.getcwd() + '/TLFNet_LFSOD_cpts/', help='the path to save models and logs')

parser.add_argument('--fs_root', type=str, default=os.getcwd() + '/LFSOD_dataset/RGBD_for_train/FS_rgb/', help='the training FS images root')
parser.add_argument('--test_fs_root', type=str, default=os.getcwd() + '/LFSOD_dataset/test_in_train/FS_rgb/', help='the testing FS images root')

opt = parser.parse_args()
