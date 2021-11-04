import os
import time

dataset = 'cholec80'  # type=str
niter = 200  # type=int, default=50, help='number of epochs to train for'
niter_lstm = 30
use_cuda = True
num_workers = 0
use_checkpoint = False
crop_type = 1
use_flip = 1
channels = 3
seed = 1
beta1 = 0.5
optimizer = 'adam'
lr = 0.0002
lr_lstm = 0.0002

content_dim = 224
pose_dim = 32
nbits = pose_dim + content_dim
margin = int(content_dim/4)

image_width = 64
sd_nf = 100

rec_weight = 1
triplet_weight = 0.01
sd_weight = 0.01


def data_info(data_type):
    if data_type == 'cholec80':
        batch_size = 64
        test_batch_size = 64
        max_step = 4

        max_frames = 20
        num_train = 4185
        num_sample = 4786

        # --------------------------- change clip_length ------------------------------
        clip_length = max_frames
        downsample_rate = 25

        # ---------------------------  origin data ------------------------------
        save_dir = './logs/'
        data_root_80 = "./data/Cholec80_frame_extract/"
        labels_name = "./data/cholec80_labels_test.mat"

        return batch_size, max_frames, test_batch_size, max_step, num_sample, save_dir, num_train, \
               clip_length, downsample_rate, data_root_80, labels_name


