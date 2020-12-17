import os
import time

dataset = 'cholec80'  # type=str, 'yfcc | fcv |
niter = 200  # type=int, default=50, help='number of epochs to train for'
niter_lstm = 10
use_cuda = True
num_workers = 0
use_checkpoint = False
lr = 0.002 #type=float, default=0.0001
lr_lstm = 0.0001

#feature_size = 2048
#anneal = 'logistic'

content_dim = 224
pose_dim = 32
nbits = pose_dim + content_dim
image_width = 64
sd_nf = 100

margin = int(nbits/4)

# sim_weight = 0.1
# rec_weight = 0.9
sim_weight = 1
rec_weight = 1
triplet_weight = 0.001
sd_weight = 0.0001

crop_type = 1
use_flip = 1
channels = 3
seed = 1
beta1 = 0.5
optimizer = 'adam'

def data_info(data_type):
    if data_type == 'cholec80':
        batch_size = 64  # type=int, default=16, help='input batch size'
        max_frames = 20
        test_batch_size = 64
        max_step = 4
        num_sample = 4786

        save_dir = '/media/f/ziyi/Video_Retrieval/UDCM_ALL/logs/'
        # save_dir = '/media/f/ziyi/Video_Retrieval/UDCM_SurgicalAction160/logs_V2_only_eval/'

        # --------------------------- change clip_length ------------------------------
        clip_length = max_frames
        downsample_rate = 25
        # extract_dir = "/media/f/ziyi/Video_Retrieval/feature_extract/cholec80_various_sample/"
        # save_extract_dir = extract_dir + 'cliplen_' + str(clip_length) + '_downsam_' + str(downsample_rate)
        # data_root_80 = save_extract_dir + '/frames/'
        # labels_name = save_extract_dir + '/cholec80_labels_test.mat'
        save_various_dir = save_dir + 'cliplen_' + str(clip_length) + '_downsam_' + str(downsample_rate) + '/'

        # ---------------------------  origin data ------------------------------
        data_root_80 = "/media/f/ziyi/Video_Retrieval/feature_extract/Cholec80_frame_extract/"
        #data_root_80 = "/media/f/ziyi/Video_Retrieval/feature_extract/cholec80_various_sample/cliplen_60_downsam_25/frames/"
        labels_name = "/media/f/ziyi/Video_Retrieval/SSVH_master_surgical/data/cholec80_labels_test.mat"
        #labels_name = "/media/f/ziyi/Video_Retrieval/feature_extract/cholec80_various_sample/cliplen_60_downsam_25/cholec80_labels_test.mat"
        data_root_test = "/media/f/ziyi/Video_Retrieval/NPH_surgical_GAN/dataset_test/Cholec80_test/video_use_2/"
        temp_labels_name = "/media/f/ziyi/Video_Retrieval/NPH_surgical_GAN_test_2/tempral_label/cholec80_temp_labels_relation_test.mat"

        return batch_size, max_frames, test_batch_size, max_step, num_sample, save_dir, clip_length, \
               downsample_rate, data_root_80, labels_name, data_root_test, temp_labels_name

    if data_type == 'inhouse':
        batch_size = 46  # type=int, default=16, help='input batch size'
        max_frames = 20
        test_batch_size = 46
        max_step = 4
        num_sample = 4156

        save_dir = '/media/f/ziyi/Video_Retrieval/UDCM_ALL/logs/'
        # save_dir = '/media/f/ziyi/Video_Retrieval/UDCM_SurgicalAction160/logs_V2_only_eval/'

        # --------------------------- change clip_length ------------------------------
        clip_length = max_frames
        downsample_rate = 25
        # extract_dir = "/media/f/ziyi/Video_Retrieval/feature_extract/cholec80_various_sample/"
        # save_extract_dir = extract_dir + 'cliplen_' + str(clip_length) + '_downsam_' + str(downsample_rate)
        # data_root_80 = save_extract_dir + '/frames/'
        # labels_name = save_extract_dir + '/cholec80_labels_test.mat'
        save_various_dir = save_dir + 'cliplen_' + str(clip_length) + '_downsam_' + str(downsample_rate) + '/'

        # ---------------------------  origin data ------------------------------
        data_root_80 = "/media/f/ziyi/Video_Retrieval/in_house_UDCM/in-house_data/human_frames_1fps/"
        labels_name = "/media/f/ziyi/Video_Retrieval/in_house_UDCM/in-house_data/labels_processed/label_test.mat"
        data_root_test = "/media/f/ziyi/Video_Retrieval/NPH_surgical_GAN/dataset_test/Cholec80_test/video_use_2/"
        temp_labels_name = "/media/f/ziyi/Video_Retrieval/NPH_surgical_GAN_test_2/tempral_label/cholec80_temp_labels_relation_test.mat"
        data_root_inhouse_test = "/media/f/ziyi/Video_Retrieval/in_house_UDCM/in-house_data/human_surgery_processed_1fps/"

        return batch_size, max_frames, test_batch_size, max_step, num_sample, save_dir, clip_length, \
               downsample_rate, data_root_80, data_root_inhouse_test, labels_name, temp_labels_name



