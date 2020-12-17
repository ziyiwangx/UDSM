# -*- coding: gbk -*-
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader
import utils
import itertools
import imageio
import numpy as np
import time
from PIL import Image, ImageOps
from data import get_train_loader
from args import num_workers, pose_dim, content_dim, channels, seed, lr
from args import image_width, optimizer, niter_lstm
from args import data_info, lr_lstm
import scipy.io as sio
from data import get_eval_loader
import tools

parser = argparse.ArgumentParser()
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('--content_model', default='dcgan', help='model type (dcgan | dcgan_unet | vgg_unet)')
parser.add_argument('--pose_model', default='dcgan', help='model type (dcgan | unet | resnet)')

parser.add_argument('-cd', '--content_dim', default=content_dim, type=int, help='content_dim')
parser.add_argument('-pd', '--pose_dim', default=pose_dim, type=int, help='pose_dim')

parser.add_argument('--niter_lstm', default=niter_lstm, type=int, help='niter_lstm')
parser.add_argument('--lr_lstm', default=lr_lstm, type=float, help='lr_lstm')

parser.add_argument('-temp', '--temp_pkl', default=True, type=bool, help='temp_pkl')
parser.add_argument('-data', '--data_type', default=None, type=str, help='data_type')
parser.add_argument('-path', '--file_path', default=None, type=str, help='file_path')

opt = parser.parse_args()
print(opt)

content_dim = opt.content_dim
pose_dim = opt.pose_dim
nbits = content_dim + pose_dim
niter_lstm = opt.niter_lstm
data_type = opt.data_type
temp_pkl = opt.temp_pkl
file_path = opt.file_path

if data_type is None:
    print('please input the name of datasets: cholec80 or inhouse')

if data_type == 'cholec80':
    batch_size, max_frames, test_batch_size, max_step, num_sample, save_dir, \
    clip_length, downsample_rate, data_root_80, labels_name, data_root_test, temp_labels_name = data_info(data_type)
elif data_type == 'inhouse':
    batch_size, max_frames, test_batch_size, max_step, num_sample, save_dir, \
    clip_length, downsample_rate, data_root_80, data_root_inhouse_test, labels_name, temp_labels_name = data_info(data_type)

print('file_path', file_path)
if not os.path.exists(file_path):
    os.makedirs(file_path)

print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
dtype = torch.cuda.FloatTensor

gpu_usg = True
num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")
print('use_gpu', use_gpu, 'num_gpu',num_gpu)

# ---------------- load the models  ----------------
checkpoint = torch.load('%s/model.pth' % file_path)
netD = checkpoint['netD']
netEP = checkpoint['netEP']
netEC = checkpoint['netEC']
netD.eval()
netEP.eval()
netEC.eval()

# ---------------- optimizers ----------------
if optimizer == 'adam':
    optimizer = optim.Adam
elif optimizer == 'rmsprop':
    optimizer = optim.RMSprop
elif optimizer == 'sgd':
    optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % optimizer)

import models.lstm as models
netlstm = models.lstm(pose_dim+content_dim, pose_dim, opt.rnn_size, opt.rnn_layers, batch_size, opt.normalize)

netlstm.apply(utils.init_weights)

optimizer = optimizer(netlstm.parameters(), lr=lr_lstm, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()

# --------- transfer to gpu ------------------------------------
netlstm.cuda()
netEP.cuda()
netEC.cuda()
netD.cuda()
mse_criterion.cuda()

# --------- load a dataset ------------------------------------

def get_seq_path_train(data_root):
    frame_path_all_all = []
    frame_path_all_1 = []
    frame_path_all_2 = []
    img_list_train_all = []
    img_list_test_all = []

    video_idx = os.listdir(data_root)
    video_idx.sort(key=lambda x: int(x.split('.')[0]))
    train_idx_1 = 0
    stop_idx_1 = int(len(video_idx) * 0.5)
    train_idx_2 = int(len(video_idx) * 0.5)
    stop_idx_2 = len(video_idx)
    sequence = True
    print('loading train data lstm')

    for i in range(train_idx_1, stop_idx_1):
        print('----------------', i, end='\r')
        video_path = os.path.join(data_root, video_idx[i])
        frame = os.listdir(video_path)
        frame.sort(key=lambda x: int(x.split('.')[0]))
        for ii in frame:
            ii_num = int(ii.split('.')[0])

            if ii_num % max_frames == 0:
                seq_list = []
                img_list_train = []
                if sequence:
                    for j in range(ii_num, ii_num + max_frames):
                        img_names = os.path.join(video_path, str(j) + '.jpg')
                        im = Image.open(img_names)
                        # imr = im.convert('RGB')
                        imm = im.resize((image_width, image_width), Image.ANTIALIAS)
                        immm = np.array(imm).astype(np.float32) / 225
                        img_list_train.append(immm)
                        seq_list.append(img_names)
                    img_list_train_all.append(np.array(img_list_train))
                    frame_path_all_1.append(np.array(seq_list))
    print('\n')
    return np.array(img_list_train_all).astype(np.float32)


def get_seq_path_train_inhouse(data_root):
    downsample_step = downsample_rate / 25
    frame_path_all_all = []
    frame_path_all_1 = []
    frame_path_all_2 = []
    img_list_train_all = []
    img_list_test_all = []

    # video_idx = os.listdir(data_root)
    # video_idx.sort(key=lambda x: int(x.split('.')[0]))
    video_idx = sorted(os.listdir(data_root))
    train_idx_1 = 0
    stop_idx_1 = int(len(video_idx) * 0.5)
    #print('self.stop_idx', stop_idx_1)
    train_idx_2 = int(len(video_idx) * 0.5)
    stop_idx_2 = len(video_idx)
    sequence = True
    print('loading training data')

    for i in range(train_idx_1, stop_idx_1):
        print('----------------', i, end='\r')
        video_path = os.path.join(data_root, video_idx[i])
        frame = os.listdir(video_path)
        frame.sort(key=lambda x: int(x.split('.')[0]))
        #print("video", i, video_idx[i], "frame_num_before", len(frame))
        final_num_clip = len(frame) // int(max_frames*downsample_step)
        frame = frame[:int(final_num_clip * max_frames*downsample_step)]
        #print("video", i, video_idx[i], "frame_num_after", len(frame), "final_num_clip", final_num_clip)

        for ii in frame:
            ii_num_origin = int(ii.split('.')[0])
            ii_num = int((ii_num_origin+1) / 2)
            if (ii_num-1) % int(max_frames*downsample_step) == 0:
                seq_list = []
                img_list_train = []
                if sequence:
                    for j in range(ii_num, ii_num + int(max_frames*downsample_step), int(downsample_step)):
                        img_names = os.path.join(video_path, str(int(j*2-1)) + '.jpg')
                        im = Image.open(img_names)
                        # imr = im.convert('RGB')
                        imm = im.resize((image_width, image_width), Image.ANTIALIAS)
                        immm = np.array(imm).astype(np.float32) / 225
                        img_list_train.append(immm)
                        seq_list.append(img_names)
                    # print('sequence_list', len(seq_list), len(seq_list[0]))
                    img_list_train_all.append(np.array(img_list_train))
                    frame_path_all_1.append(np.array(seq_list))
                #print('img_list_train_all', len(img_list_train_all), len(img_list_train_all[0]), len(img_list_train_all[0][0][0]))
    print('\n')
    return np.array(img_list_train_all).astype(np.float32)


if data_type == 'cholec80':
    seq_train = get_seq_path_train(data_root_80)
elif data_type == 'inhouse':
    seq_train = get_seq_path_train_inhouse(data_root_80)

train_lstm_loader = get_train_loader(seq_train, batch_size)


# --------- load a dataset ------------------------------------
def get_seq_path_test(data_root):
    frame_path_all_1 = []
    img_list_train_all = []

    video_idx = os.listdir(data_root)
    video_idx.sort(key=lambda x: int(x.split('.')[0]))
    train_idx_1 = 0
    stop_idx_1 = int(len(video_idx) * 0.5)
    train_idx_2 = int(len(video_idx) * 0.5)
    stop_idx_2 = len(video_idx)
    sequence = True
    print('loading test data lstm')

    for i in range(train_idx_2, stop_idx_2):
        print('----------------', i, end='\r')
        video_path = os.path.join(data_root, video_idx[i])
        frame = os.listdir(video_path)
        frame.sort(key=lambda x: int(x.split('.')[0]))
        for ii in frame:
            ii_num = int(ii.split('.')[0])
            if ii_num % max_frames == 0:
                seq_list = []
                img_list_train = []
                for j in range(ii_num, ii_num + max_frames):
                    img_names = os.path.join(video_path, str(j) + '.jpg')
                    im = Image.open(img_names)
                    # imr = im.convert('RGB')
                    imm = im.resize((image_width, image_width), Image.ANTIALIAS)
                    immm = np.array(imm).astype(np.float32) / 225
                    img_list_train.append(immm)
                    seq_list.append(img_names)
                img_list_train_all.append(np.array(img_list_train))
                frame_path_all_1.append(np.array(seq_list))
    print('\n')
    return np.array(img_list_train_all).astype(np.float32)


def get_seq_path_test_inhouse(data_root):
    downsample_step = downsample_rate / 25
    frame_path_all_1 = []
    img_list_train_all = []

    # video_idx = os.listdir(data_root)
    # video_idx.sort(key=lambda x: int(x.split('.')[0]))
    video_idx = sorted(os.listdir(data_root))
    train_idx_1 = 0
    stop_idx_1 = int(len(video_idx) * 0.5)
    train_idx_2 = int(len(video_idx) * 0.5)
    stop_idx_2 = len(video_idx)
    sequence = True
    sum_clips = 0
    ten_sample = [28, 30, 31, 34, 35, 36, 38, 39, 40, 41]
    ten_sample = [28, 30, 31, 34, 40, 41]
    print('loading testing data')

    # for i in ten_sample:
    for i in range(train_idx_1, stop_idx_2):
        #print(1)
        print('----------------', i, end='\r')
        video_path = os.path.join(data_root, video_idx[i])
        #print(2)
        frame = os.listdir(video_path)
        frame.sort(key=lambda x: int(x.split('.')[0]))
        #print(3)
        #print("video", i, video_idx[i], "frame_num_before", len(frame))
        final_num_clip = len(frame) // int(max_frames * downsample_step)
        sum_clips = sum_clips + final_num_clip
        frame = frame[:int(final_num_clip * max_frames * downsample_step)]
        #print("video", i, video_idx[i], "frame_num_after", len(frame),
              #"final_num_clip", final_num_clip, "sum_clips", sum_clips)

        for ii in frame:
            ii_num = int(ii.split('.')[0])
            if (ii_num-1) % int(max_frames*downsample_step) == 0:
                seq_list = []
                img_list_train = []
                #print('****************', ii)
                for j in range(ii_num, ii_num + int(max_frames*downsample_step), int(downsample_step)):
                    img_names = os.path.join(video_path, str(j) + '.jpg')
                    im = Image.open(img_names)
                    # imr = im.convert('RGB')
                    imm = im.resize((image_width, image_width), Image.ANTIALIAS)
                    immm = np.array(imm).astype(np.float32) / 225
                    img_list_train.append(immm)
                    seq_list.append(img_names)
                    # print('sequence_list', len(seq_list), len(seq_list[0]))
                img_list_train_all.append(np.array(img_list_train))
                frame_path_all_1.append(np.array(seq_list))
                #print(len(frame_path_all_1), '++++++++++++++++')
                #print(len(img_list_train_all), '++++++++++++++++')
                #print('img_list_train_all', len(img_list_train_all), len(img_list_train_all[0]), len(img_list_train_all[0][0][0]))
    print('\n')
    return np.array(img_list_train_all).astype(np.float32)


if data_type == 'cholec80':
    seq_test = get_seq_path_test(data_root_80)
elif data_type == 'inhouse':
    seq_test = get_seq_path_test_inhouse(data_root_inhouse_test)

#seq_test = get_seq_path_test(data_root_80)
test_loader = get_eval_loader(seq_test, test_batch_size)


# --------- training funtions ------------------------------------
class Round3(Function):
    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        output = torch.round(input)
        ctx.input = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask = 1-(ctx.input==0)
        mask = Variable(mask).cuda().float()
        grad_output = grad_output*mask
        return grad_output, None, None

def hard_sigmoid(x):
    y = (x+1.)/2.
    y[y>1] = 1
    y[y<0] = 0

    return y

def binary_tanh_unit(x):
    # round3 = Elemwise(round3_scalar)
    y = hard_sigmoid(x)
    out = 2.*Round3.apply(y)-1.
    return out

def train(x):
    netlstm.zero_grad()

    # initialize the hidden state.
    netlstm.hidden = netlstm.init_hidden()

    # get fixed content vector from last ground truth frame
    #h_c = netEC(x[int(max_frames/2)])  # choose arbitrarily or mean pooling？
    h_c_all = [netEC(x[k])[0].detach() for k in range(max_frames)]
    h_c = torch.mean(torch.stack(h_c_all), dim=0)

    '''if type(h_c) is tuple:
        h_c = h_c[0].detach()
    else:
        h_c = h_c.detach()'''

    # get sequence of pose vectors
    h_p = [netEP(x[i]).detach() for i in range(max_frames)]
    #print('h_c', len(h_c), len(h_c[0]), len(h_c[0][0]))
    #print('h_p', len(h_p), len(h_p[0]), len(h_p[0][0]))
    mse = 0
    for i in range(1, max_frames):
        fesats = torch.cat([h_p[i - 1], h_c], dim=1)
        #print('fesats', len(fesats), len(fesats[0]), len(fesats[0][0]))
        pose_pred = netlstm(fesats)  # 4,128,12,12
        #print(pose_pred)
        #print('pose_pred', pose_pred.size())
        mse += mse_criterion(pose_pred, h_p[i])

    mse.backward(retain_graph=True)
    optimizer.step()

    return mse.data.cpu().numpy()/(max_frames)


# --------- testing funtions ------------------------------------
class Array():
    def __init__(self):
        pass
    def setmatrcs(self,matrics):
        self.matrics = matrics

    def concate_v(self,matrics):
        self.matrics = np.vstack((self.matrics,matrics))

    def getmatrics(self):
        return self.matrics


def evaluate(labels_name):
    real_sample = num_sample - num_sample % test_batch_size
    hashcode = np.zeros((real_sample, nbits), dtype=np.float32)
    label_array = Array()
    batch_size = test_batch_size
    rem = num_sample % batch_size
    labels = sio.loadmat(labels_name)['labels']
    labels = labels[:real_sample, :]
    label_array.setmatrcs(labels)
    batch_num = len(test_loader)
    # print('batch_num', batch_num)

    with torch.no_grad():
        for i, (videos, index) in enumerate(test_loader):
            torch.cuda.empty_cache()
            batchsize = len(videos)
            videos = Variable(videos)
            # videos = videos.view(4, batchsize, 3, opt.image_width, opt.image_width)
            videos.transpose_(0, 1)
            videos.transpose_(3, 4).transpose_(2, 3)
            videos = videos.to(device)

            if batchsize < batch_size:
                break

            # h_c = netEC(videos[int(max_frames/2)])
            h_c_all = [netEC(videos[k])[0].detach() for k in range(max_frames)]
            h_c = torch.mean(torch.stack(h_c_all), dim=0)

            '''if type(h_c) is tuple:
                h_c = h_c[0].detach()
            else:
                h_c = h_c.detach()'''

            netlstm.hidden = netlstm.init_hidden()

            gen_seq = []
            gen_seq.append(netEP(videos[0]))
            for j in range(1, max_frames):
                h_p = netEP(videos[j - 1]).detach()
                h_p_re = h_p.view(batchsize, pose_dim, 1, 1)
                h_pred = netlstm(torch.cat([h_p_re, h_c], dim=1))
                h_pred_re = h_pred.view(batchsize, pose_dim, 1, 1)
                gen_seq.append(h_pred_re)

            h_p = torch.mean(torch.stack(gen_seq), dim=0)
            # h_p = gen_seq[int(max_frames/2)]

            feats_all = torch.cat([h_c, h_p], dim=1)
            feats_all = feats_all.view(batchsize, nbits)
            feats_co = h_c.view(batchsize, content_dim)
            feats_mo = h_p.view(batchsize, pose_dim)
            BinaryCode = torch.sign(feats_all)

            if i == batch_num - 1 and rem != 0:
                # print('i', i)
                hashcode[i * batch_size:, :] = BinaryCode[:rem, :].data.cpu().numpy()
            else:
                hashcode[i * batch_size:(i + 1) * batch_size, :] = BinaryCode.data.cpu().numpy()
            # print('i', i)

    hashcode_array = Array()
    # labels = sio.loadmat(labels_name)['labels']
    # label_array.setmatrcs(labels)
    test_hashcode = np.matrix(hashcode)

    Hamming_distance = 0.5 * (-np.dot(test_hashcode, test_hashcode.transpose()) + nbits)
    HammingRank = np.argsort(Hamming_distance, axis=0)

    labels = label_array.getmatrics()
    sim_labels = np.dot(labels, labels.transpose())

    # records = open(file_path + '/64_0.1new.txt','w+')
    maps = []
    map_list = [1, 2, 5, 10, 20]
    for i in map_list:
        map, _, _ = tools.mAP(sim_labels, HammingRank, i)
        maps.append(map)
        records.write('\ttopK: ' + str(i) + ', map: ' + str(map) + '\n')
        print('i:', i, ' map: ', map)


def evaluate2(labels_name, uidx):
    real_sample = num_sample - num_sample % test_batch_size
    hashcode = np.zeros((real_sample, nbits), dtype=np.float32)
    label_array = Array()
    batch_size = test_batch_size
    rem = num_sample % batch_size
    labels = sio.loadmat(labels_name)['labels']
    labels = labels[:real_sample, :]
    label_array.setmatrcs(labels)
    batch_num = len(test_loader)
    #print('batch_num', batch_num)

    with torch.no_grad():
        for i, (videos, index) in enumerate(test_loader):
            torch.cuda.empty_cache()
            batchsize = len(videos)
            videos = Variable(videos)
            # videos = videos.view(4, batchsize, 3, opt.image_width, opt.image_width)
            videos.transpose_(0, 1)
            videos.transpose_(3, 4).transpose_(2, 3)
            videos = videos.to(device)

            if batchsize < batch_size:
                break

            # h_c = netEC(videos[int(max_frames/2)])
            h_c_all = [netEC(videos[k])[0].detach() for k in range(max_frames)]
            h_c = torch.mean(torch.stack(h_c_all), dim=0)

            '''if type(h_c) is tuple:
                h_c = h_c[0].detach()
            else:
                h_c = h_c.detach()'''

            netlstm.hidden = netlstm.init_hidden()

            gen_seq = []
            gen_seq.append(netEP(videos[0]))
            for j in range(1, max_frames):
                h_p = netEP(videos[j - 1]).detach()
                h_p_re = h_p.view(batchsize, pose_dim, 1, 1)
                h_pred = netlstm(torch.cat([h_p_re, h_c], dim=1))
                h_pred_re = h_pred.view(batchsize, pose_dim, 1, 1)
                gen_seq.append(h_pred_re)

            h_p = torch.mean(torch.stack(gen_seq), dim=0)
            # h_p = gen_seq[int(max_frames/2)]

            feats_all = torch.cat([h_c, h_p], dim=1)
            feats_all = feats_all.view(batchsize, nbits)
            feats_co = h_c.view(batchsize, content_dim)
            feats_mo = h_p.view(batchsize, pose_dim)
            BinaryCode = torch.sign(feats_all)

            if i == batch_num - 1 and rem != 0:
                # print('i', i)
                hashcode[i * batch_size:, :] = BinaryCode[:rem, :].data.cpu().numpy()
            else:
                hashcode[i * batch_size:(i + 1) * batch_size, :] = BinaryCode.data.cpu().numpy()
            # print('i', i)

    hashcode_array = Array()
    # labels = sio.loadmat(labels_name)['labels']
    # label_array.setmatrcs(labels)
    test_hashcode = np.matrix(hashcode)

    Hamming_distance = 0.5 * (-np.dot(test_hashcode, test_hashcode.transpose()) + nbits)
    np.savetxt(file_path + '/Hamming_distance_lstm_2layers_' + str(uidx) + '.txt', Hamming_distance)

    # ------------------ temporal priori knowledge  -----------------------
    if temp_pkl is True and uidx == 10:
        temp_labels = sio.loadmat(temp_labels_name)['labels']
        for i in range(real_sample):
            for j in range(real_sample):
                if temp_labels[i, j] == 0:
                    Hamming_distance[j, i] = nbits - 1
        np.savetxt(file_path + '/Hamming_distance_lstm_temp_2layers.txt', Hamming_distance)

    # ------------------ temporal priori knowledge end -----------------------

    HammingRank = np.argsort(Hamming_distance, axis=0)
    np.savetxt(file_path + '/HammingRank_lstm__2layers' + str(uidx) + '.txt', HammingRank)

    labels = label_array.getmatrics()
    #print('labels shape: ', labels.shape)
    sim_labels = np.dot(labels, labels.transpose())

    # records = open(file_path + '/64_0.1new.txt','w+')
    maps = []
    map_list = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 40, 50, 60, 80, 100]
    for i in map_list:
        map, _, _ = tools.mAP(sim_labels, HammingRank, i)
        maps.append(map)
        records.write('epoch: ' + str(uidx) + '\ttopK: ' + str(i) + '\tmap: ' + str(map) + '\n')
        #print('i: ', i, ' map: ', map, '\n')
    # records.close()
    map_all.append(maps)

    mAP = tools.calc_map(test_hashcode, test_hashcode, labels, labels)
    records_map.write('epoch: ' + str(uidx) + '\tmAP: ' + str(mAP) + '\n')
    mAPs.append(mAP)

    if uidx == 9:
        print('\n')
        map_avg = np.mean(np.array(map_all), axis=0)
        mAP_avg = np.mean(mAPs)

        for i in range(len(map_list)):
            records.write('average: ' + '\ttopK: ' + str(map_list[i]) + '\tmap_avg: ' + str(map_avg[i]) + '\n')
            print('i: ', map_list[i], ' map_avg: ', map_avg[i], '\n')

        print('mAP_avg: ', mAP_avg, '\n')
        records_map.write('average: ' + '\tmAP: ' + str(mAP) + '\n')




# --------- training loop ------------------------------------
num_epoch = 0

for epoch in range(niter_lstm):

    num_epoch = num_epoch + 1

    netlstm.train()
    epoch_loss = 0
    train_video_num = 0

    '''for i, (videos, index) in enumerate(train_lstm_loader, start=1):
        torch.cuda.empty_cache()
        batchsize = len(videos)
        time1 = time.time()
        videos = Variable(videos)
        #videos = videos.view(4, batchsize, 3, opt.image_width, opt.image_width)
        videos.transpose_(0, 1)
        videos.transpose_(3, 4).transpose_(2, 3)
        videos = videos.to(device)
        time2 = time.time()
        if batchsize < batch_size:
            break
        train_video_num = train_video_num + batchsize

        # train lstm
        loss = train(videos)
        epoch_loss += loss

        print('[%02d][%02d] mse loss: %.6f' % (epoch, i, epoch_loss / train_video_num))

    # save the model
    torch.save({
        'lstm': netlstm,
        'opt': opt},
        '%s/model_lstm.pth' % file_path)'''

    # -------------------------- test part -------------------------------
    #if num_epoch == niter_lstm:
    if (num_epoch % 4) == 0 or num_epoch == 1 or num_epoch == niter_lstm:
        epoch_sim_loss, epoch_rec_loss, epoch_sd_loss, epoch_sd_acc = 0, 0, 0, 0

        checkpoint_1 = torch.load('%s/model_lstm.pth' % file_path)
        netlstm = checkpoint_1['lstm']

        netlstm.eval()

        records = open(file_path + '/record_lstm.txt', 'a+')
        records_map = open(file_path + '/record_lstm_map.txt', 'a+')

        map_all = []
        mAPs = []

        if num_epoch == niter_lstm:
            for i in range(11):
                print('eval_index:', i, end='\r')
                evaluate2(labels_name, i)

        else:
            time1 = time.time()
            evaluate(labels_name)
            time2 = time.time()
            print('evaluation computation costs: ', time2 - time1)

        records.close()
    # ------------------------- test part end ------------------------------
