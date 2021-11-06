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
import imageio
import numpy as np
import time
from PIL import Image, ImageOps
from data import get_train_loader
from args import num_workers, pose_dim, content_dim, channels, seed
from args import image_width, optimizer, niter_lstm
from args import data_info, lr_lstm
import scipy.io as sio
from data import get_eval_loader
import tools
from models.lstm_ae import AutoEncoderRNN
import matplotlib.pyplot as plt
import pylab as pl


parser = argparse.ArgumentParser()
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('-width', '--image_width', default=image_width, type=int, help='image_width')

parser.add_argument('-cd', '--content_dim', default=content_dim, type=int, help='content_dim')
parser.add_argument('-pd', '--pose_dim', default=pose_dim, type=int, help='pose_dim')

parser.add_argument('--niter_lstm', default=niter_lstm, type=int, help='niter_lstm')
parser.add_argument('--lr_lstm', default=lr_lstm, type=float, help='lr_lstm')

parser.add_argument('-data', '--data_type', default=None, type=str, help='data_type')
parser.add_argument('-path', '--file_path', default=None, type=str, help='file_path')
parser.add_argument('-opt', '--optimizer', default='adam', type=str, help='file_path')
parser.add_argument('-mf', '--max_frames', default='Cholec80', type=int, help='max_frames')

opt = parser.parse_args()
print(opt)

content_dim = opt.content_dim
pose_dim = opt.pose_dim
nbits = content_dim + pose_dim
niter_lstm = opt.niter_lstm
data_type = opt.data_type
file_path = opt.file_path
rnn_size = opt.rnn_size
rnn_layers = opt.rnn_layers
image_width = opt.image_width
optimizer = opt.optimizer
max_frames = opt.max_frames

if data_type == 'cholec80':
    batch_size, max_frames, test_batch_size, max_step, num_sample, save_dir, num_train, \
    clip_length, downsample_rate, data_root_80, labels_name = data_info(data_type)

print('file_path', file_path)

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
dtype = torch.cuda.FloatTensor

gpu_usg = True
num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")
print('use_gpu', use_gpu, 'num_gpu', num_gpu)

# ---------------- load the models  ----------------
checkpoint = torch.load('%s/model.pth' % file_path)
netD = checkpoint['netD']
netEP = checkpoint['netEP']
netEC = checkpoint['netEC']
netD.eval()
netEP.eval()
netEC.eval()

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()

# --------- transfer to gpu ------------------------------------
netEP.cuda()
netEC.cuda()
netD.cuda()
mse_criterion.cuda()

# --------- load a dataset ------------------------------------
def get_seq_path_train(data_root):
    video_idx = os.listdir(data_root)
    video_idx.sort(key=lambda x: int(x.split('.')[0]))
    train_idx_1 = 0
    stop_idx_1 = int(len(video_idx) * 0.5)
    print('loading lstm train data ')

    array_train_all = np.zeros((num_train, max_frames, image_width, image_width, 3), dtype=np.float32)
    array_id = 0

    for i in range(train_idx_1, stop_idx_1):
        print('----------------', i, end='\r')
        video_path = os.path.join(data_root, video_idx[i])
        frame = os.listdir(video_path)
        frame.sort(key=lambda x: int(x.split('.')[0]))
        for ii in frame:
            ii_num = int(ii.split('.')[0])
            if ii_num % max_frames == 0:
                img_array_train = np.zeros((max_frames, image_width, image_width, 3), dtype=np.float32)
                for j in range(ii_num, ii_num + max_frames):
                    img_names = os.path.join(video_path, str(j) + '.jpg')
                    im = Image.open(img_names)
                    im_re = im.resize((image_width, image_width), Image.ANTIALIAS)
                    imm = np.array(im_re).astype(np.float32) / 255
                    img_array_train[j-ii_num] = imm
                array_train_all[array_id] = img_array_train
                array_id += 1
    print('\n')
    return array_train_all


if data_type == 'cholec80':
    seq_train = get_seq_path_train(data_root_80)

train_lstm_loader = get_train_loader(seq_train, batch_size)


# --------- load a dataset ------------------------------------
def get_seq_path_test(data_root):
    video_idx = os.listdir(data_root)
    video_idx.sort(key=lambda x: int(x.split('.')[0]))
    train_idx_2 = int(len(video_idx) * 0.5)
    stop_idx_2 = len(video_idx)

    print('loading test data lstm')

    array_test_all = np.zeros((num_sample, max_frames, image_width, image_width, 3), dtype=np.float32)
    array_id = 0

    for i in range(train_idx_2, stop_idx_2):
        print('----------------', i, end='\r')
        video_path = os.path.join(data_root, video_idx[i])
        frame = os.listdir(video_path)
        frame.sort(key=lambda x: int(x.split('.')[0]))
        for ii in frame:
            ii_num = int(ii.split('.')[0])
            if ii_num % max_frames == 0:
                img_array_test = np.zeros((max_frames, image_width, image_width, 3), dtype=np.float32)
                for j in range(ii_num, ii_num + max_frames):
                    img_names = os.path.join(video_path, str(j) + '.jpg')
                    im = Image.open(img_names)
                    im_re = im.resize((image_width, image_width), Image.ANTIALIAS)
                    imm = np.array(im_re).astype(np.float32) / 255
                    img_array_test[j - ii_num] = imm
                array_test_all[array_id] = img_array_test
                array_id += 1
    print('\n')
    return array_test_all


if data_type == 'cholec80':
    seq_test = get_seq_path_test(data_root_80)

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
    y = hard_sigmoid(x)
    out = 2.*Round3.apply(y)-1.
    return out


class Array():
    def __init__(self):
        pass

    def setmatrcs(self,matrics):
        self.matrics = matrics

    def concate_v(self,matrics):
        self.matrics = np.vstack((self.matrics,matrics))

    def getmatrics(self):
        return self.matrics


# --------- training -----------
def train_lstm(x):
    lstm_ae.zero_grad()
    criterion = nn.MSELoss(size_average=False, reduce=False)

    h_p = [netEP(x[i]).detach().view(batch_size, pose_dim) for i in range(max_frames)]
    h_p_a = torch.stack(h_p, 0)    # (seq_length, batch_size, hidden_size)
    h_p_rec, h_p_binary = lstm_ae(h_p_a)
    loss = criterion(h_p_rec, h_p_a)
    loss_sum = torch.sum(loss, (1, 2)) / pose_dim

    weight = torch.arange(2, 1, -0.05).cuda()
    loss = torch.sum(loss_sum*weight)

    loss.backward()
    optimizer_ae.step()
    return loss


# --------- testing funtions ------------------------------------
def evaluate(labels_name):
    hashcode = np.zeros((num_sample, nbits), dtype=np.float32)
    label_array = Array()
    batch_size = test_batch_size
    labels = sio.loadmat(labels_name)['labels']
    video_num = 0

    with torch.no_grad():
        for i, (videos, index) in enumerate(test_loader):
            # torch.cuda.empty_cache()
            batchsize = len(videos)
            videos = Variable(videos)
            videos.transpose_(0, 1)
            videos.transpose_(3, 4).transpose_(2, 3)
            videos = videos.to(device)

            if batchsize < batch_size:
                break

            h_c_all = [netEC(videos[k]).detach().view(batch_size, content_dim) for k in range(max_frames)]
            h_c = torch.mean(torch.stack(h_c_all), dim=0)
            h_c_binary = torch.sign(h_c)
            h_p = [netEP(videos[i]).detach().view(batch_size, pose_dim) for i in range(max_frames)]

            h_p_a = torch.stack(h_p, 0)  # (seq_length, batch_size, hidden_size)
            h_p_rec, h_p_binary = lstm_ae(h_p_a)

            feats_all = torch.cat([h_c_binary, h_p_binary], dim=1)
            feats_all.view(batchsize, nbits)
            BinaryCode = feats_all

            hashcode[i * batch_size:(i + 1) * batch_size, :] = BinaryCode.data.cpu().numpy()
            video_num += batchsize

    hashcode = hashcode[:video_num, :]
    labels = labels[:video_num, :]
    label_array.setmatrcs(labels)

    test_hashcode = np.matrix(hashcode)

    time3 = time.time()
    Hamming_distance = 0.5 * (-np.dot(test_hashcode, test_hashcode.transpose()) + nbits)
    HammingRank = np.argsort(Hamming_distance, axis=0)
    time4 = time.time()
    print('HammingRank computation costs: ', time4 - time3)

    np.savetxt(file_path + '/Hamming_distance_lstm' + '.txt', Hamming_distance)
    np.savetxt(file_path + '/HammingRank_lstm' + '.txt', HammingRank)

    labels = label_array.getmatrics()
    sim_labels = np.dot(labels, labels.transpose())

    map_list = [1, 2, 5, 10, 15, 20, 50]
    for i in map_list:
        map, _, _ = tools.mAP(sim_labels, HammingRank, i)
        records.write('\ttopK: ' + str(i) + '\tmap: ' + str(map) + '\n')
        print('i: ', i, ' map: ', map, '\n')


# --------- training loop -----------------
lstm_ae = AutoEncoderRNN(pose_dim, rnn_size, rnn_layers, batch_size)
lstm_ae.cuda()

if optimizer == 'adam':
    optimizer_ae = torch.optim.Adam(lstm_ae.parameters(), lr=lr_lstm, betas=(opt.beta1, 0.999))
elif optimizer == 'sgd':
    optimizer_ae = torch.optim.SGD(lstm_ae.parameters(), lr=lr_lstm, momentum=0.9)

num_epoch = 0

for epoch in range(niter_lstm):
    lstm_ae.train()
    loss_new = 0.0
    epoch_loss_new = 0.0

    num_epoch = num_epoch + 1
    train_video_num = 0

    for i, (videos, index) in enumerate(train_lstm_loader, start=1):
        torch.cuda.empty_cache()
        batchsize = len(videos)
        time1 = time.time()
        videos = Variable(videos)
        videos.transpose_(0, 1)
        videos.transpose_(3, 4).transpose_(2, 3)
        videos = videos.to(device)
        time2 = time.time()
        if batchsize < batch_size:
            break
        train_video_num = train_video_num + batchsize

        # train lstm
        loss_new = train_lstm(videos)
        epoch_loss_new += loss_new
        lstm_loss = epoch_loss_new / train_video_num

        if i % 10 == 0:
            print('[%02d][%02d] mse loss: %.6f' % (epoch, i, lstm_loss))

    # save the model
    torch.save({
        'lstm': lstm_ae,
        'opt': opt},
        '%s/model_lstm.pth' % file_path)

    # -------------------------- test part -------------------------------
    if num_epoch == niter_lstm:
        print('num_epoch', num_epoch)

        checkpoint_1 = torch.load('%s/model_lstm.pth' % file_path)
        lstm_ae = checkpoint_1['lstm']
        lstm_ae.eval()

        records = open(file_path + '/record_lstm.txt', 'a+')

        time1 = time.time()
        evaluate(labels_name)
        time2 = time.time()
        print('evaluation computation costs: ', time2 - time1)

        records.write('\n')
        records.close()
# ------------------------- test part end -----------------------------
