import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
import numpy as np
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader
import utils
import scipy.io as sio
import tools
import progressbar
from data import Cholec80
from data import get_train_loader, get_eval_loader
import time
from PIL import Image, ImageOps
import torch.nn.functional as F
from args import num_workers, pose_dim, content_dim, channels, seed, lr, margin
from args import image_width, beta1, sd_weight, sim_weight, rec_weight, sd_nf, optimizer, niter
from args import triplet_weight, data_info
from triplet_loss import TripletLoss
import matplotlib.pyplot as plt
import pylab as pl
import pdb
import random

parser = argparse.ArgumentParser()
parser.add_argument('--content_model', default='dcgan_unet', help='model type (dcgan | dcgan_unet | vgg_unet)')
parser.add_argument('--pose_model', default='dcgan', help='model type (dcgan | unet | resnet)')
parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('-width', '--image_width', default=image_width, type=int, help='image_width')

parser.add_argument('--lr', default=lr, type=float, help='learning rate')
parser.add_argument('-rw', '--rec_weight', default=rec_weight, type=float, help='rec_weight')
parser.add_argument('-sw', '--sim_weight', default=sim_weight, type=float, help='sim_weight')
parser.add_argument('-aw', '--sd_weight', default=sd_weight, type=float, help='sd_weight')
parser.add_argument('-tw', '--triplet_weight', default=triplet_weight, type=float, help='triplet_weight')

parser.add_argument('-cd', '--content_dim', default=content_dim, type=int, help='content_dim')
parser.add_argument('-pd', '--pose_dim', default=pose_dim, type=int, help='pose_dim')
parser.add_argument('--margin', default=margin, type=int, help='margin')

parser.add_argument('--niter', default=niter, type=int, help='niter')
parser.add_argument('-data', '--data_type', default=None, type=str, help='data_type')
parser.add_argument('-path', '--file_path', default=None, type=str, help='file_path')

parser.add_argument('-temp', '--temp_pkl', default=True, type=bool, help='temp_pkl')
parser.add_argument('-tri', '--use_trip', default=False, type=bool, help='use_trip')

opt = parser.parse_args()
print(opt)

lr = opt.lr
rec_weight = opt.rec_weight
sim_weight = opt.sim_weight
sd_weight = opt.sd_weight
triplet_weight = opt.triplet_weight
content_dim = opt.content_dim
pose_dim = opt.pose_dim
nbits = content_dim + pose_dim
margin = opt.margin
niter = opt.niter
image_width = opt.image_width
data_type = opt.data_type
temp_pkl = opt.temp_pkl
use_trip = opt.use_trip
file_path = opt.file_path

if data_type is None:
    print('please input the name of datasets: cholec80 or inhouse')

if data_type == 'cholec80':
    batch_size, max_frames, test_batch_size, max_step, num_sample, save_dir, \
    clip_length, downsample_rate, data_root_80, labels_name, data_root_test, temp_labels_name = data_info(data_type)
elif data_type == 'inhouse':
    batch_size, max_frames, test_batch_size, max_step, num_sample, save_dir, \
    clip_length, downsample_rate, data_root_80, data_root_inhouse_test, labels_name, temp_labels_name = data_info(data_type)


current_time = str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

file_path = save_dir + str(data_type) + '_' + str(image_width) + 'x' + str(image_width) + '_' + str(nbits) + '_bits_' + \
            str(content_dim) + '+' + str(pose_dim) + '_weight_' + str(rec_weight) + '+' + str(sim_weight) + '+' \
            + str(sd_weight) + '+' + str(triplet_weight) + '_lr_' + str(lr) + '_mg_' + str(margin) \
            + '_trip_' + current_time
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

if image_width == 64:
    import models.resnet_64 as resnet_models
    import models.dcgan_64 as dcgan_models
    import models.dcgan_unet_64 as dcgan_unet_models
    import models.vgg_unet_64 as vgg_unet_models
elif image_width == 128:
    import models.resnet_128 as resnet_models
    import models.dcgan_128 as dcgan_models
    import models.dcgan_unet_128 as dcgan_unet_models
    import models.vgg_unet_128 as vgg_unet_models

if opt.content_model == 'dcgan_unet':
    netEC = dcgan_unet_models.content_encoder(content_dim, channels)
    netD = dcgan_unet_models.decoder(content_dim, pose_dim, channels)
elif opt.content_model == 'vgg_unet':
    netEC = vgg_unet_models.content_encoder(content_dim, channels)
    netD = vgg_unet_models.decoder(content_dim, pose_dim, channels)
elif opt.content_model == 'dcgan':
    netEC = dcgan_models.content_encoder(content_dim, channels)
    netD = dcgan_models.decoder(content_dim, pose_dim, channels)
else:
    raise ValueError('Unknown content model: %s' % opt.content_model)

if opt.pose_model == 'dcgan':
    netEP = dcgan_models.pose_encoder(pose_dim, channels, normalize=opt.normalize)
elif opt.pose_model == 'resnet':
    netEP = resnet_models.pose_encoder(pose_dim, channels, normalize=opt.normalize)
else:
    raise ValueError('Unknown pose model: %s' % opt.pose_model)

import models.classifiers as classifiers
netC = classifiers.scene_discriminator(pose_dim, sd_nf)

netEC.apply(utils.init_weights)
netEP.apply(utils.init_weights)
netD.apply(utils.init_weights)
netC.apply(utils.init_weights)

# ---------------- optimizers ----------------
if optimizer == 'adam':
    optimizer = optim.Adam
elif optimizer == 'rmsprop':
    optimizer = optim.RMSprop
elif optimizer == 'sgd':
    optimizer = optim.SGD
else:
  raise ValueError('Unknown optimizer: %s' % optimizer)

optimizerC = optimizer(netC.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerEC = optimizer(netEC.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerEP = optimizer(netEP.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optimizer(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
bce_criterion = nn.BCELoss()

# --------- transfer to gpu ------------------------------------
netEP.cuda()
netEC.cuda()
netD.cuda()
netC.cuda()
mse_criterion.cuda()
bce_criterion.cuda()
triplet_loss = TripletLoss(margin, device).to(device)

# --------- load a dataset ------------------------------------

def get_frames_path_train(data_root):
    frame_path_all_all = []
    frame_path_all_1 = []
    frame_path_all_2 = []
    img_list_train_all = []
    img_list_test_all = []

    video_idx = os.listdir(data_root)
    video_idx.sort(key=lambda x: int(x.split('.')[0]))
    train_idx_1 = 0
    stop_idx_1 = int(len(video_idx) * 0.5)
    #print('self.stop_idx', stop_idx_1)
    train_idx_2 = int(len(video_idx) * 0.5)
    stop_idx_2 = len(video_idx)
    drnet = True
    print('loading training data')

    for i in range(train_idx_1, stop_idx_1):
        #print(1)
        #print('----------------', i)
        video_path = os.path.join(data_root, video_idx[i])
        #print(2)
        frame = os.listdir(video_path)
        frame.sort(key=lambda x: int(x.split('.')[0]))
        #print(3)
        for ii in frame:
            ii_num = int(ii.split('.')[0])
            if ii_num % max_frames == 0:
                seq_list = []
                img_list_train = []
                #print('****************', ii)
                if drnet:
                    for j in range(ii_num, ii_num + max_frames, max_frames//4):
                        img_names = os.path.join(video_path, str(j) + '.jpg')
                        #print(11)
                        '''imgs = self.loader(img_names)
                        if self.transform is not None:
                            imgs = self.transform(imgs)
                        imgs = np.array(imgs)'''
                        im = Image.open(img_names)
                        # imr = im.convert('RGB')
                        imm = im.resize((image_width, image_width), Image.ANTIALIAS)
                        immm = np.array(imm).astype(np.float32) / 225
                        img_list_train.append(immm)
                        seq_list.append(img_names)
                        #print(12)
                    # print('sequence_list', len(seq_list), len(seq_list[0]))
                    img_list_train_all.append(np.array(img_list_train))
                    frame_path_all_1.append(np.array(seq_list))
                    #print(len(frame_path_all_1), '++++++++++++++++')
                    #print(len(img_list_train_all), '++++++++++++++++')
                # print('frame_path_all', len(frame_path_all), len(frame_path_all[0]), len(frame_path_all[0][0][0]))
    return np.array(img_list_train_all).astype(np.float32)


def get_frames_path_train_inhouse(data_root):
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
    #print(len(video_idx), train_idx_1, stop_idx_1)
    drnet = True
    sum_clips = 0
    print('loading training data')

    for i in range(train_idx_1, stop_idx_1):
        #print(1)
        print('----------------', i, end='\r')
        video_path = os.path.join(data_root, video_idx[i])
        #print(2)
        frame = os.listdir(video_path)
        frame.sort(key=lambda x: int(x.split('.')[0]))
        #print(3)
        #print("video", i, video_idx[i], "frame_num_before", len(frame))
        final_num_clip = len(frame) // int(max_frames*downsample_step)
        sum_clips = sum_clips + final_num_clip
        frame = frame[:int(final_num_clip*max_frames*downsample_step)]
        #print("video", i, video_idx[i], "frame_num_after", len(frame),
              #"final_num_clip", final_num_clip, "sum_clips", sum_clips)

        for ii in frame:
            ii_num_origin = int(ii.split('.')[0])
            ii_num = int((ii_num_origin+1) / 2)
            if (ii_num-1) % int(max_frames*downsample_step) == 0:
                seq_list = []
                img_list_train = []
                #print('****************', ii)
                if drnet:
                    for j in range(ii_num, ii_num + int(max_frames*downsample_step), int((max_frames*downsample_step)//4)):
                        img_names = os.path.join(video_path, str(int(j*2-1)) + '.jpg')
                        #print(11)
                        '''imgs = self.loader(img_names)
                        if self.transform is not None:
                            imgs = self.transform(imgs)
                        imgs = np.array(imgs)'''
                        im = Image.open(img_names)
                        # imr = im.convert('RGB')
                        imm = im.resize((image_width, image_width), Image.ANTIALIAS)
                        immm = np.array(imm).astype(np.float32) / 225
                        img_list_train.append(immm)
                        seq_list.append(img_names)
                        #print(12)
                    # print('sequence_list', len(seq_list), len(seq_list[0]))
                    img_list_train_all.append(np.array(img_list_train))
                    frame_path_all_1.append(np.array(seq_list))
                    #print(len(frame_path_all_1), '++++++++++++++++')
                    #print(len(img_list_train_all), '++++++++++++++++')
                # print('frame_path_all', len(frame_path_all), len(frame_path_all[0]), len(frame_path_all[0][0][0]))
        num_sample = sum_clips
        #print('num_sample', num_sample)
    print('\n')
    return np.array(img_list_train_all).astype(np.float32)


if data_type == 'cholec80':
    img_train = get_frames_path_train(data_root_80)
elif data_type == 'inhouse':
    img_train = get_frames_path_train_inhouse(data_root_80)

train_loader = get_train_loader(img_train, batch_size)
#test_loader = get_eval_loader(opt.data_root, img_test)

# -------------------------- load test data --------------------------
def get_frames_path_test(data_root):
    frame_path_all_1 = []
    frame_path_all_2 = []
    img_list_train_all = []
    img_list_test_all = []

    video_idx = os.listdir(data_root)
    video_idx.sort(key=lambda x: int(x.split('.')[0]))
    len(video_idx)
    train_idx_1 = 0
    stop_idx_1 = int(len(video_idx) * 0.5)
    #print('self.stop_idx', stop_idx_1)
    train_idx_2 = int(len(video_idx) * 0.5)
    stop_idx_2 = len(video_idx)
    #print(train_idx_2, stop_idx_2)
    print('loading testing data')

    for i in range(train_idx_2, stop_idx_2):
        #print(1)
        print('----------------', i, end='\r')
        video_path = os.path.join(data_root, video_idx[i])
        #print(2)
        frame = os.listdir(video_path)
        frame.sort(key=lambda x: int(x.split('.')[0]))
        #print(3)
        for ii in frame:
            ii_num = int(ii.split('.')[0])
            if ii_num % max_frames == 0:
                seq_list = []
                img_list_test = []
                #print('****************', ii)
                for j in range(ii_num, ii_num + max_frames, int(max_frames//4)):
                    img_names = os.path.join(video_path, str(j) + '.jpg')
                    #print(11)
                    '''imgs = self.loader(img_names)
                    if self.transform is not None:
                        imgs = self.transform(imgs)
                    imgs = np.array(imgs)'''
                    im = Image.open(img_names)
                    # imr = im.convert('RGB')
                    imm = im.resize((image_width, image_width), Image.ANTIALIAS)
                    immm = np.array(imm).astype(np.float32) / 225
                    img_list_test.append(immm)
                    seq_list.append(img_names)
                    #print(12)
                # print('sequence_list', len(seq_list), len(seq_list[0]))
                frame_path_all_2.append(np.array(seq_list))
                img_list_test_all.append(np.array(img_list_test))
                # print('frame_path_all', len(frame_path_all), len(frame_path_all[0]), len(frame_path_all[0][0][0]))
    print('\n')
    return np.array(img_list_test_all).astype(np.float32)


def get_frames_path_test_inhouse(data_root):
    downsample_step = downsample_rate / 25
    frame_path_all_1 = []
    frame_path_all_2 = []
    img_list_train_all = []
    img_list_test_all = []

    # video_idx = os.listdir(data_root)
    # video_idx.sort(key=lambda x: int(x.split('.')[0]))
    video_idx = sorted(os.listdir(data_root))
    len(video_idx)
    train_idx_1 = 0
    stop_idx_1 = int(len(video_idx) * 0.5)
    # print('self.stop_idx', stop_idx_1)
    train_idx_2 = int(len(video_idx) * 0.5)
    stop_idx_2 = len(video_idx)
    #print(train_idx_2, stop_idx_2)
    drnet = True
    sum_clips = 0
    ten_sample = [28,30,31,34,35,36,38,39,40,41]
    ten_sample = [28,30,31,34,40,41]
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
        final_num_clip = len(frame) // int(max_frames*downsample_step)
        sum_clips = sum_clips + final_num_clip
        frame = frame[:int(final_num_clip * max_frames * downsample_step)]
        #print("video", i, video_idx[i], "frame_num_after", len(frame),
             # "final_num_clip", final_num_clip, "sum_clips", sum_clips)

        for ii in frame:
            ii_num = int(ii.split('.')[0])
            if (ii_num-1) % int(max_frames*downsample_step) == 0:
                seq_list = []
                img_list_test = []
                #print('****************', ii)
                if drnet:
                    for j in range(ii_num, ii_num + int(max_frames*downsample_step), int(max_frames*downsample_step//4)):
                        img_names = os.path.join(video_path, str(j) + '.jpg')
                        #print(11)
                        '''imgs = self.loader(img_names)
                        if self.transform is not None:
                            imgs = self.transform(imgs)
                        imgs = np.array(imgs)'''
                        im = Image.open(img_names)
                        # imr = im.convert('RGB')
                        imm = im.resize((image_width, image_width), Image.ANTIALIAS)
                        immm = np.array(imm).astype(np.float32) / 225
                        img_list_test.append(immm)
                        seq_list.append(img_names)
                        #print(12)
                    # print('sequence_list', len(seq_list), len(seq_list[0]))
                    frame_path_all_2.append(np.array(seq_list))
                    img_list_test_all.append(np.array(img_list_test))
                    #print(len(frame_path_all_2), '++++++++++++++++')
                    #print(len(img_list_test_all), '++++++++++++++++')
                # print('frame_path_all', len(frame_path_all), len(frame_path_all[0]), len(frame_path_all[0][0][0]))
        num_sample = sum_clips
        #print('num_sample', num_sample)
    print('\n')
    return np.array(img_list_test_all).astype(np.float32), num_sample

if data_type == 'cholec80':
    img_test = get_frames_path_test(data_root_80)
elif data_type == 'inhouse':
    img_test, num_sample = get_frames_path_test_inhouse(data_root_inhouse_test)

#img_test = get_frames_path_test(data_root_80)
#print('img_test', img_test.shape)
#train_loader = get_train_loader(opt.data_root, img_train)
test_loader = get_eval_loader(img_test, test_batch_size)


# ------------------  funtions ------------------------------------
class Round3(Function):
    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        output = torch.round(input)
        ctx.input = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask = ~(ctx.input==0)
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
    # pdb.set_trace()
    out = 2.*Round3.apply(y)-1.
    return out


# --------- training funtions ------------------------------------
def train(x):

    netEP.zero_grad()
    netEC.zero_grad()
    netD.zero_grad()

    # torch.cuda.empty_cache()  # shifang xiancun

    x_c1 = x[0]
    x_c2 = x[1]
    x_p1 = x[2]
    x_p2 = x[3]

    # content_encoder 生成的特征已经经过 tanh 归一化处理了
    h_c1 = netEC(x_c1)
    h_c2 = netEC(x_c2)[0].detach() if opt.content_model[-4:] == 'unet' else netEC(x_c2).detach()  # used as target for sim loss
    h_c3 = netEC(x_p1)[0].detach() if opt.content_model[-4:] == 'unet' else netEC(x_c2).detach()
    h_p1 = netEP(x_p1) # used for scene discriminator
    h_p2 = netEP(x_p2).detach()

    # use trip_loss or sim_loss to calculate loss
    # similarity loss: ||h_c1 - h_c2||
    if use_trip is False:
        sim_loss = mse_criterion(h_c1[0] if opt.content_model[-4:] == 'unet' else h_c1, h_c2)
        x_weight = sim_weight
        x_loss = sim_loss

    # triplet loss: ||a-p||2 - ||a-n||2 + margin
    elif use_trip is True:
        batch_list = list(range(int(batch_size)))
        batch_rand = list(range(int(batch_size)))
        random.shuffle(batch_rand)

        # 用全部的三元组计算loss
        # triplets = [[pos_item, pos_item, neg_item] for pos_item in batch_list for neg_item in batch_list]
        #pdb.set_trace()

        # 不用全部的数作为 triplets， 而是随机选取
        triplets = []
        for i in batch_list:
            trip_generate = [i, i, batch_rand[i]]
            triplets.append(trip_generate)
        #print(np.array(triplets).shape)

        for trips in triplets:
            if trips[0] == trips[2]:
                triplets.remove(trips)
        #print(np.array(triplets).shape)

        trip_loss = triplet_loss(h_c1[0], h_c2, h_c3, triplets)
        x_weight = triplet_weight
        x_loss = trip_loss

    # reconstruction loss: ||D(h_c1, h_p1), x_p1||
    #print('h_c1', len(h_c1), len(h_c1[0]), type(h_c1))
    # feature to hashing then reconstruction
    #if type(h_c1) is tuple:
    h_c11 = h_c1[0].detach()
    h_c21 = h_c2.detach()

    hc_hc1 = binary_tanh_unit(F.tanh(h_c11))
    hc_hc2 = binary_tanh_unit(F.tanh(h_c21))
    hc_hp1 = binary_tanh_unit(F.tanh(h_p1))

    #hc_hc1 = torch.sign(h_c11)
    #hc_hp1 = torch.sign(h_p1)

    #print('h_c1', len(hc_hc1), len(hc_hc1[0]), type(hc_hc1))
    hc_hc_tu = tuple((hc_hc2, h_c1[1]))
    #-------------------------------------
    rec = netD([hc_hc_tu, hc_hp1])
    rec_loss = mse_criterion(rec, x_p1)

    # scene discriminator loss: maximize entropy of output
    target = torch.cuda.FloatTensor(batch_size, 1).fill_(0.5)
    out = netC([h_p1, h_p2])
    sd_loss = bce_criterion(out, Variable(target))

    # full loss
    loss = x_weight * x_loss + rec_weight * rec_loss + sd_weight * sd_loss
    loss.backward()

    optimizerEC.step()
    optimizerEP.step()
    optimizerD.step()

    return x_weight * x_loss.data.cpu().numpy(), rec_weight * rec_loss.data.cpu().numpy(), \
                sd_weight * sd_loss.data.cpu().numpy()


def train_scene_discriminator(x):

    netC.zero_grad()
    #torch.cuda.empty_cache()  # shifang xiancun

    target = torch.cuda.FloatTensor(batch_size, 1)
    x1 = x[0]
    x2 = x[1]
    #print(len(x))
    #print(len(x1))
    #print(len(x2))
    h_p1 = netEP(x1).detach()
    h_p2 = netEP(x2).detach()

    half = int(batch_size/2)
    #print(half)
    rp = torch.randperm(half).cuda()
    h_p2[:half] = h_p2[rp]
    target[:half] = 1
    target[half:] = 0
    #print(target)

    out = netC([h_p1, h_p2])
    bce = bce_criterion(out, Variable(target))

    bce.backward()
    optimizerC.step()

    acc =out[:half].gt(0.5).sum() + out[half:].le(0.5).sum()
    return bce.data.cpu().numpy(), acc.data.cpu().numpy()/batch_size


def plot_loss(data_loss):
    plt_epoch = data_loss[:, 0]
    plt_rec = data_loss[:, 1]
    plt_sim = data_loss[:, 2]
    plt_sum = data_loss[:, 3]
    plt_acc = data_loss[:, 4]

    plt.figure()
    pl.plot(plt_epoch, plt_rec, 'g-', label='loss_rec')
    pl.plot(plt_epoch, plt_sim, 'b-', label='loss_sim')
    pl.plot(plt_epoch, plt_sum, 'r-', label='loss_sum')

    pl.xlabel('epoch')
    pl.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title('loss_curve')
    plt.savefig(file_path + '/loss_curve' + ".png")  # should before show method

    plt.figure()
    pl.plot(plt_epoch, plt_acc, 'b-', label='adv_acc')

    pl.xlabel('epoch')
    pl.ylabel('acc')
    plt.legend(loc='upper right')
    plt.title('acc_curve')
    plt.savefig(file_path + '/acc_curve' + ".png")  # should before show method

    plt.show()


# ---------------------------- test function ------------------------------
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
    hashcode = np.zeros((num_sample, nbits), dtype=np.float32)
    label_array = Array()
    batch_size = test_batch_size
    rem = num_sample % batch_size
    labels = sio.loadmat(labels_name)['labels']
    label_array.setmatrcs(labels)
    batch_num = len(test_loader)
    # print('batch_num', batch_num)

    with torch.no_grad():
        for i, (videos, index) in enumerate(test_loader):
            torch.cuda.empty_cache()
            batchsize = len(videos)
            videos = Variable(videos)
            videos.transpose_(0, 1)
            videos.transpose_(3, 4).transpose_(2, 3)
            videos = videos.to(device)

            x_c = videos[0]
            x_p = videos[np.random.randint(1, max_step)]
            h_c = netEC(x_c)
            h_p = netEP(x_p)

            if type(h_c) is tuple:
                h_c = h_c[0].detach()
            else:
                h_c = h_c.detach()

            feats_all = torch.cat([h_c, h_p], dim=1)
            feats_all = feats_all.view(batchsize, nbits)
            feats_co = h_c.view(batchsize, content_dim)
            feats_mo = h_p.view(batchsize, pose_dim)
            BinaryCode = torch.sign(feats_all)

            if i == batch_num-1 and rem != 0:
                # print('i', i)
                hashcode[i*batch_size:,:] = BinaryCode[:rem,:].data.cpu().numpy()
            else:
                hashcode[i*batch_size:(i+1)*batch_size,:] = BinaryCode.data.cpu().numpy()
            #print('i', i)

    hashcode_array = Array()
    test_hashcode = np.matrix(hashcode)

    Hamming_distance = 0.5 * (-np.dot(test_hashcode, test_hashcode.transpose()) + nbits)

    ''''# ------------------ temporal priori knowledge  --------------- todo too slow! delete it when not necessary!
    if temp_pkl == True:
        temp_labels = sio.loadmat(temp_labels_name)['labels']
        for i in range(temp_labels.shape[0]):
            for j in range(temp_labels.shape[0]):
                if temp_labels[i, j] == 0:
                    Hamming_distance[j, i] = nbits - 1

    # ------------------ temporal priori knowledge end -----------------------'''

    HammingRank = np.argsort(Hamming_distance, axis=0)

    labels = label_array.getmatrics()
    sim_labels = np.dot(labels, labels.transpose())

    map_print = []
    map_list = [1, 2, 5, 10, 20]
    for i in map_list:
        map, _, _ = tools.mAP(sim_labels, HammingRank, i)
        map_print.append(map)
        records.write('\ttopK: ' + str(i) + ', map: ' + str(map) + '\n')
        print('i:', i, ' map: ', map)
    # records.close()

    '''# mAP
    mAP = tools.calc_map(test_hashcode, test_hashcode, labels, labels)
    print('mAP', mAP)
    records_map.write('\tmAP: ' + str(mAP) + '\n')'''


def evaluate2(labels_name, uidx):
    hashcode = np.zeros((num_sample, nbits), dtype=np.float32)
    label_array = Array()
    batch_size = test_batch_size
    rem = num_sample % batch_size
    labels = sio.loadmat(labels_name)['labels']
    label_array.setmatrcs(labels)
    batch_num = len(test_loader)
    # print('batch_num', batch_num)

    with torch.no_grad():
        for i, (videos, index) in enumerate(test_loader):
            torch.cuda.empty_cache()
            batchsize = len(videos)
            videos = Variable(videos)
            #print(batchsize)
            #print(len(videos), len(videos[0]), len(videos[0][0]), len(videos[0][0][0]))
            #videos = videos.view(4, batchsize, 3, opt.image_width, opt.image_width)
            videos.transpose_(0, 1)
            videos.transpose_(3, 4).transpose_(2, 3)
            videos = videos.to(device)

            x_c = videos[0]
            x_p = videos[np.random.randint(1, max_step)]
            h_c = netEC(x_c)
            h_p = netEP(x_p)

            if type(h_c) is tuple:
                h_c = h_c[0].detach()
            else:
                h_c = h_c.detach()

            feats_all = torch.cat([h_c, h_p], dim=1)
            feats_all = feats_all.view(batchsize, nbits)
            feats_co = h_c.view(batchsize, content_dim)
            feats_mo = h_p.view(batchsize, pose_dim)
            BinaryCode = torch.sign(feats_all)


            if i == batch_num-1 and rem != 0:
                # print('i', i)
                hashcode[i*batch_size:,:] = BinaryCode[:rem,:].data.cpu().numpy()
            else:
                hashcode[i*batch_size:(i+1)*batch_size,:] = BinaryCode.data.cpu().numpy()
            #print('i', i)

    hashcode_array = Array()
    #labels = sio.loadmat(labels_name)['labels']
    #label_array.setmatrcs(labels)
    test_hashcode = np.matrix(hashcode)

    time1 = time.time()
    Hamming_distance = 0.5 * (-np.dot(test_hashcode, test_hashcode.transpose()) + nbits)
    time2 = time.time()
    #print('hamming distance computation costs: ', time2 - time1)
    np.savetxt(file_path + '/Hamming_distance_' + str(uidx) + '.txt', Hamming_distance)

    # ------------------ temporal prior knowledge  -----------------------
    if temp_pkl is True and uidx == 10:
        temp_labels = sio.loadmat(temp_labels_name)['labels']
        for i in range(temp_labels.shape[0]):
            for j in range(temp_labels.shape[0]):
                if temp_labels[i, j] == 0:
                    Hamming_distance[j, i] = nbits - 1
        np.savetxt(file_path + '/Hamming_distance_temp.txt', Hamming_distance)

    # ------------------ temporal priori knowledge end -----------------------

    #HammingRank = np.argsort(Hamming_distance, axis=0, kind='mergesort')
    HammingRank = np.argsort(Hamming_distance, axis=0)
    time3 = time.time()
    #print('hamming ranking costs: ', time3 - time2)
    np.savetxt(file_path + '/HammingRank_' + str(uidx) + '.txt', HammingRank)

    labels = label_array.getmatrics()
    #print('labels shape: ', labels.shape)
    sim_labels = np.dot(labels, labels.transpose())
    time6 = time.time()
    #print('similarity labels generation costs: ', time6 - time3)

    # records = open(file_path + '/64_0.1new.txt','w+')
    maps = []
    map_list = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 40, 50, 60, 80, 100]
    for i in map_list:
        map, _, _ = tools.mAP(sim_labels, HammingRank, i)
        maps.append(map)
        records.write('epoch: ' + str(uidx) + '\ttopK: ' + str(i) + '\tmap: ' + str(map) + '\n')
        #print('i: ', i, ' map: ', map, '\n')
    time7 = time.time()
    # records.close()
    map_all.append(maps)

    # mAP
    # pdb.set_trace()
    mAP = tools.calc_map(test_hashcode, test_hashcode, labels, labels)
    #print('test_hashcode', test_hashcode.shape, 'labels', labels.shape)
    #print('epoch: ', uidx, 'mAP', mAP)
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
#v = Cholec80(data_root_test, img_train)

num_epoch = 0
records_loss = open(file_path + '/loss_all.txt', 'a+')

for epoch in range(niter):

    num_epoch = num_epoch + 1

    netEP.train()
    netEC.train()
    netD.train()
    netC.train()

    epoch_sim_loss, epoch_rec_loss, epoch_sd_loss, epoch_sd_acc, epoch_adv_loss = 0, 0, 0, 0, 0
    train_video_num = 0
    #i = 0

    #progress = progressbar.ProgressBar(maxval=opt.epoch_size).start()

    for i, (videos, index) in enumerate(train_loader, start=1):
    # for videos in data_loader:
        torch.cuda.empty_cache()
        #progress.update(i + 1)
        #i = i + 1
        batchsize = len(videos)
        time1 = time.time()
        videos = Variable(videos)

        #videos = videos.view(4, batchsize, 3, opt.image_width, opt.image_width)
        videos.transpose_(0, 1)
        videos.transpose_(3, 4).transpose_(2, 3)
        videos = videos.to(device)
        time2 = time.time()
        #print('epoch', epoch, '  i', i, '  time', time2-time1)
        if batchsize < batch_size:
            break
        train_video_num = train_video_num + batchsize

        # train scene discriminator
        sd_loss, sd_acc = train_scene_discriminator(videos)
        epoch_sd_loss += sd_loss
        epoch_sd_acc += sd_acc

        # train main model
        sim_loss, rec_loss, adv_loss = train(videos)
        epoch_sim_loss += sim_loss
        epoch_rec_loss += rec_loss
        epoch_adv_loss += adv_loss

    rec_loss = epoch_rec_loss / train_video_num
    sim_loss = epoch_sim_loss / train_video_num
    adv_loss = epoch_adv_loss / train_video_num
    loss_sum = rec_loss + sim_loss + adv_loss
    acc_all = batch_size * epoch_sd_acc / train_video_num

    if (num_epoch % 2) == 0 or num_epoch == 1:
        print('[%02d] rec: %.4f | sim: %.4f | loss_sum: %.4f | scene_acc: %.3f%% ' % (
               epoch, rec_loss, sim_loss, loss_sum, 100 * batch_size * epoch_sd_acc / train_video_num))

    records_loss.write(str(int(num_epoch)) + '\t' + str(rec_loss) + '\t' + str(sim_loss) +
                       '\t ' + str(loss_sum) + '\t' + str(acc_all) + '\n')

    torch.save({
        'netD': netD,
        'netEP': netEP,
        'netEC': netEC,
        'opt': opt},
        '%s/model.pth' % file_path)

    # -------------------------- test part -------------------------------
    '''checkpoint = torch.load('%s/model.pth' % file_path)
    netD = checkpoint['netD']
    netEP = checkpoint['netEP']
    netEC = checkpoint['netEC']'''


    if (num_epoch % 4) == 0 or num_epoch == 1 or num_epoch == niter:
    #if num_epoch == niter:
        epoch_sim_loss, epoch_rec_loss, epoch_sd_loss, epoch_sd_acc = 0, 0, 0, 0

        netEP.eval()
        netEC.eval()
        netD.eval()

        records = open(file_path + '/record.txt', 'a+')
        records_map = open(file_path + '/records_map.txt', 'a+')
        map_all = []
        mAPs = []

        if num_epoch == niter:
            for i in range(11):
                print('eval_index:', i, end='\r')
                evaluate2(labels_name, i)
        else:
            evaluate(labels_name)

        records.close()
    # ------------------------- test part end ------------------------------

# finish training
records_loss.close()
data_loss = np.loadtxt(file_path + '/loss_all.txt')

plot_loss(data_loss)

'''
test_func = "python /media/f/ziyi/Video_Retrieval/UDCM_ALL/eval_drnet.py " + \
            '-cd {} -pd {} -path {} -width {} -data {}'.format(content_dim, pose_dim, file_path, image_width, data_type)

os.system(test_func)
'''
