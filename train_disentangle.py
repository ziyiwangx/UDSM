import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import argparse
import os
import random
import numpy as np
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader
import utils
import scipy.io as sio
import tools
from data import get_train_loader, get_eval_loader, Cholec80
import time
from PIL import Image, ImageOps
import torch.nn.functional as F
from args import num_workers, pose_dim, content_dim, channels, seed, lr, margin
from args import image_width, beta1, sd_weight, rec_weight, sd_nf, optimizer, niter
from args import triplet_weight, data_info
from triplet_loss import TripletLoss
import matplotlib.pyplot as plt
import pylab as pl
import models.classifiers as classifiers
import models.dcgan_64 as dcgan_models

parser = argparse.ArgumentParser()
parser.add_argument('--content_model', default='dcgan', help='model type')
parser.add_argument('--pose_model', default='dcgan', help='model type')
parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('-width', '--image_width', default=image_width, type=int, help='image_width')

parser.add_argument('--lr', default=lr, type=float, help='learning rate')
parser.add_argument('-rw', '--rec_weight', default=rec_weight, type=float, help='rec_weight')
parser.add_argument('-aw', '--sd_weight', default=sd_weight, type=float, help='sd_weight')
parser.add_argument('-tw', '--triplet_weight', default=triplet_weight, type=float, help='triplet_weight')

parser.add_argument('-cd', '--content_dim', default=content_dim, type=int, help='content_dim')
parser.add_argument('-pd', '--pose_dim', default=pose_dim, type=int, help='pose_dim')
parser.add_argument('--margin', default=margin, type=int, help='margin')

parser.add_argument('--niter', default=niter, type=int, help='niter')
parser.add_argument('-data', '--data_type', default='Cholec80', type=str, help='data_type')

opt = parser.parse_args()
print(opt)

lr = opt.lr
rec_weight = opt.rec_weight
sd_weight = opt.sd_weight
triplet_weight = opt.triplet_weight
content_dim = opt.content_dim
pose_dim = opt.pose_dim
nbits = content_dim + pose_dim
margin = opt.margin
niter = opt.niter
image_width = opt.image_width
data_type = opt.data_type

# parameter input
batch_size, max_frames, test_batch_size, max_step, num_sample, save_dir, num_train,\
clip_length, downsample_rate, data_root_80, labels_name = data_info(data_type)

file_path = save_dir + str(data_type) + '_' + str(nbits) + '_' + str(content_dim) + '+' + str(pose_dim) + \
    '_w_' + str(rec_weight) + '+' + str(sd_weight) + '+' + str(triplet_weight) + '_lr_' + str(lr) + '_' + str(niter)
print('file_path', file_path)

if not os.path.exists(file_path):
    os.makedirs(file_path)

# ----------- random seed ----------
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
dtype = torch.cuda.FloatTensor

# -----------  settings ------------
gpu_usg = True
num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")
print('use_gpu', use_gpu, 'num_gpu',num_gpu)

netEC = dcgan_models.content_encoder(content_dim, channels)
netD = dcgan_models.decoder(content_dim, pose_dim, channels)
netEP = dcgan_models.pose_encoder(pose_dim, channels, normalize=opt.normalize)
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
    video_idx = os.listdir(data_root)
    video_idx.sort(key=lambda x: int(x.split('.')[0]))
    train_idx_1 = 0
    stop_idx_1 = int(len(video_idx) * 0.5)
    print('loading training data')

    array_train_all = np.zeros((num_train, 4, image_width, image_width, 3), dtype=np.float32)
    array_id = 0

    for i in range(train_idx_1, stop_idx_1):
        print('----------------', i, end='\r')
        video_path = os.path.join(data_root, video_idx[i])
        frame = os.listdir(video_path)
        frame.sort(key=lambda x: int(x.split('.')[0]))
        for ii in frame:
            ii_num = int(ii.split('.')[0])
            if ii_num % max_frames == 0:
                img_array_train = np.zeros((4, image_width, image_width, 3), dtype=np.float32)
                arr_im_id = 0
                for j in random.sample(range(ii_num, ii_num + max_frames), 4):
                    img_names = os.path.join(video_path, str(j) + '.jpg')
                    im = Image.open(img_names)
                    im_re = im.resize((image_width, image_width), Image.ANTIALIAS)
                    imm = np.array(im_re).astype(np.float32) / 255
                    img_array_train[arr_im_id] = imm
                    arr_im_id += 1
                array_train_all[array_id] = img_array_train
                array_id += 1

    return array_train_all


img_train = get_frames_path_train(data_root_80)
train_loader = get_train_loader(img_train, batch_size)


# -------------------------- load test data --------------------------
def get_frames_path_test(data_root):
    video_idx = os.listdir(data_root)
    video_idx.sort(key=lambda x: int(x.split('.')[0]))
    train_idx_2 = int(len(video_idx) * 0.5)
    stop_idx_2 = len(video_idx)
    print('\nloading testing data')

    array_test_all = np.zeros((num_sample, 4, image_width, image_width, 3), dtype=np.float32)
    array_id = 0

    for i in range(train_idx_2, stop_idx_2):
        print('----------------', i, end='\r')
        video_path = os.path.join(data_root, video_idx[i])
        frame = os.listdir(video_path)
        frame.sort(key=lambda x: int(x.split('.')[0]))
        for ii in frame:
            ii_num = int(ii.split('.')[0])
            if ii_num % max_frames == 0:
                img_array_test = np.zeros((4, image_width, image_width, 3), dtype=np.float32)
                arr_im_id = 0
                for j in random.sample(range(ii_num, ii_num + max_frames), 4):
                    img_names = os.path.join(video_path, str(j) + '.jpg')
                    im = Image.open(img_names)
                    im_re = im.resize((image_width, image_width), Image.ANTIALIAS)
                    imm = np.array(im_re).astype(np.float32) / 255
                    img_array_test[arr_im_id] = imm
                    arr_im_id += 1
                array_test_all[array_id] = img_array_test
                array_id += 1
    print('\n')
    return array_test_all


img_test = get_frames_path_test(data_root_80)
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


# --------- training functions ------------------------------------
def train(x, num_epoch):

    netEP.zero_grad()
    netEC.zero_grad()
    netD.zero_grad()

    x_c1 = x[0]
    x_c2 = x[1]
    x_c3 = x[random.randint(0,3)]

    x_p1 = x[1]   # t+k
    x_p2 = x[2]   # ta

    h_c1 = netEC(x_c1)           # anchor
    h_c2 = netEC(x_c2).detach()  # positive
    h_c3 = netEC(x_c3).detach()  # negative

    h_p1 = netEP(x_p1)
    h_p2 = netEP(x_p2).detach()

    # triplet loss

    batch_list = list(range(int(batch_size)))
    batch_rand = list(range(int(batch_size)))
    random.shuffle(batch_rand)

    triplets = []
    for i in batch_list:
        trip_generate = [i, i, batch_rand[i]]
        triplets.append(trip_generate)

    for trips in triplets:
        if trips[0] == trips[2]:
            triplets.remove(trips)

    trip_loss = triplet_loss(h_c1, h_c2, h_c3, triplets)

    # reconstruction loss
    hc_hc1 = binary_tanh_unit(F.tanh(h_c1))   # scene at t step
    hc_hp1 = binary_tanh_unit(F.tanh(h_p1))   # motion at t+k step

    rec = netD([hc_hc1, hc_hp1])
    rec_loss = mse_criterion(rec, x_p1)

    # scene discriminator loss
    target = torch.cuda.FloatTensor(batch_size, 1).fill_(0.5)
    out = netC([h_p1, h_p2])
    sd_loss = bce_criterion(out, Variable(target))

    # loss
    loss = triplet_weight * trip_loss + rec_weight * rec_loss + sd_weight * sd_loss
    loss.backward()

    optimizerEC.step()
    optimizerEP.step()
    optimizerD.step()

    return triplet_weight * trip_loss.data.cpu().numpy(), rec_weight * rec_loss.data.cpu().numpy(), \
           sd_weight * sd_loss.data.cpu().numpy()


def train_scene_discriminator(x):
    netC.zero_grad()

    target = torch.cuda.FloatTensor(batch_size, 1)
    x1 = x[1]
    x2 = x[2]

    h_p1 = netEP(x1).detach()
    h_p2 = netEP(x2).detach()

    half = int(batch_size/2)
    rp = torch.randperm(half).cuda()
    h_p2[:half] = h_p2[rp]
    target[:half] = 0
    target[half:] = 1
    # print(target)

    out = netC([h_p1, h_p2])
    bce = bce_criterion(out, Variable(target))

    bce.backward()
    optimizerC.step()

    acc =out[:half].le(0.5).sum() + out[half:].gt(0.5).sum()
    return bce.data.cpu().numpy(), acc.data.cpu().numpy()/batch_size


# --------- training loop ------------------------------------
num_epoch = 0
records_loss = open(file_path + '/loss_all.txt', 'a+')

for epoch in range(niter):
    num_epoch = num_epoch + 1

    netEP.train()
    netEC.train()
    netD.train()
    netC.train()

    epoch_sim_loss, epoch_rec_loss, epoch_sd_loss, epoch_adv_loss = 0, 0, 0, 0
    train_video_num = 0

    for i, (videos, index) in enumerate(train_loader, start=1):
        torch.cuda.empty_cache()
        batchsize = len(videos)
        videos = Variable(videos)

        videos.transpose_(0, 1)
        videos.transpose_(3, 4).transpose_(2, 3)
        videos = videos.to(device)

        if batchsize < batch_size:
            break
        train_video_num = train_video_num + batchsize

        sd_loss, _ = train_scene_discriminator(videos)
        epoch_sd_loss += sd_loss

        sim_loss, rec_loss, adv_loss = train(videos, num_epoch)
        epoch_sim_loss += sim_loss
        epoch_rec_loss += rec_loss
        epoch_adv_loss += adv_loss

    rec_loss = epoch_rec_loss / train_video_num
    sim_loss = epoch_sim_loss / train_video_num
    adv_loss = epoch_adv_loss / train_video_num
    loss_sum = rec_loss + sim_loss + adv_loss

    if (num_epoch % 5) == 0:
        print('[%02d] rec: %.4f | sim: %.4f | adv: %.4f | loss_sum: %.4f ' %
              (epoch, rec_loss, sim_loss, adv_loss, loss_sum))

    records_loss.write(str(int(num_epoch)) + '\t' + str(rec_loss) + '\t' + str(sim_loss) + '\t' + str(adv_loss) +
                       '\t ' + str(loss_sum) + '\n')

    torch.save({
        'netD': netD,
        'netEP': netEP,
        'netEC': netEC,
        'opt': opt},
        '%s/model.pth' % file_path)

    print('train disentangled feature finished')


