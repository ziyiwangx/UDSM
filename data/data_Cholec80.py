import random
import os
import numpy as np
import socket
import torch
from scipy import misc
import imageio
from PIL import Image
import torch.utils.data as data

class Cholec80(object):

    def __init__(self, train, data_root, seq_len=15, image_size=64, data_type='sequence'):
        self.data_root = data_root
        self.seq_len = seq_len
        self.image_size = image_size
        self.dirs = os.listdir(self.data_root)
        self.data_type = data_type
        self.dirs.sort(key=lambda x: int(x.split('.')[0]))

        if train:
            self.start_idx = 0
            self.stop_idx = int(len(self.dirs) * 0.5)
            print('Loaded train data(%d - %d objects)' % (self.start_idx, self.stop_idx))
        else:
            self.start_idx = int(len(self.dirs) * 0.5)
            self.stop_idx = len(self.dirs)
            print('Loaded test data(%d - %d objects)' % (self.start_idx, self.stop_idx))

        self.seed_set = False

    def get_sequence(self):
        #print('10')
        t = self.seq_len  #
        frame_name = []

        idx = np.random.randint(self.start_idx, self.stop_idx)
        obj_dir = self.dirs[idx]  # 选定某个文件夹
        #print('obj_dir', obj_dir)
        video_root = os.path.join(self.data_root, obj_dir)  # 该文件夹路径
        #print('video_root', video_root)
        frame_name = os.listdir(video_root)
        #print('frame_name', frame_name)
        #frame_name.sort(key=lambda x: int(x.split('.')[0]))
        video_clip_num = int(len(frame_name)//20)
        #print('video_clip_num', video_clip_num)
        fid_idx_start = np.random.randint(video_clip_num) * 20  # 20的整数倍 clip起始值
        fid_idx_end = fid_idx_start + (20 - t)
        #print('fid_idx_start', fid_idx_start)
        #print('fid_idx_end', fid_idx_end)
        st = random.randint(fid_idx_start, fid_idx_end)
        #print('st', st)
        #print('11')
        seq = []
        for i in range(st, st + t):
            fname = '%s/%d.jpg' % (video_root, i)
            #im = imageio.imread(fname) / 255.
            im = Image.open(fname)
            #imr = im.convert('RGB')
            imm = im.resize((64, 64), Image.ANTIALIAS)
            immm = np.array(imm)/225
            seq.append(immm)
            #print('12')
        #print('13')
        #print(len(seq))
        return np.array(seq)

    # to speed up training of drnet, don't get a whole sequence when we only need 4 frames
    # x_c1, x_c2, x_p1, x_p2
    def get_drnet_data(self):
        '''c_idx = np.random.randint(len(self.classes))
        c = self.classes[c_idx]
        vid_idx = np.random.randint(len(self.data[c]))
        vid = self.data[c][vid_idx]
        seq_idx = np.random.randint(len(vid['files']))
        dname = '%s/%s/%s' % (self.data_root, c, vid['vid'])
        seq_len = len(vid['files'][seq_idx])'''

        t = self.seq_len  #
        frame_name = []

        idx = np.random.randint(self.start_idx, self.stop_idx)
        obj_dir = self.dirs[idx]  # 选定某个文件夹
        #print('obj_dir', obj_dir)
        video_root = os.path.join(self.data_root, obj_dir)  # 该文件夹路径
        #print('video_root', video_root)
        frame_name = os.listdir(video_root)
        #print('frame_name', frame_name)
        #frame_name.sort(key=lambda x: int(x.split('.')[0]))
        video_clip_num = int(len(frame_name)//20)
        fid_idx_start = np.random.randint(video_clip_num) * 20  # 20的整数倍 clip起始值

        seq = []
        for i in range(4):
            t = np.random.randint(fid_idx_start, fid_idx_start + 19)
            #fname = '%s/%s' % (dname, vid['files'][seq_idx][t])
            fname = '%s/%d.jpg' % (video_root, t)
            im = Image.open(fname)
            # imr = im.convert('RGB')
            imm = im.resize((64, 64), Image.ANTIALIAS)
            immm = np.array(imm) / 225
            seq.append(immm)
        #print(len(seq), len(seq[0]), len(seq[0][0]), len(seq[0][0][0]))
        return np.array(seq)


    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            # torch.manual_seed(index)
        if self.data_type == 'sequence':
            return torch.from_numpy(self.get_sequence())
        elif self.data_type == 'drnet':
            return torch.from_numpy(self.get_drnet_data())
        else:
            raise ValueError('Unknown data type: %d. Valid type: drnet | sequence.' % self.data_type)

    def __len__(self):
        return len(self.dirs)


class Cholec80_test(data.Dataset):

    def __init__(self, feature_h5_path, test_assist_path, anchor_path):
        # pdb.set_trace()
        with h5py.File(feature_h5_path,'r') as h5_file:
          self.video_feats = h5_file['feats'][:]
        with h5py.File(anchor_path,'r') as h5_file:
          self.achors = h5_file['feats'][:]
        with h5py.File(test_assist_path,'r') as h5_file:
          self.z = h5_file['z'][:]
          self.neighbor = h5_file['pos'][:]



    def __getitem__(self, index):

        video_feat = torch.from_numpy(self.video_feats[index])
        neighbor = torch.from_numpy(self.neighbor[index])
        anchor1 = self.achors[neighbor[0]]
        anchor2 = self.achors[neighbor[1]]
        anchor3 = self.achors[neighbor[2]]
        return video_feat,anchor1,anchor2,anchor3

    def __len__(self):
        return len(self.video_feats)


