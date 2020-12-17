# coding: utf-8
import pickle
import h5py
import torch
import torch.utils.data as data
from PIL import Image, ImageOps
import random
import numbers
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time



def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

'''
class RandomCrop(object):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // max_frames)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // max_frames
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count // max_frames
        self.count += 1
        random.seed(seed)
        angle = random.randint(-self.degrees, self.degrees)
        return TF.rotate(img, angle)


class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count // max_frames
        self.count += 1
        random.seed(seed)
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img, brightness_factor)
        img_ = TF.adjust_contrast(img_, contrast_factor)
        img_ = TF.adjust_saturation(img_, saturation_factor)
        img_ = TF.adjust_hue(img_, hue_factor)

        return img_
'''

class Cholec80(Dataset):
    def __init__(self, feats_path, transform=None,
                 loader=pil_loader):
        #self.dir_root = dir_root
        self.feats_path = feats_path
        #self.sequence = sequence
        self.transform = transform
        self.loader = loader
        #self.img_rgb = self.get_frames_path()

    def __getitem__(self, index):
        #img_list_batch=[]
        #print(14)
        #img_rgb = self.get_frames_path()
        #img_rgb = self.feats_path
        '''a1 = len(self.feats_path)
        a2 = len(self.feats_path[0])
        for i in range(a1):
            img_list = []
            print(16)
            for ii in range(a2):
                im = Image.open(self.feats_path[i][ii])
                print(17)
                # imr = im.convert('RGB')
                imm = im.resize((64, 64), Image.ANTIALIAS)
                immm = np.array(imm) / 225
                print(18)
                img_list.append(immm)
            img_list_batch.append(img_list)
            print(19)
        img_output = np.array(img_list_batch)[index]
        print(20)'''
        time1 = time.time()
        img_output = np.array(self.feats_path[index])
        time2 = time.time()
        #print('img_output', len(img_output), len(img_output[0]), len(img_output[0][0]), len(img_output[0][0][0]))
        time3 = time.time()
        #print('time1', time2-time1, 'time2', time3-time2)
        return img_output, index

    def __len__(self):
        return len(self.feats_path)


def get_train_loader(path_train, batch_size, shuffle=True, num_workers=0, pin_memory=True):
    #print('feature_h5_path', feature_h5_path)
    train_transforms = None

    '''if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.40577063, 0.27282622, 0.28533617], [0.24071056, 0.19952665, 0.20165241])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(64),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.40577063, 0.27282622, 0.28533617], [0.24071056, 0.19952665, 0.20165241])
        ])'''

    # v = Cholec80(feature_h5_path, path_train, train, sequence, train_transforms)
    v = Cholec80(path_train, train_transforms)
    data_loader = torch.utils.data.DataLoader(dataset=v,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              #collate_fn=train_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


def get_eval_loader(path_test, batch_size, shuffle=False, num_workers=0, pin_memory=False):

    test_transforms = None

    '''if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.40577063, 0.27282622, 0.28533617], [0.24071056, 0.19952665, 0.20165241])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.40577063, 0.27282622, 0.28533617], [0.24071056, 0.19952665, 0.20165241])
        ])'''

    vd = Cholec80(path_test, test_transforms)

    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              #collate_fn=eval_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader
