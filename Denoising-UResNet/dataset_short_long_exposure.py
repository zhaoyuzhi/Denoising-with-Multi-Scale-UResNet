import os
import random
import numpy as np
import cv2
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]

# QuadBayer Denoising 
class DenoisingDataset(Dataset):
    def __init__(self, opt, baseroot):                                  # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(baseroot)

    def __getitem__(self, index):

        ### Note!!!
        ### pattern is B G  ,  and opencv read image as BGR
        ###            G R

        ### specify the pos for short and long exposure pixels
        if self.opt.short_expo_per_pattern == 2:
            short_pos = [[0,0], [1,1]]
            long_pos = [[0,1], [1,0]]
        if self.opt.short_expo_per_pattern == 3:
            short_pos = [[0,0], [0,1], [1,0]]
            long_pos = [[1,1]]

        ### read an image
        img = cv2.imread(self.imglist[index], -1) # preserve the original dynamic range
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        noisy_img = np.zeros(img.shape, dtype = np.float64)

        # normalization
        img = img / self.opt.max_white_value

        ### specify the mask region of short and long: 1 represents valid, 0 represents none
        mask_short = np.zeros((img.shape[0], img.shape[1], 1), dtype = np.uint8)
        mask_long = np.ones((img.shape[0], img.shape[1], 1), dtype = np.uint8)
        for pos in short_pos:
            mask_short[pos[0]::2,pos[1]::2,:] += 1
        mask_long = mask_long - mask_short

        ### data augmentation
        # color bias, only for short exposure pixels
        if self.opt.color_bias_aug:
            blue_coeff = np.random.random_sample() * 2 * self.opt.color_bias_level + (1 - self.opt.color_bias_level)
            green_coeff = np.random.random_sample() * 2 * self.opt.color_bias_level + (1 - self.opt.color_bias_level)
            red_coeff = np.random.random_sample() * 2 * self.opt.color_bias_level + (1 - self.opt.color_bias_level)
            blue_offset = np.random.random_sample() * 0.1 * self.opt.color_bias_level
            green_offset = np.random.random_sample() * 0.1 * self.opt.color_bias_level
            red_offset = np.random.random_sample() * 0.1 * self.opt.color_bias_level
            for pos in short_pos:
                img[pos[0]::4,pos[1]::4,0] = img[pos[0]::4,pos[1]::4,0] * blue_coeff + blue_offset
                img[pos[0]::4,pos[1]+2::4,1] = img[pos[0]::4,pos[1]+2::4,1] * green_coeff + green_offset
                img[pos[0]+2::4,pos[1]::4,1] = img[pos[0]+2::4,pos[1]::4,1] * green_coeff + green_offset
                img[pos[0]+2::4,pos[1]+2::4,2] = img[pos[0]+2::4,pos[1]+2::4,2] * red_coeff + red_offset
            img = np.clip(img, 0, 1)
        
        # noise augmentation, only for short exposure pixels
        if self.opt.noise_aug:
            noise = np.random.normal(self.opt.mu, self.opt.sigma, img.shape)
            for pos in short_pos:
                noisy_img[pos[0]::4,pos[1]::4,0] = img[pos[0]::4,pos[1]::4,0] + noise[pos[0]::4,pos[1]::4,0]
                noisy_img[pos[0]::4,pos[1]+2::4,1] = img[pos[0]::4,pos[1]::4,0] + noise[pos[0]::4,pos[1]+2::4,1]
                noisy_img[pos[0]+2::4,pos[1]::4,1] = img[pos[0]::4,pos[1]::4,0] + noise[pos[0]+2::4,pos[1]::4,1]
                noisy_img[pos[0]+2::4,pos[1]+2::4,2] = img[pos[0]::4,pos[1]::4,0] + noise[pos[0]+2::4,pos[1]+2::4,2]
            img = np.clip(img, 0, 1)
        else:
            noisy_img = img

        # make long exposure term equal to 0
        if self.opt.cover_long_exposure:
            for pos in long_pos:
                img[pos[0]::2,pos[1]::2,:] *= 0

        # augment short exposure data
        if self.opt.extra_process_train_data:
            for pos in short_pos:
                img[pos[0]::2,pos[1]::2,:] *= 4

        # random crop
        h, w = img.shape[:2]
        rand_h = random.randint(0, h-self.opt.crop_size)
        rand_w = random.randint(0, w-self.opt.crop_size)
        img = img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]
        noisy_img = noisy_img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]
        mask_short = mask_short[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]
        mask_long = mask_long[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]
        
        ### return mode
        if self.opt.return_mode == 'short_denoising':
            img_short = np.zeros((img.shape[0] // 2, img.shape[1] // 2, img.shape[2] * 2), dtype = np.float64)
            noisy_img_short = np.zeros((img.shape[0] // 2, img.shape[1] // 2, img.shape[2] * 2), dtype = np.float64)
            if mask_short[0, 0, 0] == 1: # upper left pixel is short exposure
                img_short[:, :, :3] = img[0::2, 0::2, :]
                img_short[:, :, 3:6] = img[1::2, 1::2, :]
                noisy_img_short[:, :, :3] = noisy_img[0::2, 0::2, :]
                noisy_img_short[:, :, 3:6] = noisy_img[1::2, 1::2, :]
            else: # upper left pixel is not short exposure
                img_short[:, :, :3] = img[0::2, 1::2, :]
                img_short[:, :, 3:6] = img[1::2, 0::2, :]
                noisy_img_short[:, :, :3] = noisy_img[0::2, 1::2, :]
                noisy_img_short[:, :, 3:6] = noisy_img[1::2, 0::2, :]

        ### to tensor
        img_short = torch.from_numpy(img_short).float().permute(2, 0, 1).contiguous()
        noisy_img_short = torch.from_numpy(noisy_img_short).float().permute(2, 0, 1).contiguous()

        return img_short, noisy_img_short
    
    def __len__(self):
        return len(self.imglist)

class FullResDenoisingDataset(Dataset):
    def __init__(self, opt, baseroot):                          	    # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(baseroot)

    def __getitem__(self, index):

        ## read an image
        img = cv2.imread(self.imglist[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # normalization
        img = img / self.opt.max_white_value

        ## re-arrange the data for fitting network
        H_in = img[0].shape[0]
        W_in = img[0].shape[1]
        H_out = int(math.floor(H_in / 4)) * 4
        W_out = int(math.floor(W_in / 4)) * 4
        img = cv2.resize(img, (W_out, H_out))
        
        # add noise
        img = img.astype(np.float32) # RGB image in range [0, 255]
        noise = np.random.normal(self.opt.mu, self.opt.sigma, img.shape).astype(np.float32)
        noisy_img = img + noise

        # normalization
        img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        noisy_img = torch.from_numpy(noisy_img.transpose(2, 0, 1)).contiguous()

        return noisy_img, img
    
    def __len__(self):
        return len(self.imglist)

