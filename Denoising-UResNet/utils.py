import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv
import os

import network

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def create_generator(opt):
    if opt.pre_train:
        # Initialize the network
        generator = network.UResNet464(opt)
        # Init the network
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        # Initialize the network
        generator = network.UResNet464(opt)
        # Load a pre-trained network
        pretrained_net = torch.load(opt.load_name + '.pth')
        load_dict(generator, pretrained_net)
        print('Generator is loaded!')
    return generator
    
def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(opt, epoch, noisy_img, recon_img, gt_img, addition_str = ''):
    # Save image one-by-one
    if not os.path.exists(opt.sample_root):
        os.makedirs(opt.sample_root)
    for i in range(2):
        # Recover normalization: * 255 because last layer is sigmoid activated
        if i == 0:
            img = noisy_img * 255
        if i == 1:
            img = recon_img * 255
        else:
            img = gt_img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, 255)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        # If there is QuadBayer
        if img_copy.shape[2] == 6:
            temp = np.zeros((img_copy.shape[0] * 2, img_copy.shape[1] * 2, img_copy.shape[2] // 2), dtype = np.uint8)
            temp[0::2, 0::2, :] = img_copy[:, :, :3]
            temp[1::2, 1::2, :] = img_copy[:, :, 3:6]
            img_copy = temp
        # Save to certain path
        if i == 0:
            save_img_name = 'epoch' + str(epoch) + addition_str + '_noisy.png'
        if i == 1:
            save_img_name = 'epoch' + str(epoch) + addition_str + '_recon.png'
        else:
            save_img_name = 'epoch' + str(epoch) + addition_str + '_gt.png'
        save_img_path = os.path.join(opt.sample_root, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def psnr(pred, target, pixel_max_cnt = 1):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    # this 1 represents last layer is sigmoid activation
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 1):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    # this 1 represents last layer is sigmoid activation
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim

# ----------------------------------------
#             PATH processing
# ----------------------------------------
def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(os.path.join(root,filespath)) 
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(filespath) 
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()
