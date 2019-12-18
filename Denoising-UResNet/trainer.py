import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import dataset
import utils

def Trainer(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Loss functions
    criterion_L2 = torch.nn.MSELoss().cuda()

    # Initialize DnCNN
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        # Set the learning rate to the specific value
        if epoch >= opt.epoch_decreased:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr_decreased

    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, network):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Judge name
        if not os.path.exists(opt.save_root):
            os.makedirs(opt.save_root)
        # Save model dict
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                modelname = 'DnCNN_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.train_batch_size, opt.mu, opt.sigma)
                modelpath = os.path.join(opt.save_root, modelname)
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(network.module.state_dict(), modelpath)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                modelname = 'DnCNN_iter%d_bs%d_mu%d_sigma%d.pth' % (iteration, opt.train_batch_size, opt.mu, opt.sigma)
                modelpath = os.path.join(opt.save_root, modelname)
                if iteration % opt.save_by_iter == 0:
                    torch.save(network.module.state_dict(), modelpath)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                modelname = 'DnCNN_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.train_batch_size, opt.mu, opt.sigma)
                modelpath = os.path.join(opt.save_root, modelname)
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(network.state_dict(), modelpath)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                modelname = 'DnCNN_iter%d_bs%d_mu%d_sigma%d.pth' % (iteration, opt.train_batch_size, opt.mu, opt.sigma)
                modelpath = os.path.join(opt.save_root, modelname)
                if iteration % opt.save_by_iter == 0:
                    torch.save(network.state_dict(), modelpath)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.DenoisingDataset(opt, opt.train_root)
    valset = dataset.DenoisingDataset(opt, opt.val_root)
    print('The overall number of training images:', len(trainset))
    print('The overall number of validation images:', len(valset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    val_loader = DataLoader(valset, batch_size = opt.val_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # Tensorboard
    writer = SummaryWriter()

    # For loop training
    for epoch in range(opt.epochs):

        # Record learning rate
        for param_group in optimizer_G.param_groups:
            writer.add_scalar('data/lr', param_group['lr'], epoch)
            print('learning rate = ', param_group['lr'])
        
        if epoch == 0:
            iters_done = 0

        ### Training
        for i, (noisy_img, img) in enumerate(train_loader):

            # To device
            noisy_img = noisy_img.cuda()
            img = img.cuda()
            gt_res_img = noisy_img - img

            # Train Generator
            optimizer_G.zero_grad()

            # Forword propagation
            res_img = generator(noisy_img)
            loss = criterion_L2(res_img, gt_res_img)
            recon_img = noisy_img - res_img

            # Record losses
            writer.add_scalar('data/L2Loss', loss.item(), iters_done)

            # Overall Loss and optimize
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Recon Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(train_loader), loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)

        ### Sampling
        utils.save_sample_png(opt, epoch, noisy_img, recon_img, img, addition_str = 'training')

        ### Validation
        val_PSNR = 0
        num_of_val_image = 0

        for j, (val_noisy_img, val_img) in enumerate(val_loader):
            
            # To device
            # A is for input image, B is for target image
            val_noisy_img = val_noisy_img.cuda()
            val_img = val_img.cuda()
            gt_res_img = noisy_img - img

            # Forward propagation
            res_img = generator(val_noisy_img)
            val_recon_img = val_noisy_img - res_img

            # Accumulate num of image and val_PSNR
            num_of_val_image += val_noisy_img.shape[0]
            val_PSNR += utils.psnr(val_recon_img, val_img, 1) * val_noisy_img.shape[0]

        val_PSNR = val_PSNR / num_of_val_image

        # Record average PSNR
        writer.add_scalar('data/val_PSNR', val_PSNR, epoch)
        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))

        ### Sampling
        utils.save_sample_png(opt, epoch, val_noisy_img, val_recon_img, val_img, addition_str = 'validation')

    writer.close()
