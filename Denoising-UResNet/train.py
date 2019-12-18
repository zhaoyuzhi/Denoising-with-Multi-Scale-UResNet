import argparse
import os

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre-train ot not')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 10, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 1000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 50, help = 'number of epochs of training that ensures 100K training iterations')
    parser.add_argument('--train_batch_size', type = int, default = 128, help = 'size of the batches, 128 is recommended')
    parser.add_argument('--val_batch_size', type = int, default = 16, help = 'size of the batches, 1 is recommended')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.9, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--epoch_decreased', type = int, default = 30, help = 'the certain epoch that lr decreased')
    parser.add_argument('--lr_decreased', type = float, default = 0.0001, help = 'decreased learning rate at certain epoch')
    parser.add_argument('--num_workers', type = int, default = 4, help = 'number of cpu threads to use during batch generation')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ', type = str, default = 'relu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'bn', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input channels for generator')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output channels for generator')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for generator')
    parser.add_argument('--m_block', type = int, default = 17, help = 'number of convolutional blocks in DnCNN, 17 or 20')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--train_root', type = str, default = '../Documents/data/train', help = 'training images baseroot')
    parser.add_argument('--val_root', type = str, default = '../Documents/data/val', help = 'validation images baseroot')
    parser.add_argument('--save_root', type = str, default = './model', help = 'models saving root')
    parser.add_argument('--sample_root', type = str, default = './sample', help = 'sample images root')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    # Normal RGB parameters
    parser.add_argument('--geometry_aug', type = bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = bool, default = False, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    # QuadBayer parameters
    parser.add_argument('--short_expo_per_pattern', type = int, default = 2, help = 'QuadBayer short exposure number')
    parser.add_argument('--color_bias_aug', type = bool, default = False, help = 'max scaling factor')
    parser.add_argument('--color_bias_level', type = bool, default = 0.0, help = 'max scaling factor')
    parser.add_argument('--noise_aug', type = bool, default = True, help = 'max scaling factor')
    parser.add_argument('--mu', type = int, default = 0, help = 'Gaussian noise mean')
    parser.add_argument('--sigma', type = float, default = 0.04, help = 'Gaussian noise variance')
    parser.add_argument('--max_white_value', type = int, default = 255, help = 'dynamic range of images')
    parser.add_argument('--extra_process_train_data', type = bool, default = False, help = 'whether enhance short exposure pixels by multipling 4')
    parser.add_argument('--cover_long_exposure', type = bool, default = False, help = 'whether make long exposure pixels equal to 0')
    parser.add_argument('--return_mode', type = str, default = 'short_denoising', help = 'dataset return mode')
    opt = parser.parse_args()

    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    
    # ----------------------------------------
    #                 Trainer
    # ----------------------------------------
    trainer.Trainer(opt)
