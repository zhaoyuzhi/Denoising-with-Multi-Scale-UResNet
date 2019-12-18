import torch
import torch.nn as nn
from network_module import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#           Denoising UResNet
# ----------------------------------------
class UResNet363(nn.Module):
    def __init__(self, opt):
        super(UResNet363, self).__init__()
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T2 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T3 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T4 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T5 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T6 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, scale_factor = 2)
        self.D3 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'none', norm = 'none')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 64 * 256 * 256
        E2 = self.E2(E1)                                        # out: batch * 128 * 128 * 128
        E3 = self.E3(E2)                                        # out: batch * 256 * 64 * 64
        # Bottleneck
        x = self.T1(E3)                                         # out: batch * 256 * 64 * 64
        x = self.T2(x)                                          # out: batch * 256 * 64 * 64
        x = self.T3(x)                                          # out: batch * 256 * 64 * 64
        x = self.T4(x)                                          # out: batch * 256 * 64 * 64
        x = self.T5(x)                                          # out: batch * 256 * 64 * 64
        x = self.T6(x)                                          # out: batch * 256 * 64 * 64
        # Decode the center code
        D1 = self.D1(x)                                         # out: batch * 128 * 128 * 128
        D1 = torch.cat((D1, E2), 1)                             # out: batch * 256 * 128 * 128
        D2 = self.D2(D1)                                        # out: batch * 64 * 256 * 256
        D2 = torch.cat((D2, E1), 1)                             # out: batch * 128 * 256 * 256
        x = self.D3(D2)                                         # out: batch * out_channel * 256 * 256

        return x

class UResNet464(nn.Module):
    def __init__(self, opt):
        super(UResNet464, self).__init__()
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T2 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T3 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T4 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T5 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T6 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, scale_factor = 2)
        self.D4 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'none', norm = 'none')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 64 * 256 * 256
        E2 = self.E2(E1)                                        # out: batch * 128 * 128 * 128
        E3 = self.E3(E2)                                        # out: batch * 256 * 64 * 64
        E4 = self.E4(E3)                                        # out: batch * 512 * 32 * 32
        # Bottleneck
        x = self.T1(E4)                                         # out: batch * 512 * 32 * 32
        x = self.T2(x)                                          # out: batch * 512 * 32 * 32
        x = self.T3(x)                                          # out: batch * 512 * 32 * 32
        x = self.T4(x)                                          # out: batch * 512 * 32 * 32
        x = self.T5(x)                                          # out: batch * 512 * 32 * 32
        x = self.T6(x)                                          # out: batch * 512 * 32 * 32
        # Decode the center code
        D1 = self.D1(x)                                         # out: batch * 256 * 64 * 64
        D1 = torch.cat((D1, E3), 1)                             # out: batch * 512 * 64 * 64
        D2 = self.D2(D1)                                        # out: batch * 128 * 128 * 128
        D2 = torch.cat((D2, E2), 1)                             # out: batch * 256 * 128 * 128
        D3 = self.D3(D2)                                        # out: batch * 64 * 256 * 256
        D3 = torch.cat((D3, E1), 1)                             # out: batch * 128 * 256 * 256
        x = self.D4(D3)                                         # out: batch * out_channel * 256 * 256

        return x

class UResNet565(nn.Module):
    def __init__(self, opt):
        super(UResNet565, self).__init__()
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.E5 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T2 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T3 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T4 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T5 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.T6 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, scale_factor = 2)
        self.D5 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'none', norm = 'none')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 64 * 256 * 256
        E2 = self.E2(E1)                                        # out: batch * 128 * 128 * 128
        E3 = self.E3(E2)                                        # out: batch * 256 * 64 * 64
        E4 = self.E4(E3)                                        # out: batch * 512 * 32 * 32
        E5 = self.E5(E4)                                        # out: batch * 512 * 16 * 16
        # Bottleneck
        x = self.T1(E5)                                         # out: batch * 512 * 16 * 16
        x = self.T2(x)                                          # out: batch * 512 * 16 * 16
        x = self.T3(x)                                          # out: batch * 512 * 16 * 16
        x = self.T4(x)                                          # out: batch * 512 * 16 * 16
        x = self.T5(x)                                          # out: batch * 512 * 16 * 16
        x = self.T6(x)                                          # out: batch * 512 * 16 * 16
        # Decode the center code
        D1 = self.D1(x)                                         # out: batch * 512 * 32 * 32
        D1 = torch.cat((D1, E4), 1)                             # out: batch * 1024 * 32 * 32
        D2 = self.D2(D1)                                        # out: batch * 256 * 64 * 64
        D2 = torch.cat((D2, E3), 1)                             # out: batch * 512 * 64 * 64
        D3 = self.D3(D2)                                        # out: batch * 128 * 128 * 128
        D3 = torch.cat((D3, E2), 1)                             # out: batch * 256 * 128 * 128
        D4 = self.D4(D3)                                        # out: batch * 64 * 256 * 256
        D4 = torch.cat((D4, E1), 1)                             # out: batch * 128 * 256 * 256
        x = self.D5(D4)                                         # out: batch * out_channel * 256 * 256

        return x
