import torch
import torch.nn.functional as F
from torch import nn


class UNet(nn.Module):
    '''
    U-Net for a dose prediction network of depth 4.
    Parameters
    ----------

    channels_in : int, optional
        number of channels of the input image. The default is 1
    channels_out : int, optional
        number of classes of the input label image. The default is 5
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : int, optional
        Number of initial channels. The default is 64.
    channels_list: list, optional
        List of number of channels at every depth in case of customized number of channels.
    '''

    def __init__(self, channels_in=4, channels_out=1, depth=4, initial_channels=16, channels_list = None):

        super().__init__()
        self.depth = depth
        prev_channels = channels_in

        self.down_path = nn.ModuleList()
        for i in range(self.depth):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]
            if i < 2:
                self.down_path.append(ConvBlock(prev_channels, current_channels))
            elif i == 2:
                self.down_path.append(ConvBlock(prev_channels, current_channels, padding_2=1))
            elif i == 3:
                self.down_path.append(ConvBlock(prev_channels, current_channels, padding_1=1, padding_2=1))
            prev_channels = current_channels

        self.res_list = nn.ModuleList()

        self.up_path = nn.ModuleList()

        for i in reversed(range(self.depth-1)):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]
            if i == 2:
                self.up_path.append(UpBlock(prev_channels + current_channels, current_channels, padding_1=1))
            else:
                self.up_path.append(UpBlock(prev_channels + current_channels, current_channels))
            prev_channels = current_channels

            self.res_list.append(nn.Conv3d(channels_list[i], channels_out, kernel_size=1))

        # self.res_list.append(nn.Conv3d(channels_list[0], channels_out, kernel_size=1))

    def forward(self, x):
        blocks = []
        out = []
        for i, down in enumerate(self.down_path):
            # print(x.size())
            x = down(x)
            if i < self.depth - 1:
                blocks.append(x)
                x = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True,
                                  recompute_scale_factor=False)

        for i, (up, res) in enumerate(zip(self.up_path, self.res_list)):


            x = up(x, blocks[-i - 1])
            out.append(res(x))



        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_1=0, padding_2=0, LeakyReLU_slope=0.2):
        super().__init__()
        block = []

        block.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding_1 , padding_mode='zeros'))
        block.append(nn.BatchNorm3d(out_channels))
        block.append(nn.LeakyReLU(LeakyReLU_slope))

        block.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=padding_2, padding_mode='zeros'))
        block.append(nn.LeakyReLU(LeakyReLU_slope))
        block.append(nn.BatchNorm3d(out_channels))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_1=0):
        super().__init__()

        self.conv_block = ConvBlock(in_channels, out_channels, padding_1=padding_1)

    def forward(self, x, skip):
        # print(x.size())
        x_up_conv = F.interpolate(x, scale_factor=2.0, mode='trilinear', align_corners=True)
        # print(x_up_conv.size())
        lower = int((skip.shape[2] - x_up_conv.shape[2]) / 2)
        upper = int(skip.shape[2] - lower)
        # print(skip.size())
        cropped = skip[:, :, lower:upper, lower:upper, lower:upper]
        # print(cropped.size())
        out = torch.cat([x_up_conv, cropped], 1)
        # print(out.size())
        out = self.conv_block(out)
        return out

