import torch
import torch.nn.functional as F
from torch import nn


class UNet(nn.Module):
    '''
    U-net implementation.

    Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

    Parameters
    ----------
    in_channels : int
        number of input channels.
    out_channels : int
        number of output channels.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : TYPE, optional
        Number of initial channels. The default is 32.
    channels_list: list, optional
        List of number of channels at every depth in case of customized number of channels.
    '''

    def __init__(self, in_channels, out_channels, depth=5, initial_channels=32, channels_list= None):

        super().__init__()
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.cs_units = []
        self.res_list = nn.ModuleList()
        for i in range(self.depth):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]
            self.down_path.append(ConvBlock(prev_channels, current_channels))
            prev_channels = current_channels
            if i < self.depth - 1:
                # define cross-stitch units
                self.cs_units.append(
                    nn.Parameter(0.5 * torch.ones(prev_channels, 2, 2).cuda(), requires_grad=True))

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth-1)):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]
            self.up_path.append(UpBlock(prev_channels+current_channels, current_channels))
            prev_channels = current_channels
            self.res_list.append(nn.Conv3d(channels_list[i+1], out_channels, kernel_size=1))

        self.res_list.append(nn.Conv3d(channels_list[0], out_channels, kernel_size=1))
        self.cs_units = torch.nn.ParameterList(self.cs_units)

    def apply_cross_stitch(self, a, b, alpha):


        shape = a.shape
        newshape = [shape[0], shape[1],  shape[2] * shape[3] * shape[4]]  # [bs][n_f][x][y] ==> [bs][n_f][x*y]

        a_flat = a.view(newshape)  # [bs][n_f][x*y]
        b_flat = b.view(newshape)  # [bs][n_f][x*y]

        a_flat = torch.unsqueeze(a_flat, 2)  # [bs][n_f][1][x*y]
        b_flat = torch.unsqueeze(b_flat, 2)  # [bs][n_f][1][x*y]
        a_concat_b = torch.cat([a_flat, b_flat], dim=2)  # [bs][n_f][2][x*y]

        alphas_tiled = torch.unsqueeze(alpha, 0).repeat([shape[0], 1, 1, 1])  # [bs][n_f][2][2]

        out = torch.matmul(alphas_tiled, a_concat_b)  # [bs][n_f][2][2] * [bs][n_f][2][x*y] ==> [bs][n_f][2][x*y]
        out = out.permute(2, 0, 1, 3)  # [2][bs][n_f][x*y]

        out_a = out[0, :, :, :]  # [bs][n_f][x*y]
        # out_b = out[1, :, :, :]  # [bs][n_f][x*y]

        out_a = out_a.view(shape)  # [bs][n_f][x][y]
        # out_b = out_b.view(shape)  # [bs][n_f][x][y]

        return out_a #, out_b

    def forward(self, x, feature_maps):
        blocks = []
        out = []

        for i, down in enumerate(self.down_path):
            # print(x.shape)
            x = down(x)
            if i < self.depth - 1:
                blocks.append(x)
                x = self.apply_cross_stitch(x, feature_maps[2-i], self.cs_units[i])
                x = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True,
                                  recompute_scale_factor=False)

        for i, (up, res) in enumerate(zip(self.up_path, self.res_list)):
            if i == 0:
                # print(x.shape)
                out.append(res(x))
                feature_maps.append(x)
                x = up(x, blocks[-i - 1])
            else:
                # print(x.shape)
                out.append(res(x))
                feature_maps.append(x)
                x = up(x, blocks[-i - 1])

        out.append(self.res_list[-1](x))
        feature_maps.append(x)


        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, LeakyReLU_slope=0.2):
        super().__init__()
        block = []

        block.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1 , padding_mode='replicate'))
        block.append(nn.BatchNorm3d(out_channels))
        block.append(nn.LeakyReLU(LeakyReLU_slope))

        block.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'))
        block.append(nn.LeakyReLU(LeakyReLU_slope))
        block.append(nn.BatchNorm3d(out_channels))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        # print('before upconv')
        # print(x.shape)
        x_up_conv = F.interpolate(x, scale_factor=2.0, mode='trilinear', align_corners=True)
        # print('after upconv')
        # print(x_up_conv.shape)
        if skip.shape[2] < x_up_conv.shape[2]:
            cropped = F.pad(skip, (1,1,1,1,1,1))
        else:
            lower = int((skip.shape[2] - x_up_conv.shape[2]) / 2)
            upper = int(skip.shape[2] - lower)
            cropped = skip[:, :, lower:upper, lower:upper, lower:upper]
        out = torch.cat([x_up_conv, cropped], 1)
        # print('after concat')
        # print(out.shape)
        out = self.conv_block(out)
        # print('after conv')
        # print(out.shape)
        # print('tweede')
        # print(out.requires_grad)
        return out