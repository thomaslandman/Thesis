import torch
import torch.nn.functional as F
from torch import nn
import itertools

from .unet_deep import ConvBlock, UpBlock


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class UNet(nn.Module):
    '''
    U-net implementation with modifications.
        1. Works for input of 2D or 3D
        2. Change batch normalization to instance normalization

    Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

    Parameters
    ----------
    in_channels : int
        number of input channels.
    out_channels_a : int
        number of output channels.
    dim : (2 or 3), optional
        The dimention of input data. The default is 2.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : TYPE, optional
        Number of initial channels. The default is 32.
    channels_list: list, optional
        List of number of channels at every depth in case of customized number of channels.
    '''

    def __init__(self, channels_in_a, channels_out_a, depth_a, channels_in_b, channels_out_b, depth_b, initial_channels=16, channels_list = None):

        super().__init__()


        prev_channels_a = channels_in_a
        prev_channels_b = channels_in_b
        self.depth_a = depth_a
        self.depth_b = depth_b
        self.down_path_a = nn.ModuleList()
        self.down_path_b = nn.ModuleList()
        self.cs_unit_encoder = []
        self.cs_unit_decoder = []
        self.res_list_a = nn.ModuleList()
        self.res_list_b = nn.ModuleList()

        for i in range(depth_a):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]

            self.down_path_a.append(ConvBlock(prev_channels_a, current_channels))

            prev_channels_a = current_channels
            if i < self.depth_a-1:
                # define cross-stitch units
                self.cs_unit_encoder.append(nn.Parameter(0.5*torch.ones(prev_channels_a, 2, 2).cuda(), requires_grad=True))

        for i in range(self.depth_b):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]

            if depth_b == 3:
                self.down_path_b.append(ConvBlock(prev_channels_b, current_channels))

            if depth_b == 4:
                if i < 2:
                    self.down_path_b.append(ConvBlock(prev_channels_b, current_channels))
                elif i == 2:
                    self.down_path_b.append(ConvBlock(prev_channels_b, current_channels, padding_2=1))
                elif i == 3:
                    self.down_path_b.append(ConvBlock(prev_channels_b, current_channels, padding_1=1, padding_2=1))

            prev_channels_b = current_channels


        self.up_path_a = nn.ModuleList()
        self.up_path_b = nn.ModuleList()
        for i in reversed(range(self.depth_a - 1)):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]
            self.up_path_a.append(UpBlock(prev_channels_a+current_channels, current_channels))

            prev_channels_a = current_channels
            self.res_list_a.append(nn.Conv3d(channels_list[i + 1], channels_out_a, kernel_size=1))

            # define cross-stitch units
            self.cs_unit_decoder.append(nn.Parameter(0.5 * torch.ones(prev_channels_a, 2, 2).cuda(), requires_grad=True))

        for i in reversed(range(self.depth_b - 1)):
            if channels_list == None:
                current_channels = 2 ** i * initial_channels
            else:
                current_channels = channels_list[i]

            if depth_b == 3:
                self.up_path_b.append(UpBlock(prev_channels_b + current_channels, current_channels))

            if depth_b == 4:
                if i < 2:
                    self.up_path_b.append(UpBlock(prev_channels_b + current_channels, current_channels))
                elif i == 2:
                    self.up_path_b.append(UpBlock(prev_channels_b + current_channels, current_channels, padding_1=1))



            prev_channels_b = current_channels
            if depth_b != 4 or i != 2:
                self.res_list_b.append(nn.Conv3d(channels_list[i + 1], channels_out_b, kernel_size=1))

            
        self.res_list_a.append(nn.Conv3d(channels_list[0], channels_out_a, kernel_size=1))
        self.res_list_b.append(nn.Conv3d(channels_list[0], channels_out_b, kernel_size=1))

        self.cs_unit_encoder = torch.nn.ParameterList(self.cs_unit_encoder)
        self.cs_unit_decoder = torch.nn.ParameterList(self.cs_unit_decoder)


    def apply_cross_stitch(self, a, b, alpha):
        # print('apply cross-stitch:')
        # print(a.shape)

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
        out_b = out[1, :, :, :]  # [bs][n_f][x*y]

        out_a = out_a.view(shape)  # [bs][n_f][x][y]
        out_b = out_b.view(shape)  # [bs][n_f][x][y]

        return out_a, out_b

    def forward(self, x_a, x_b):

        blocks_a = []
        blocks_b = []
        out_a = []
        out_b = []


        for i, (down_a, down_b) in enumerate(itertools.zip_longest(self.down_path_a, self.down_path_b)):

            if i == 3:
                # print('hoi')
                blocks_b.append(x_b)
                x_b = F.interpolate(x_b, scale_factor=0.5, mode='trilinear', align_corners=True,
                                    recompute_scale_factor=False)
                x_b = down_b(x_b)
                continue
            # print(len(self.down_path_b))
            # print(i)
            x_a = down_a(x_a)
            x_b = down_b(x_b)

            if i < self.depth_a - 1:

                blocks_a.append(x_a)
                blocks_b.append(x_b)

                x_a = F.interpolate(x_a, scale_factor=0.5, mode='trilinear', align_corners=True,
                                      recompute_scale_factor=False)
                x_b = F.interpolate(x_b, scale_factor=0.5, mode='trilinear', align_corners=True,
                                      recompute_scale_factor=False)
                # print('shape after pooling:')
                # print(x_a.shape)
                x_a, x_b = self.apply_cross_stitch(x_a, x_b, self.cs_unit_encoder[i])



        for i, (up_a, up_b, res_a, res_b) in enumerate(zip(self.up_path_a, self.up_path_b[1:],
                                                                   self.res_list_a, self.res_list_b)):
            # print(i)
            if self.depth_b == 4 and i==0:
                x_b = self.up_path_b[0](x_b, blocks_b[-i - 1])
                # continue


            out_a.append(res_a(x_a))
            out_b.append(res_b(x_b))
            # print('add to res list')
            # print(res_a(x_a).shape)
            x_a_before = up_a(x_a, blocks_a[-i - 1])
            # print(x_b.shape)
            # print(blocks_b[-i-1].shape)
            x_b_before = up_b(x_b, blocks_b[-i - 2])
            # print(x_a_before.shape)
            x_a, x_b = self.apply_cross_stitch(x_a_before, x_b_before, self.cs_unit_decoder[i])

        out_a.append(self.res_list_a[-1](x_a))
        out_b.append(self.res_list_b[-1](x_b))
        # print('add to res list')
        # print(self.res_list_a[-1](x_a).shape)

        return out_a, out_b


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dim, normalization, LeakyReLU_slope=0.2):
#         super().__init__()
#         block = []
#         if dim == 2:
#             block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             if normalization:
#                 block.append(nn.InstanceNorm2d(out_channels))
#             block.append(nn.LeakyReLU(LeakyReLU_slope))
#         elif dim == 3:
#             block.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
#             if normalization:
#                 block.append(nn.InstanceNorm3d(out_channels))
#             block.append(nn.LeakyReLU(LeakyReLU_slope))
#         else:
#             raise (f'dim should be 2 or 3, got {dim}')
#         self.block = nn.Sequential(*block)
#
#     def forward(self, x):
#         out = self.block(x)
#         return out
#
#
# class UpBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dim, normalization):
#         super().__init__()
#         self.dim = dim
#         if dim == 2:
#             self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         elif dim == 3:
#             self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
#         self.conv_block = ConvBlock(in_channels, out_channels, dim, normalization)
#
#     def forward(self, x, skip):
#         x_up = F.interpolate(x, skip.shape[2:], mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=True)
#         x_up_conv = self.conv(x_up)
#         out = torch.cat([x_up_conv, skip], 1)
#         out = self.conv_block(out)
#         return out

