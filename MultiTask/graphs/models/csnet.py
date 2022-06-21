import torch
import torch.nn as nn

from utils.SpatialTransformer import SpatialTransformer
from . import cs_unet


class CSNet(nn.Module):
    '''
    Groupwise implicit template CNN registration method.
    Parameters
    ----------
    dim : int
        Dimension of input image.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : int, optional
        Number of initial channels. The default is 64.
    '''

    def __init__(self, channels_in_a, channels_out_a, depth_a, channels_in_b, channels_out_b, depth_b, initial_channels=16, channels_list = None):

        super().__init__()

        self.channels_out_a = channels_out_a
        self.channels_out_b = channels_out_b
        self.channels_list = channels_list

        if channels_list != None:
            assert len(channels_list) == max(depth_a, depth_b)

        self.unet = cs_unet.UNet(channels_in_a=channels_in_a, channels_out_a=channels_out_a, depth_a=depth_a, channels_in_b=channels_in_b, channels_out_b=channels_out_b, depth_b=depth_b, initial_channels=16, channels_list = channels_list)
        self.spatial_transform = SpatialTransformer(3)


    def forward(self, fixed_image, moving_image=None, fixed_segmentation=None, moving_segmentation=None, moving_dose=None, fixed_torso=None, is_reg=False):
        '''
        Parameters
        ----------
        fixed_image, moving_image: (n, 1, d, h, w)
            Fixed and moving image to be registered
        moving_label : optional, (n, 1, d, h, w)
            Moving label
        Returns
        -------
        warped_moving_image : (n, 1, d, h, w)
            Warped moving image.
        disp : (n, 3, d, h, w)
            Flow field from fixed image to moving image.
        '''

        input_seg = fixed_image
        input_reg = fixed_image
        input_dose = fixed_image

        if moving_image is not None:
            input_dose = torch.cat((input_dose, moving_image), dim=1)  # (n, 3, d, h, w)
            input_reg = torch.cat((input_reg, moving_image), dim=1)  # (n, 3, d, h, w)


        if fixed_segmentation is not None:
            input_dose = torch.cat((input_dose, fixed_segmentation), dim=1)  # (n, 3, d, h, w)

        if moving_segmentation is not None:
            input_seg = torch.cat((input_seg, moving_segmentation), dim=1)  # (n, 3, d, h, w)
            input_dose = torch.cat((input_dose, moving_segmentation), dim=1)  # (n, 3, d, h, w)

        if moving_dose is not None:
            input_dose = torch.cat((input_dose, moving_dose), dim=1)  # (n, 3, d, h, w)

        if fixed_torso is not None:
            input_dose = torch.cat((input_dose, fixed_torso), dim=1)  # (n, 3, d, h, w)

        # if self.args.network == 'CS_seg_reg':
        #     seg_list, dvf_list = self.unet(input_seg, input_reg)  # (n, 6, d, h, w), (n, 3, d, h, w)
        #
        #     res = {'seg_low': seg_list[0], 'seg_mid': seg_list[1], 'seg_high': seg_list[2],
        #        'dvf_low': dvf_list[0], 'dvf_mid': dvf_list[1], 'dvf_high': dvf_list[2]}

        if is_reg == False:
            seg_list, dose_list = self.unet(input_seg, input_dose)  # (n, 6, d, h, w), (n, 3, d, h, w)

            res = {'seg_low': seg_list[0], 'seg_mid': seg_list[1], 'seg_high': seg_list[2],
               'dose_low': dose_list[0], 'dose_mid': dose_list[1], 'dose_high': dose_list[2]}

        if is_reg == True:
            reg_list, dose_list = self.unet(input_reg, input_dose)  # (n, 6, d, h, w), (n, 3, d, h, w)

            res = {'dvf_low': reg_list[0], 'dvf_mid': reg_list[1], 'dvf_high': reg_list[2],
                   'dose_low': dose_list[0], 'dose_mid': dose_list[1], 'dose_high': dose_list[2]}

        return res