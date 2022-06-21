import torch
import torch.nn as nn
from . import unet_deep


class DoseNet_Good(nn.Module):
    '''
    CNN dose prediction network.
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
        self.channels_out = channels_out
        self.channels_list = channels_list
        if channels_list != None:
            assert len(channels_list) == depth

        self.unet = unet_deep.UNet(channels_in=channels_in, channels_out=channels_out, depth=depth,
                              initial_channels=initial_channels, channels_list= channels_list)


    def forward(self, fixed_image, moving_image=None, moving_contours=None, moving_dose=None, fixed_contours=None, fixed_torso=None):
        '''
        Parameters
        ----------
        input_image : (n, c, d, h, w)
            The first dimension contains the number of patches.
        Returns
        -------
        dose : (n, classes, d, h, w)
            dose distributions of the images
        '''

        input_image = fixed_image                   # (n, 1, d, h, w)

        if moving_image is not None:
            input_image = torch.cat((input_image, moving_image), dim=1)

        if moving_contours is not None:
            input_image = torch.cat((input_image, moving_contours), dim=1)

        if moving_dose is not None:
            input_image = torch.cat((input_image, moving_dose), dim=1)

        if fixed_contours is not None:
            gtv = torch.where(fixed_contours == 4, 1, 0)
            sv = torch.where(fixed_contours == 3, 1, 0)
            rectum = torch.where(fixed_contours == 2, 1, 0)
            bladder = torch.where(fixed_contours == 1, 1, 0)
            input_image = torch.cat((input_image, gtv, sv, rectum, bladder), dim=1)

        if fixed_torso is not None:
            gtv_mask = torch.where(fixed_contours == 1, 1, 0)
            sv_mask = torch.where(fixed_contours == 2, 1, 0)
            input_image = torch.cat((input_image, gtv_mask, sv_mask), dim=1)

        dose_list = self.unet(input_image)          # (n, 1, d, h, w)

        res = {'dose_low': dose_list[0], 'dose_mid': dose_list[1], 'dose_high': dose_list[2]}

        return res