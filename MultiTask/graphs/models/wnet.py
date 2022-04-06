import torch
import torch.nn as nn
import torch.nn.functional as F
from . import w_unet1, w_unet2


class WNet(nn.Module):
    '''
    CNN segmentation network.
    Parameters
    ----------

    in_channels : int, optional
        number of channels of the input image. The default is 1
    classes : int, optional
        number of classes of the input label image. The default is 5
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : int, optional
        Number of initial channels. The default is 64.
    channels_list: list, optional
        List of number of channels at every depth in case of customized number of channels.
    '''

    def __init__(self, in_channels_1=2, classes_1=5, depth_1=3, in_channels_2=7, classes_2=1, depth_2=4, initial_channels=16, channels_list = None):

        super().__init__()
        # self.num_Classes = classes
        # self.channels_list = channels_list
        # assert len(channels_list) == depth

        self.unet_1 = w_unet1.UNet(in_channels=in_channels_1, out_channels=classes_1, depth=depth_1,
                              initial_channels=initial_channels, channels_list=[16,32,64])

        self.unet_2 = w_unet2.UNet(in_channels=in_channels_2, out_channels=classes_2, depth=depth_2,
                                  initial_channels=initial_channels, channels_list=[16,32,64,128])


    def forward(self, fixed_image, moving_image=None, moving_segmentation=None, moving_dose=None, fixed_segmentation=None):
        '''
        Parameters
        ----------
        input_image : (n, c, d, h, w)
            The first dimension contains the number of patches.
        Returns
        -------
        logits : (n, classes, d, h, w)
            logits of the images.
        '''

        input_1 = fixed_image
        if moving_image != None:
            input_1 = torch.cat((input_1, moving_image), dim=1)

        if moving_segmentation != None:
            input_1 = torch.cat((input_1, moving_segmentation), dim=1)

        logits_list, feature_maps = self.unet_1(input_1)

        # probs_list = [F.softmax(x, dim=1) for x in logits_list]
        # predicted_label_list = [torch.max(x, dim=1, keepdim=True)[1] for x in probs_list]

        res_1 = {'logits_low': logits_list[0], 'logits_mid': logits_list[1], 'logits_high': logits_list[2]}

        # # maxima = torch.max(logits_list[2], dim=1, keepdim=True)[0]

        ### Edit out or in

        input_2 = F.softmax(logits_list[2], dim=1)

        # print(input_2.shape)
        # print(input_2.dtype)
        # print(input_2.requires_grad)

        input_2 = torch.cat((input_2, fixed_image), dim=1)

        if moving_dose != None:
            input_2 = torch.cat((input_2, moving_dose), dim=1)


        if moving_image != None:
            input_2 = torch.cat((input_2, moving_image), dim=1)

        # print('second unet')
        logits_list = self.unet_2(input_2, feature_maps)

        res_2 = {'logits_low': logits_list[1], 'logits_mid': logits_list[2], 'logits_high': logits_list[3]}

        # print(len(res_2))

        return res_1, res_2