import torch
import torch.nn as nn

from . import unet


class DoseNet(nn.Module):
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

    def __init__(self, in_channels=1, classes= 1, depth=5, initial_channels=32, channels_list = None):

        super().__init__()
        self.num_Classes = classes
        self.channels_list = channels_list
        # assert len(channels_list) == depth

        self.unet = unet.UNet(in_channels=in_channels, out_channels=classes, depth=depth,
                              initial_channels=initial_channels, channels_list= channels_list)


    def forward(self, fixed_segmentation=None, fixed_image=None, moving_dose=None):
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
        if fixed_image == None and moving_dose == None:
            input_image = fixed_segmentation
            print('model 1')
        elif fixed_segmentation == None and fixed_image == None:
            print('model 2')
            input_image = moving_dose
        elif moving_dose == None:
            input_image = torch.cat((fixed_segmentation, fixed_image), dim=1) # (n, 1, d, h, w)
            print('model 3')
        else:
            input_image = torch.cat((fixed_segmentation, fixed_image, moving_dose), dim=1)  # (n, 1, d, h, w)
            print('model 4')
        logits_list = self.unet(input_image)
        # probs_list = [F.softmax(x, dim=1) for x in logits_list]
        # predicted_label_list = [torch.max(x, dim=1, keepdim=True)[1] for x in probs_list]

        res = {'logits_low': logits_list[0], 'logits_mid': logits_list[1], 'logits_high': logits_list[2]}

        return res