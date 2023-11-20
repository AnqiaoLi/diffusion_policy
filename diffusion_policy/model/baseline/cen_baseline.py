from typing import Union
import logging
import torch
from torchvision import models
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from diffusion_policy.model.baseline.cen_model import Forward_Dynamics_Model
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)

class CEN(nn.Module):
    def __init__(self,
                 horizon:int = 16,
                 obs_dim: int = 80,
                 action_dim:int = 37):
        super().__init__()
        with open('/home/anqiao/tmp/diffusion_policy/diffusion_policy/model/baseline/cen_config.yaml', 'r') as f:
            cfg = YAML().load(f)
        self.fdm = Forward_Dynamics_Model(state_encoding_config=cfg['architecture']['state_encoder'],
                                   command_encoding_config=cfg['architecture']['command_encoder'],
                                   recurrence_config=cfg['architecture']['recurrence'],
                                   prediction_config=cfg['architecture']['traj_predictor'],
                                   device='cuda')
        # # Load the pre-trained ResNet18 model
        # resnet = models.resnet18(pretrained=False)

        # # Change the first convolutional layer
        # # Assuming your data has a single channel (like grayscale), if not change the 'in_channels' argument
        # resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), bias=False)
        
        # # Remove average pooling layer and fully connected layer
        # modules = list(resnet.children())[:-2]  # remove the last two layers
        # self.resnet = nn.Sequential(*modules)
        
        # # Custom layers to produce the desired output shape
        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512*1*6, 1024),  # This size might change based on the output from ResNet
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 16*37)
        # )
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.training = True
    
    def forward(self, x):
        '''
            x : [ batch_size x obs_step x obs_dim ]

            returns:
            out : [ batch_size x out_channels x action_dim ]
        '''
        assert x.shape[1] == self.horizon and x.shape[2] == self.obs_dim
        state = x[:, :, :68].reshape(-1, self.horizon*68)
        command = torch.moveaxis(x[:, :, 68:], 1, 0) # (traj_len, n_sample, single_command_dim)

        _, out = self.fdm(state, command, training = True) # training is only used to output a tensor

        out = torch.moveaxis(out, 1, 0)
        return out

if __name__ == "__main__":
    model = CEN().to('cuda')
    x = torch.zeros((2, 16, 80)).to('cuda')
    o = model(x)
    print(o.shape)
    # with open('/home/anqiao/tmp/diffusion_policy/diffusion_policy/model/baseline/cen_config.yaml', 'r') as f:
    #     cfg = YAML().load(f)
    
    # model = Forward_Dynamics_Model(state_encoding_config=cfg['architecture']['state_encoder'],
    #                                command_encoding_config=cfg['architecture']['command_encoder'],
    #                                recurrence_config=cfg['architecture']['recurrence'],
    #                                prediction_config=cfg['architecture']['traj_predictor'],
    #                                device='cuda')
    # model.to('cuda')
    # """

    # :return:
    #     p_col: (traj_len, n_sample, 1)
    #     coordinate: (traj_len, n_sample, 2)
    # """

    # """
    # :param state: (n_sample, state_dim)
    # :param command_traj: (traj_len, n_sample, single_command_dim)
    # """
    # nsample = 100
    # state = torch.randn((nsample, 68*16)).cuda()
    # command_traj = torch.randn((16, nsample, 12)).cuda()
    # output = model(state, command_traj)


    # print(output[1].shape)


    