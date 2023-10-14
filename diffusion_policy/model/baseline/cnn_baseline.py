from typing import Union
import logging
import torch
from torchvision import models
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange


logger = logging.getLogger(__name__)

class Vanilla_CNN(nn.Module):
    def __init__(self,
                 horizon:int = 16,
                 obs_dim: int = 82,
                 action_dim:int = 37):
        super().__init__()

        # Load the pre-trained ResNet18 model
        resnet = models.resnet18(pretrained=False)

        # Change the first convolutional layer
        # Assuming your data has a single channel (like grayscale), if not change the 'in_channels' argument
        resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), bias=False)
        
        # Remove average pooling layer and fully connected layer
        modules = list(resnet.children())[:-2]  # remove the last two layers
        self.resnet = nn.Sequential(*modules)
        
        # Custom layers to produce the desired output shape
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*1*6, 1024),  # This size might change based on the output from ResNet
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 16*37)
        )
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
    
    def forward(self, x):
        '''
            x : [ batch_size x obs_step x obs_dim ]

            returns:
            out : [ batch_size x out_channels x action_dim ]
        '''
        assert x.shape[1] == self.horizon and x.shape[2] == self.obs_dim
        x.unsqueeze_(1)
        x = self.resnet(x)
        x = self.fc(x)
        return x.view(-1, 16, 37)

if __name__ == "__main__":
    model = Vanilla_CNN().to('cuda')
    x = torch.zeros((2, 16, 82)).to('cuda')
    o = model(x)
    print(o.shape)