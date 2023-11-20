from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.normalizer import LinearNormalizer


logger = logging.getLogger(__name__)

class Vanilla_AE(nn.Module):
    def __init__(self, 
                 inchannels: int=1,
                 input_timestamp: bool=False,
                 input_dim: int=1280,
                 down_dims: list=[512, 256, 128],
                 n_obs_steps: int=16) -> None:
        super().__init__()

        # input_shape 1x1230
        all_dims = [input_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        # encoder 
        down_modules = nn.ModuleList([])
        for ind, (in_dim, out_dim) in enumerate(in_out):
            # down_modules.append(nn.Sequential(
            #     nn.Conv1d(in_dim, out_dim, 3, padding=1),
            #     nn.Linera(out_dim, out_dim),
            # ))
            down_modules.append(nn.Sequential(
                nn.Mish(),
                nn.Linear(in_dim, out_dim),
            ))
        
        # decoder
        up_modules = nn.ModuleList([])
        for ind, (out_dim, in_dim) in enumerate(reversed(in_out)):
            up_modules.append(nn.Sequential(
                nn.Mish(),
                nn.Linear(in_dim, out_dim),
            ))

        self.down_modules = down_modules
        self.up_modules = up_modules
        self.normalizer = LinearNormalizer()
        self.n_obs_steps = n_obs_steps

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, x):
        out = x
        for down_module in self.down_modules:
            out = down_module(out)
        for up_module in self.up_modules:
            out = up_module(out)
        return out
    
    def compute_loss(self, batch, per_batch = False):
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        x = obs[:,:self.n_obs_steps,:].reshape(
                    obs.shape[0], -1)
        out = self(x)
        if per_batch:
            # computer loss for each batch
            loss = torch.mean((out - x)**2, dim=1)
        else:
            loss = torch.mean((out - x)**2)
        return loss
    
    
if __name__ == "__main__":
    model = Vanilla_AE().cuda()
    x = torch.zeros((1,1230)).cuda()
    out = model(x)
    print(out.shape)