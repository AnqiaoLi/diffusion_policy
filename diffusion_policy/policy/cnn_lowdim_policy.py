from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.baseline.resnet_baseline import Vanilla_CNN
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class CNNLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: Vanilla_CNN,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            mstep_prediction=False):
        super().__init__()
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.mstep_prediction = mstep_prediction    

    
    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], commands_dict = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        if self.mstep_prediction:
            nobs = nobs[:, :self.n_obs_steps, 2:]
            ncommands = self.normalizer['command'].normalize(commands_dict['command'])
        
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim

        if self.mstep_prediction:
            naction_pred = self.model(nobs, cond = ncommands)
        else:
            naction_pred = self.model(nobs)

        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        if self.mstep_prediction:
            # normalize the predicted action to be relative to the current state
            batch['action'][:, :, 0:2] = batch['action'][:, :, 0:2] - batch['obs'][:,self.n_obs_steps-1:self.n_obs_steps, 0:2]
        
        # normalize input
        nbatch = self.normalizer.normalize(batch)
        # remove x, y in input coordinate
        if self.mstep_prediction:
            nbatch['obs'] = nbatch['obs'][:, :, 2:]
            obs = nbatch['obs'][:, :self.n_obs_steps]
            action_pred = self.model(obs, cond = nbatch['command'])
        else:
            action_pred = self.model(nbatch['obs'])
        
        loss = F.mse_loss(action_pred, nbatch['action'])
        return loss
