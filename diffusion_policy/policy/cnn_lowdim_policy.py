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
            mstep_prediction=False,
            pred_action_steps_only = False,
            normalize_action=True):
        super().__init__()
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.mstep_prediction = mstep_prediction  
        self.pred_action_steps_only = pred_action_steps_only  
        self.normalize_action = normalize_action

    
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
        if self.normalize_action:
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
        else:
            action_pred = naction_pred
        if self.pred_action_steps_only:
            action = action_pred[:, -self.n_action_steps:]
        else:
            action = action_pred
        result = {
            'action': action,
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
            naction_pred = self.model(obs, cond = nbatch['command'])
        else:
            naction_pred = self.model(nbatch['obs'])
        
        if not self.normalize_action:
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
            gt = batch['action']
        else:
            action_pred = naction_pred
            gt = nbatch['action']

        if self.pred_action_steps_only:
            action_pred = action_pred[:, -self.n_action_steps:]
            gt = gt[:, -self.n_action_steps:]

        loss = F.mse_loss(action_pred, gt)
        debug = False
        if debug:
            self.plot_debug(naction_pred, nbatch, i = 10)
        return loss
    
    def plot_debug(self, action_pred, nbatch, i=0):
        import matplotlib.pyplot as plt
            
        plt.plot(action_pred.detach().cpu().numpy()[i, :, 0], action_pred.detach().cpu().numpy()[i, :, 1], label='npred')
        plt.plot(nbatch['action'].detach().cpu().numpy()[i, :, 0], nbatch['action'].detach().cpu().numpy()[i, :, 1], label='ngt')
        plt.legend()
        plt.show()
        # second figure
        action_pred_unnorm = self.normalizer['action'].unnormalize(action_pred)
        gt = self.normalizer['action'].unnormalize(nbatch['action'])
        plt.plot(action_pred_unnorm.detach().cpu().numpy()[i, :, 0], action_pred_unnorm.detach().cpu().numpy()[i, :, 1], label='pred')
        plt.plot(gt.detach().cpu().numpy()[i, :, 0], gt.detach().cpu().numpy()[i, :, 1], label='gt')
        plt.legend()
        plt.show()  


