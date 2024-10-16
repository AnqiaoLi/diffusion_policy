from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

import matplotlib.pyplot as plt
import numpy as np
class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            mstep_prediction = False,
            add_noise = False,
            noise_range = 0.0,
            history_consistency = 10,
            res_iter = True,
            uniformly_downsample=0,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.mstep_prediction = mstep_prediction
        self.add_noise = add_noise
        self.noise_range = noise_range
        self.history_cosistency = history_consistency
        self.res_iter = res_iter
        self.kwargs = kwargs
        self.uniformly_downsample = uniformly_downsample
        self.observation_indices = torch.arange(self.n_obs_steps - 1, -1, -self.uniformly_downsample).flip(0)
        # self.horizon = 12

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        save_fancy_image = False
        if save_fancy_image:
            colors = np.linspace(0, 1, trajectory.shape[1])

        for idx, t in enumerate(scheduler.timesteps):
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            if save_fancy_image:
                plt.scatter(trajectory[0, :, 0].detach().cpu().numpy(), trajectory[0, :, 1].detach().cpu().numpy(),c = colors,cmap='viridis', s=2)
                plt.ylim(-1, 0.5)
                plt.xlim(-1.25, 1)
                plt.savefig('/home/anqiao/SP-DGDM/videos/diffusion_step_image/{:04d}.png'.format(idx))
                plt.clf()
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], commands_dict = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        
        # add observation noise
        obs_dict = obs_dict.copy()
        obs_dict['obs'] += torch.randn_like(obs_dict['obs'], device=self.device)*self.noise_range
        
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        if  self.mstep_prediction:
            # nobs = nobs[:, :, 2:]
            ncommands = self.normalizer['command'].normalize(commands_dict['command'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            if self.mstep_prediction:
                global_cond = torch.concat([nobs[:,:To].reshape(nobs.shape[0], -1), ncommands[:,:T].reshape(nobs.shape[0], -1)], dim=1)
            else:
                if self.uniformly_downsample > 1:
                    global_cond = nobs[:, :To, :]
                    global_cond = global_cond[:, self.observation_indices]
                    global_cond = global_cond.reshape(global_cond.shape[0], -1)
                else:
                    global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        # residual prediction
        if self.res_iter: 
            obs = obs_dict['obs']
            base_action = obs[:,To-1].unsqueeze(1).repeat(1, self.n_action_steps, 1).to(self.device)
            res_action = action[:, :, :self.obs_dim]
            full_action = action[:, :, self.obs_dim:]
            for i in range(self.n_action_steps):
                base_action[:, i:] += res_action[:, i:i+1]
            action = torch.cat([base_action, full_action], dim=-1)
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        # if self.mstep_prediction:
        #     # normalize the predicted action to be relative to the current state
        #     batch['action'][:, :, 0:2] = batch['action'][:, :, 0:2] - batch['obs'][:,self.n_obs_steps-1:self.n_obs_steps, 0:2]
        
        # add noise
        batch = batch.copy()
        if self.add_noise:
            batch['obs'] += torch.randn_like(batch['obs'], device=self.device)*self.noise_range
            
        # normalize input
        nbatch = self.normalizer.normalize(batch)

        if self.mstep_prediction:
            # nbatch['obs'] = nbatch['obs'][:, :, 2:]
            command = nbatch['command']
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            if self.mstep_prediction:
                global_cond = torch.concat([obs[:,:self.n_obs_steps,:].reshape(obs.shape[0], -1), 
                                            command[:,:self.horizon].reshape(obs.shape[0], -1)], dim=1)
            else:
                if self.uniformly_downsample > 1:
                    global_cond = obs[:, :self.n_obs_steps, :]
                    global_cond = global_cond[:, self.observation_indices]
                    global_cond = global_cond.reshape(global_cond.shape[0], -1)
                else:
                    global_cond = obs[:,:self.n_obs_steps,:].reshape(
                        obs.shape[0], -1)   
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)
        
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        weights = torch.ones_like(pred, dtype=torch.float32)
        weights[:, :self.n_obs_steps] *= self.history_cosistency  # Increase the weights for the first self.n_obs_steps horizon

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * weights
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
