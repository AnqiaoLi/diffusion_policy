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
import time

class DiffusionUnetAcceleratedPolicy(BaseLowdimPolicy):
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
            acceleration = True,
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
        self.acceleration = acceleration
        self.kwargs = kwargs
        # if debug = True, the same global condition is used for all diffusion steps
        self.debug = False

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            global_cond_expend = None,
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

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            if self.acceleration:
                    model_output = model(trajectory, t,
                        local_cond=local_cond, global_cond=global_cond_expend[t] if not self.debug else global_cond_expend[0])
            else:
                model_output = model(trajectory, t, 
                    local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def warm_up(self, 
                warm_dict, 
                generator = None):
        """
        This function is used to generate a batch of noise for the first diffusion step.
        The most denoised action is denoised num_train_diffusion_steps times-1,and the 
        least one is a random noise.
        args:
            - warm_dict: a dictionary containing all the observation needed for the condtioning of first 
                          num_train_timesteps denoising steps
            -generator: a torch.Generator object for noise generation
        """
        # dnoise once need n_obs_steps observations, denoise num_train_diffusion_steps-1 times needs 
        # n_obs_steps+(num_train_diffusion_steps-2) observations
        assert warm_dict['obs'].shape[0] == 1
        assert warm_dict['obs'].shape[1] == self.n_obs_steps + self.noise_scheduler.config.num_train_timesteps-2
        # denoise from the intial trajectories to get the initial input
        action_bs = self.noise_scheduler.config.num_train_timesteps
        # build conditions for each diffusion steps
        obs = torch.zeros(size=(action_bs-1, self.horizon, self.obs_dim), device=self.device, dtype=self.dtype)
        for step_i in range(action_bs-1):
            obs[step_i] = warm_dict['obs'][:,step_i:step_i+self.horizon]
        nobs = self.normalizer['obs'].normalize(obs)

        # init the random noise
        noise = torch.randn(
            size = (action_bs, self.horizon, self.action_dim),
            dtype= self.dtype,
            device = self.device,
            generator=generator
        )
        # init the partially denoised noise 
        warm_noise = noise.clone()
        for t in range(self.noise_scheduler.config.num_train_timesteps-1):
            diff_t = torch.arange(action_bs-t-1, action_bs).to(self.device)
            model_output = self.model(noise[:t+1], diff_t, 
                    global_cond=nobs[t].reshape(1, -1).repeat(t+1, 1))
            for ba in range(t+1):
                noise[ba] = self.noise_scheduler.step(
                    model_output[ba], diff_t[ba], noise[ba], 
                    generator=generator,
                    **self.kwargs
                    ).prev_sample
            
        self.warm_noise = noise

    def predict_action_acceleration(self, obs_dict: Dict[str, torch.Tensor], generator = None):
        """
        Predict the action with one diffusion step.
        """
        assert 'obs' in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        # run epsilon estimation
        t = torch.arange(0, self.noise_scheduler.config.num_train_timesteps)
        model_output = self.model(self.warm_noise, t.to(self.device), 
                global_cond=nobs.reshape(1, -1).repeat(self.noise_scheduler.config.num_train_timesteps, 1))
        # noise update
        for i in range(self.noise_scheduler.config.num_train_timesteps):
            self.warm_noise[i] = self.noise_scheduler.step(
                model_output[i], t[i], self.warm_noise[i], 
                generator=generator,
                **self.kwargs
                ).prev_sample
        # output one noise 
        naction_pred = self.warm_noise[0].clone().unsqueeze(0)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # generate a new noise
        new_noise = torch.randn(
            size = (1, self.horizon, self.action_dim),
            dtype= self.dtype,
            device = self.device,
            generator=generator
        )
        self.warm_noise = torch.concat([self.warm_noise[1:], new_noise], dim=0)
        return {"action_pred": action_pred}

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], torque_dict = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        if  self.mstep_prediction:
            ntorques = self.normalizer['torque'].normalize(torque_dict['torques'])
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
                global_cond = torch.concat([nobs[:,:To].reshape(nobs.shape[0], -1), ntorques[:,:T].reshape(nobs.shape[0], -1)], dim=1)
            else:
                global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            if self.acceleration:
                # global_cond (bs, cond_dim) -> global_cond_expend (diff_step, bs, cond_dim)) 
                global_cond_expend = global_cond.unsqueeze(0).repeat(self.noise_scheduler.config.num_train_timesteps, 1, 1)
                for i in range(self.noise_scheduler.config.num_train_timesteps):
                    t = self.noise_scheduler.config.num_train_timesteps - i - 1
                    global_cond_expend[i] = nobs[:,t:t+self.n_obs_steps].reshape(nobs.shape[0], -1)
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
            global_cond_expend = global_cond_expend if self.acceleration else None,
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
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        if self.mstep_prediction:
            torque = nbatch['torque']
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
                                            torque[:,:self.horizon].reshape(obs.shape[0], -1)], dim=1)
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

        # change the condition based on the sampled timesteps
        if self.acceleration:
            global_cond = self.reform_for_acceleration(global_cond, timesteps, obs)
    
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

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def reform_for_acceleration(self, global_cond, timesteps, obs):
        """
        reform the global condition for acceleration based on the timestep.
        global_cond[i] for t = timestamp[i] is used in the t_th diffusion step, and thus should be the (diff_step - t - 1)th observation 
        args:
            - global_cond: (B,global_cond_dim)
        return:
            - global_cond: (B,global_cond_dim)

        """
        global_cond = global_cond.clone()
        for i in range(global_cond.shape[0]):
            start_idx = self.noise_scheduler.config.num_train_timesteps - timesteps[i] - 1
            if self.debug:
                start_idx = 9
            global_cond[i] = obs[i, start_idx:start_idx+self.n_obs_steps].reshape(1, -1)
        return global_cond