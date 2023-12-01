from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.quat import quat_rotate_numpy, quat_rotate_inverse_numpy
import tqdm


class SlippernessLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            state_key='state',
            command_key='command',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            n_obs_steps= 25,
            offset_action_type = 'last_obs'
            ):
        super().__init__()
        assert offset_action_type in ['last_obs', 'first_obs']
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[state_key, action_key, command_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        self.state_key = state_key
        self.action_key = action_key
        self.command_key = command_key
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.offset_action_type = offset_action_type

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self, mode='limits', offset_action_with_obs = False, **kwargs):
        if offset_action_with_obs:
            data = self._sample_to_data(self.replay_buffer, expand_action=True)
        else:
            data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample, expand_action=False):
        if expand_action:
            # only for generating the normalizer
            data = {
            'obs': sample[self.state_key][:, 2:36], # T, D_o
            'command': sample[self.command_key],
            #TODO: check if this is correct
            'action': self.expand_with_action_step(sample), # T, D_a
            }
        else:
            if self.offset_action_type == 'last_obs':
                # TODO Consistent with first_obs
                # data = {
                # 'obs': sample[self.state_key][:, 2:], # T, D_o
                # 'command': sample[self.command_key],
                # 'action': sample[self.action_key] - sample[self.state_key][self.n_obs_steps-1, 0:3], # T, D_a
                # }
                data = {
                'obs': sample[self.state_key][:, 2:36], # T, D_o
                'command': sample[self.command_key],
                'action': quat_rotate_inverse_numpy(sample[self.state_key][self.n_obs_steps-1:self.n_obs_steps, 36:40].repeat(self.horizon, axis=0), (sample[self.action_key] - sample[self.state_key][self.n_obs_steps-1, 0:3])), # T, D_a
                }
            elif self.offset_action_type == 'first_obs':
                data = {
                'obs': sample[self.state_key][:, 2:36], # T, D_o
                'command': sample[self.command_key],
                'action': quat_rotate_inverse_numpy(sample[self.state_key][0:1, 36:40].repeat(self.horizon, axis=0), (sample[self.action_key] - sample[self.state_key][0, 0:3])), # T, D_a
                # 'action': sample[self.action_key] - sample[self.state_key][0, 0:3], # T, D_a
                # 'action': quat_rotate_inverse_numpy(sample[self.state_key][:, 36:40], (sample[self.action_key] - sample[self.state_key][0, 0:3])), # T, D_a

                # 'action': np.linspace(0, 5, self.horizon+1)[1:, None].repeat(3, axis=1).astype('float32'), # T, D_a
                }
        return data

    def expand_with_action_step(self, sample):
        """ Expand the action to (n_sample, horizon, D_a), then offset with respect to the last observation. 
            This funciton is only used when normalizing the action
            args:
                sample: the sample from the replay buffer
                offset_action_type: 'last_obs' or 'fisrt_obs'    
        """
        # check if all the indices are with the same length
        buffer_start_idx, buffer_end_idx, _, _ = self.sampler.indices.T
        assert ((buffer_end_idx - buffer_start_idx) == 276).all()
        # expand action
        indices_buffer = buffer_start_idx[:, None] + np.arange(self.horizon)
        sample_action = sample[self.action_key][indices_buffer]
        # normalize with respect to the last observation
        if self.offset_action_type == 'last_obs':
            sample_action[:, :, 0:2] = sample_action[:, :, 0:2] - sample[self.state_key][buffer_start_idx+self.n_obs_steps-1,None, 0:2]
        elif self.offset_action_type == 'first_obs':
            # sample_action (n_sample, horizon, D_a)
            sample_action[:, :, 0:2] = sample_action[:, :, 0:2] - sample[self.state_key][buffer_start_idx,None, 0:2]
        quat = sample[self.state_key][buffer_start_idx,None, -4:].repeat(self.horizon, axis = 1)
        quat = quat.reshape(-1, quat.shape[-1])
        sample_action = sample_action.reshape(-1, sample_action.shape[-1])
        sample_action = quat_rotate_inverse_numpy(quat, sample_action)
        return sample_action

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)            
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
