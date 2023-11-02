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

class AccelerationDataset(BaseLowdimDataset):
    """
    This dataset is a basic dataset for accelerated diffusion (pipline diffuison)
    """
    def __init__(self, 
            zarr_path, 
            horizon=16,
            pad_before=0,
            pad_after=0,
            state_key='state',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            diffusion_step = 10,
            only_predict_base = True
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[state_key, action_key])
        
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
            sequence_length=horizon + diffusion_step - 1,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        assert (train_mask & val_mask).sum() == 0
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.diffusion_step = diffusion_step
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.only_predict_base = only_predict_base
        if self.only_predict_base:
            # base pose + joint + base_vels = 37
            self.action_mask = list(range(0,7)) + list(range(13, 37)) + list(range(40, 46))
        else:
            self.action_mask = list(range(0, 46))

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon + self.diffusion_step - 1,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        data = {
            'obs': sample[self.state_key], # T, D_o
            'action': sample[self.action_key][:, self.action_mask], # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        data['action'] = data['action'][-self.horizon:]
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
