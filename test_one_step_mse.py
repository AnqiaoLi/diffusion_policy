# Diffusion
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import dill
import hydra
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply

import time
import tqdm

import torch
from torch.utils.data import DataLoader


def one_step_mse(checkpoint_path = None, workspace = None, only_last_step = False):
    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg_diff = payload['cfg']
    cfg_diff = cfg_diff
    cls = hydra.utils.get_class(cfg_diff._target_)
    workspace = cls(cfg_diff)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg_diff.training.use_ema:
        policy = workspace.ema_model
    policy.eval()
    policy.to(device)
    # for name, child in self.workspace.model.model.named_children():
    #     print(name)

    # load data
    cfg_diff.task.dataset.zarr_path = "/home/anqiao/SP-DGDM/Data/traj_data_pyramid_terrain/traj_expand_obs_dim.zarr"
    # cfg_diff.val_dataloader.batch_size = 1
    dataset = hydra.utils.instantiate(cfg_diff.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, **cfg_diff.val_dataloader)
    train_dataloader = DataLoader(dataset, **cfg_diff.dataloader)
    
    # init error container
    epoch = 0
    mse = {"sum":0, "position":0, "quat":0, "linear_vel":0, "angular_vel":0, "joint_pos":0, "joint_vel":0}
    mse_sim = {"sum":0, "position":0, "quat":0, "linear_vel":0, "angular_vel":0, "joint_pos":0, "joint_vel":0}
    with tqdm.tqdm(val_dataloader, desc=f"One Step MSE", 
        leave=False, mininterval=cfg_diff.training.tqdm_interval_sec) as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            with torch.no_grad():
            # sample trajectory from training set, and evaluate difference
                obs_dict = {'obs': batch['obs']}
                gt_action = batch['action']
                if cfg_diff.policy.model.mrollouts:
                    torques_dict = {'torques': batch['torque']}
                    result = policy.predict_action(obs_dict, torques_dict)
                else:
                    s = time.time()
                    result = policy.predict_action(obs_dict)
                    # print("time: ", time.time() - s)
                if cfg_diff.pred_action_steps_only:
                    pred_action = result['action']
                    start = cfg_diff.n_obs_steps - 1
                    end = start + cfg_diff.n_action_steps
                    gt_action = gt_action[:,start:end]
                else:
                    pred_action = result['action_pred']
                # mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                # mse #
                index = -1 if only_last_step else 0
                mse['sum'] += torch.nn.functional.mse_loss(pred_action[:,index:], gt_action[:,index:])# sum
                mse["position"] += torch.nn.functional.mse_loss(pred_action[:,index:, 0:3], gt_action[:,index:,0:3])# position
                mse["quat"] += torch.nn.functional.mse_loss(pred_action[:,index:,3:7], gt_action[:,index:,3:7])# quat
                mse["linear_vel"] += torch.nn.functional.mse_loss(pred_action[:,index:,7:10], gt_action[:,index:,7:10])# linear_vel
                mse["angular_vel"] += torch.nn.functional.mse_loss(pred_action[:,index:,10:13], gt_action[:,index:,10:13])# angular_vel
                mse["joint_pos"] += torch.nn.functional.mse_loss(pred_action[:,index:,13:25], gt_action[:,index:,13:25])# joint_pos
                mse["joint_vel"] += torch.nn.functional.mse_loss(pred_action[:,index:,25:37], gt_action[:,index:,25:37])# joint_vel

                # mse_sim #
                pred_action.zero_()
                mse_sim['sum'] += torch.nn.functional.mse_loss(pred_action[:,index:], gt_action[:,index:])# sum
                mse_sim["position"] += torch.nn.functional.mse_loss(pred_action[:,index:, 0:3], gt_action[:,index:,0:3])# position
                mse_sim["quat"] += torch.nn.functional.mse_loss(pred_action[:,index:,3:7], gt_action[:,index:,3:7])# quat
                mse_sim["linear_vel"] += torch.nn.functional.mse_loss(pred_action[:,index:,7:10], gt_action[:,index:,7:10])# linear_vel
                mse_sim["angular_vel"] += torch.nn.functional.mse_loss(pred_action[:,index:,10:13], gt_action[:,index:,10:13])# angular_vel
                mse_sim["joint_pos"] += torch.nn.functional.mse_loss(pred_action[:,index:,13:25], gt_action[:,index:,13:25])# joint_pos
                mse_sim["joint_vel"] += torch.nn.functional.mse_loss(pred_action[:,index:,25:37], gt_action[:,index:,25:37])# joint_vel

    for key in mse.keys():
        mse[key] /= len(tepoch)
        mse_sim[key] /= len(tepoch)

    print("mse: ", mse)
    print("mse_sim: ", mse_sim)

if __name__ == "__main__":
    checkpoint_path = "/media/anqiao/My Passport/SP/SP-checkpoitns/2023.09.15_input_sim/20.07.29_input_sim/checkpoints/latest.ckpt"
    one_step_mse(checkpoint_path)