import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from torch.utils.data import DataLoader
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.anomaly_detection.model import Vanilla_AE
from torch import optim
import tqdm
import wandb
import torch
import numpy as np
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent)
)

def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    
    # init
    global_step = 0
    epoch = 0
    best_models_val_loss = [] 

    # configures
    gradient_accumulate_every = 1

    # configure model
    ae_model = Vanilla_AE()

    # configure dataset
    dataset: BaseLowdimDataset
    dataset = hydra.utils.instantiate(cfg.dataset)
    assert isinstance(dataset, BaseLowdimDataset)
    train_dataloader = DataLoader(dataset, **cfg.dataloader)
    normalizer = dataset.get_normalizer(range_eps=1e-10, mode="limits")
    ae_model.set_normalizer(normalizer)

    # configure validation dataset
    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)   

    # configure optimizer
    optimizer = hydra.utils.instantiate(
        cfg.optimizer, params=ae_model.parameters())    # configure lr scheduler
    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=(
            len(train_dataloader) * cfg.training.num_epochs) \
                // cfg.training.gradient_accumulate_every,
        # pytorch assumes stepping LRScheduler every epoch
        # however huggingface diffusers steps it every batch
        last_epoch=global_step-1
    )
    # configure logging
    pathlib.Path(cfg.training.outdir).mkdir(parents=True, exist_ok=True)
    checkpoint_save_path = cfg.training.outdir + '/checkpoints'
    pathlib.Path(checkpoint_save_path).mkdir(parents=True, exist_ok=True)
    wandb_run = wandb.init(
        dir=str(cfg.training.outdir),
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.logging
    )
    wandb.config.update({"output_dir": cfg.training.outdir})
    # device transfer
    optimizer_to(optimizer, cfg.device)
    ae_model.to(cfg.device)
    #########################################################
    # Training
    for local_epoch_idx in range(cfg.training.num_epochs):
        train_losses = list()
        step_log = dict()
        with tqdm.tqdm(train_dataloader, desc=f"Training epoch {epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(cfg.device, non_blocking=True))
                raw_loss = ae_model.compute_loss(batch)
                loss = raw_loss / gradient_accumulate_every
                loss.backward()
                if global_step % cfg.training.gradient_accumulate_every == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                # logging
                raw_loss_cpu = raw_loss.item()
                tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                train_losses.append(raw_loss_cpu)
                step_log = {
                    'train_loss': raw_loss_cpu,
                    'global_step': global_step,
                    'epoch': epoch,
                    'lr': lr_scheduler.get_last_lr()[0]
                }

                is_last_batch = (batch_idx == (len(train_dataloader)-1))
                if not is_last_batch:
                    # log of last step is combined with validation and rollout
                    wandb_run.log(step_log, step=global_step)
                    global_step += 1
        
        train_loss = np.mean(train_losses)
        step_log['train_loss'] = train_loss

        # ========= eval for this epoch ==========
        ae_model.eval()

        if (epoch % cfg.training.val_every) == 0:
            with torch.no_grad():
                with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    val_losses = list()
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(cfg.device, non_blocking=True))
                        raw_loss = ae_model.compute_loss(batch)
                        val_losses.append(raw_loss.item())
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        step_log['val_loss'] = val_loss

        # checkpoint
        if (epoch % cfg.training.checkpoint_every) == 0:
            # Log the best sevral models
            if len(best_models_val_loss) < cfg.training.save_best_num or val_loss < max(best_models_val_loss):
                torch.save(ae_model.state_dict(), checkpoint_save_path+'/checkpoint={:03d}-val_loss={:.3f}.pt'.format(epoch, val_loss))
                best_models_val_loss.append(val_loss)
            # if more than save_best_num, pop out the worst one 
            if len(best_models_val_loss) > cfg.training.save_best_num:
                worst_model_idx = best_models_val_loss.index(max(best_models_val_loss))
                best_models_val_loss.pop(worst_model_idx)
        
        # log last step
        wandb_run.log(step_log, step=global_step)
        global_step += 1
        epoch += 1
                

if __name__ == "__main__":
    main()