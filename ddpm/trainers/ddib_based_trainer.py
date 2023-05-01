from itertools import chain
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import Dict, List, Tuple, Any, Optional, Callable
import functools
import copy
import warnings 
import time
import matplotlib.pyplot as plt
import gc

import torch
import torch.cuda.amp as amp
from torch import Tensor, nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributions.categorical import Categorical
from torch.nn.utils import clip_grad_value_
from torch.nn.parallel import DistributedDataParallel

import PIL
import os

from torch.utils.tensorboard import SummaryWriter
import submitit

# from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
from ddpm.datasets import get_dataset
from ddpm.trainers.base_trainer import BaseTrainer
from ddpm.ddib_diffusion import GaussianDiffusion, SpacedDiffusion, ModelMeanType, ModelVarType, LossType, get_named_beta_schedule, space_timesteps
from ddpm.ddib_model import UNetModel
from ddpm.ddib_utils import *
from ddpm.ddib_samplers import LossAwareSampler, UniformSampler, create_named_schedule_sampler
from ddpm.fp_16_utils import MixedPrecisionTrainer
import wandb

LOG = logging.getLogger(__name__)

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source, alpha= 1 - rate)


def infiniteloop(dataloader, target: bool = False, corrupted_image : bool = False):
    if target:
        while True:
            for x, y, z in iter(dataloader):
                yield x
    elif corrupted_image:
        while True:
            for x, y in iter(dataloader):
                yield y
    else:
        while True:
            for x, y in iter(dataloader):
                yield x, y


def create_dataloader(dataset: Dataset,
                        rank: int = 0,
                        world_size: int = 1,
                        max_workers: int = 0,
                        batch_size: int = 1,
                        collate_fn: Optional[Callable] = None):

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        num_workers=max_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    return loader


class DDIB_Trainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def warmup_lr(self, step):
        return min(step, self.cfg.trainer.warmup) / self.cfg.trainer.warmup

    def setup_trainer(self) -> None:

        LOG.info(f"CityscapeTrainer: {self.cfg.trainer.rank}, gpu: {self.cfg.trainer.gpu}")
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.makedirs(os.path.join(self.cfg.trainer.logdir, 'sample'), exist_ok=True)
        self.writer = None

        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if self.cfg.trainer.rank == 0:
            if self.cfg.trainer.use_clearml:
                from clearml import Task
                task = Task.init(project_name="Cityscape_Diffusion", task_name=self.cfg.trainer.ml_exp_name)
            if self.cfg.trainer.use_wandb:
                wandb.init(project="DDIB_Training", entity=self.cfg.trainer.wandb_entity, sync_tensorboard=True)
                wandb.run.name = self.cfg.trainer.ml_exp_name
                wandb.run.save()
            self.writer = SummaryWriter(self.cfg.trainer.logdir)

        if self.cfg.trainer.gpu is not None:
            torch.cuda.set_device(self.cfg.trainer.gpu)

        if self.cfg.trainer.mean_type == "EPSILON":
            self.model_mean_type = ModelMeanType.EPSILON
        elif self.cfg.trainer.mean_type == "START_X":
            self.model_mean_type = ModelMeanType.START_X
        elif self.cfg.trainer.mean_type == "PREVIOUS_X":
            self.model_mean_type = ModelMeanType.PREVIOUS_X
        else:
            raise NotImplementedError
        
        if self.cfg.trainer.var_type == "LEARNED":
            self.model_var_type = ModelVarType.LEARNED
        elif self.cfg.trainer.var_type == "FIXED_SMALL":
            self.model_var_type = ModelVarType.FIXED_SMALL
        elif self.cfg.trainer.var_type == "FIXED_LARGE":
            self.model_var_type = ModelVarType.FIXED_LARGE
        elif self.cfg.trainer.var_type == "LEARNED_RANGE":
            self.model_var_type = ModelVarType.LEARNED_RANGE
        else:
            raise NotImplementedError
        
        if self.cfg.trainer.loss_type == "MSE":
            self.loss_type = LossType.MSE
        elif self.cfg.trainer.loss_type == "RESCALED_MSE":
            self.loss_type = LossType.RESCALED_MSE
        elif self.cfg.trainer.loss_type == "KL":
            self.loss_type = LossType.KL
        elif self.cfg.trainer.loss_type == "RESCALED_KL":
            self.loss_type = LossType.RESCALED_KL
        else:
            raise NotImplementedError

        self.learn_sigma = False if self.model_var_type in [ModelVarType.FIXED_LARGE,ModelVarType.FIXED_SMALL] else True

        self.create_model()

        # self.optimizer = torch.optim.Adam(self.net_model.parameters(), lr=self.cfg.trainer.lr)
        self.optimizer = torch.optim.AdamW(self.mp_trainer.master_params, lr=self.cfg.trainer.lr, weight_decay=self.cfg.trainer.weight_decay)
        if self.cfg.trainer.warmup > 0:
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.warmup_lr)
        else:
            self.sched = None
        self.step = 0
        self.resume_step = 0
        self.lr_anneal_steps = self.cfg.trainer.total_steps 

        self.train_dataset, self.test_dataset = get_dataset(None, self.cfg)

        LOG.info(f"train dataset length {len(self.train_dataset)}")
        dist.barrier()

        self.train_dataloader = create_dataloader(
            self.train_dataset,
            rank=self.cfg.trainer.rank,
            max_workers=self.cfg.trainer.num_workers,
            world_size=self.cfg.trainer.world_size,
            batch_size=self.cfg.trainer.batch_size,
            )
        self.datalooper = infiniteloop(self.train_dataloader)
        
        if self.test_dataset is not None:
            print("test dataset length",len(self.test_dataset))
            self.test_dataloader = create_dataloader(
                    self.test_dataset,
                    rank=self.cfg.trainer.rank,
                    max_workers=self.cfg.trainer.num_workers,
                    world_size=self.cfg.trainer.world_size,
                    batch_size=self.cfg.trainer.batch_size,
                )
        else:
            self.test_dataloader = self.train_dataloader
            self.test_dataset = self.train_dataset


        self.microbatch = self.cfg.trainer.microbatch if (self.cfg.trainer.microbatch is not None and self.cfg.trainer.microbatch > 0)  else self.cfg.trainer.batch_size

        if self.cfg.trainer.load_imagenet_256_ckpt:
            self.create_diffusion_imagenet_256()
        else:
            self.create_diffusion()

        self.schedule_sampler = create_named_schedule_sampler(self.cfg.trainer.schedule_sampler, self.diffusion)

        x_T = torch.randn(self.cfg.trainer.sample_size, self.cfg.trainer.input_channel, self.cfg.trainer.img_size, self.cfg.trainer.img_size)
        self.x_T = x_T.cuda(self.cfg.trainer.gpu)
        # show model size
        model_size = 0
        for param in self.net_model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))


    def _anneal_lr(self):
            if not self.lr_anneal_steps:
                return
            frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
            lr = self.cfg.trainer.lr * (1 - frac_done)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr


    def train(self,) -> None:
        self.net_model.train()
        self.mp_trainer.zero_grad()
        for step in range(self.cfg.trainer.total_steps):
            self.step = step + self.resume_step
            batch, _ = next(self.datalooper)
            batch = batch.cuda(self.cfg.trainer.gpu)
            number_of_accumulation = len(range(0, batch.shape[0], self.microbatch))
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch]
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], micro.device)

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.net_model,
                    micro,
                    t,)

                if last_batch:
                    losses = compute_losses()
                else:
                    with self.net_model.no_sync():
                        losses = compute_losses()
                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean() / number_of_accumulation
                if self.writer is not None:
                    if (loss.detach().cpu().item() * number_of_accumulation * 100) < 2.5:
                        self.writer.add_scalar('loss', loss.detach().cpu().item() * number_of_accumulation, step)
                    else:
                        self.writer.add_scalar('weird_loss', loss.detach().cpu().item() * number_of_accumulation * 100, step)

                self.mp_trainer.backward(loss)
            took_step = self.mp_trainer.optimize(self.optimizer)
            if took_step:
                ema(self.net_model, self.ema_model, self.cfg.trainer.ema_decay)
            else: 
                print(f"NaN at step {self.step}")
            if self.sched is not None:
                self.sched.step()
            else:
                self._anneal_lr()

            self.log(batch)
    # log
    def log(self, batch):
        if self.cfg.trainer.sample_step > 0 and self.step % self.cfg.trainer.sample_step == 0:                
            self.net_model.eval()
            self.ema_model.eval()
            if self.writer is not None:
                grid_ori = make_grid(batch) 
                img_grid_ori = wandb.Image(grid_ori.permute(1,2,0).cpu().numpy())
                wandb.log({"Original_Image": img_grid_ori}) 
            if self.step > 0 :
                with torch.no_grad():
                    x_0 = self.diffusion.p_sample_loop(self.net_model, shape = self.x_T.shape,noise=self.x_T,
                                                                clip_denoised=True,)
                    grid = make_grid(x_0)
                    path = os.path.join(
                        self.cfg.trainer.logdir, 'sample', 'ddpm_%d.png' % self.step)
                    if self.writer is not None:
                        img_grid = wandb.Image(grid.permute(1,2,0).cpu().numpy())
                        wandb.log({"Sample_DDPM": img_grid})

                with torch.no_grad():
                    x_0_ddim = self.spaced_diffusion.ddim_sample_loop(self.net_model, shape = self.x_T.shape,noise=self.x_T,
                                                                clip_denoised=True,)
                    grid_ddim = make_grid(x_0_ddim)
                    path = os.path.join(
                        self.cfg.trainer.logdir, 'sample', 'ddim_%d.png' % self.step)
                    if self.writer is not None:
                        img_grid_ddim = wandb.Image(grid_ddim.permute(1,2,0).cpu().numpy())
                        wandb.log({"Sample_DDIM": img_grid_ddim})
                
                with torch.no_grad():
                    x_0_ddim_ema = self.spaced_diffusion.ddim_sample_loop(self.ema_model, shape = self.x_T.shape,noise=self.x_T,
                                                                clip_denoised=True,)
                    grid_ddim_ema = make_grid(x_0_ddim_ema)
                    path = os.path.join(
                        self.cfg.trainer.logdir, 'sample', 'ddim_ema%d.png' % self.step)
                    if self.writer is not None:
                        img_grid_ddim_ema = wandb.Image(grid_ddim_ema.permute(1,2,0).cpu().numpy())
                        wandb.log({"Sample_DDIM_EMA": img_grid_ddim_ema})
            self.net_model.train()
            self.ema_model.train()

        # save
        if self.cfg.trainer.save_step > 0 and self.step % self.cfg.trainer.save_step == 0 and self.step > 0:
            ckpt = {
                'net_model': self.net_model.state_dict(),
                'ema_model': self.ema_model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'x_T': self.x_T,
                'step': self.step,
                "config": OmegaConf.to_container(self.cfg)
            }
            torch.save(ckpt, os.path.join(self.cfg.trainer.logdir, f'ckpt_{self.step}.pt'))


    def create_diffusion(self,):
        self.betas = get_named_beta_schedule(self.cfg.trainer.beta_schedule, self.cfg.trainer.num_timesteps)
        self.diffusion = GaussianDiffusion(betas = self.betas,
                                           model_mean_type = self.model_mean_type,
                                           model_var_type = self.model_var_type,
                                           loss_type = self.loss_type,
                                           rescale_timesteps = False)
        if self.cfg.trainer.timesteps_respacing is not None:
            timesteps_respacing = self.cfg.trainer.timesteps_respacing
        else:
            timesteps_respacing = "ddim25"

        self.spaced_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.cfg.trainer.T, timesteps_respacing),
                                betas=self.betas,
                                model_mean_type=self.model_mean_type,
                                model_var_type=self.model_var_type,
                                loss_type=self.loss_type,
                                rescale_timesteps=False,
                            )


    def create_diffusion_imagenet_256(self):
        noise_schedule = 'linear' 
        diffusion_steps = 1000
        self.betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        self.diffusion = GaussianDiffusion(betas = self.betas,
                                    model_mean_type = self.model_mean_type,
                                    model_var_type = self.model_var_type,
                                    loss_type = self.loss_type,
                                    rescale_timesteps = False)

        if self.cfg.trainer.timesteps_respacing is not None:
            timesteps_respacing = self.cfg.trainer.timesteps_respacing
        else:
            timesteps_respacing = "ddim25"

        self.spaced_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(diffusion_steps, timesteps_respacing),
                                betas=self.betas,
                                model_mean_type=ModelMeanType.EPSILON,
                                model_var_type=ModelVarType.LEARNED_RANGE,
                                loss_type=LossType.MSE,
                                rescale_timesteps=False,
                            )


    def create_model(self,):
        if self.cfg.trainer.checkpointpath is not None:
            try:            
                ckpt = torch.load(self.cfg.trainer.checkpointpath)
                ckpt_config = DictConfig(ckpt['config'])
                with open_dict(self.cfg):
                    self.cfg.trainer.ch = ckpt_config.trainer.ch
                    self.cfg.trainer.ch_mult = ckpt_config.trainer.ch_mult
                    self.cfg.trainer.attn = ckpt_config.trainer.attn
                    self.cfg.trainer.num_res_blocks = ckpt_config.trainer.num_res_blocks
                    self.cfg.trainer.dropout = ckpt_config.trainer.dropout
                    self.cfg.trainer.input_channel = ckpt_config.trainer.input_channel
                    self.cfg.trainer.kernel_size = ckpt_config.trainer.kernel_size
                    self.cfg.trainer.original_img_size = ckpt_config.trainer.original_img_size
                    self.cfg.trainer.first_crop = ckpt_config.trainer.first_crop
                    self.cfg.trainer.lower_image_size = ckpt_config.trainer.lower_image_size
                    self.cfg.trainer.img_size = ckpt_config.trainer.img_size
                self.net_model_state_dict= ckpt['net_model']
                self.ema_model_state_dict = ckpt['ema_model']
                del ckpt_config
                del ckpt
                LOG.info(f"Checkpoint Loaded.")
            except Exception as e:
                LOG.info(f"Error {e} while trying to load the checkpoint.")
        else:
            self.net_model_state_dict = None
            self.ema_model_state_dict = None

        ### Channel Multiplier based on img-size
        self.channel_mult = self.cfg.trainer.ch_mult
        self.image_size = self.cfg.trainer.img_size
        if self.channel_mult is None:
            if self.image_size == 512:
                self.channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif self.image_size == 256:
                self.channel_mult = (1, 1, 2, 2, 4, 4)
            elif self.image_size == 128:
                self.channel_mult = (1, 1, 2, 3, 4)
            elif self.image_size == 64:
                self.channel_mult = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {self.image_size}")
        else:
            channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

        attention_ds = []
        for res in self.cfg.trainer.attention_resolutions.split(","):
            attention_ds.append(self.image_size // int(res))

        if self.cfg.trainer.input_channel == 1:
            out_channels = 1
        else:
            out_channels = (3 if not self.learn_sigma else 6)

        if self.cfg.trainer.load_imagenet_256_ckpt:
            self.load_imagenet_256()

        else:
            self.net_model = UNetModel(self.image_size,
                        in_channels = self.cfg.trainer.input_channel,
                        model_channels = self.cfg.trainer.ch, #128
                        out_channels = out_channels,
                        num_res_blocks = self.cfg.trainer.num_res_blocks, #2
                        attention_resolutions = tuple(attention_ds),
                        dropout=self.cfg.trainer.dropout,
                        channel_mult=self.channel_mult,
                        conv_resample=True,
                        dims=2,
                        num_classes=None,
                        use_checkpoint=False,
                        use_fp16=self.cfg.trainer.use_fp16,
                        num_heads=self.cfg.trainer.num_heads,
                        num_head_channels=self.cfg.trainer.num_head_channels,
                        num_heads_upsample=-1,
                        use_scale_shift_norm=self.cfg.trainer.use_scale_shift_norm,
                        resblock_updown=self.cfg.trainer.resblock_updown,
                        use_new_attention_order=False)

        self.ema_model = copy.deepcopy(self.net_model)
        if self.cfg.trainer.use_fp16:
            self.ema_model.convert_to_fp16()
        if self.net_model_state_dict is not None:
            self.net_model.load_state_dict(self.net_model_state_dict)
        if self.ema_model_state_dict is not None:
            self.ema_model.load_state_dict(self.ema_model_state_dict)
        
        self.net_model = self.net_model.cuda(self.cfg.trainer.gpu)
        self.ema_model = self.ema_model.cuda(self.cfg.trainer.gpu)
        
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.net_model,
            use_fp16=self.cfg.trainer.use_fp16,
            fp16_scale_growth=self.cfg.trainer.fp16_scale_growth,
        )
        self.net_model = DistributedDataParallel(self.net_model, device_ids=[self.cfg.trainer.gpu])
        self.ema_model = DistributedDataParallel(self.ema_model, device_ids=[self.cfg.trainer.gpu])

        # show model size
        model_size = 0
        for param in self.net_model.parameters():
            model_size += param.data.nelement()
        LOG.info('Model params: %.2f M' % (model_size / 1024 / 1024))


    def load_imagenet_256(self,):
        attention_resolutions = [32,16,8]
        class_cond = False 
        diffusion_steps = 1000 
        image_size = 256 
        learn_sigma = True 
        noise_schedule = 'linear' 
        num_channels = 256 
        num_head_channels = 64 
        num_res_blocks = 2 
        resblock_updown = True 
        use_fp16 = True 
        use_scale_shift_norm = True
        input_channels = 3
        if self.cfg.trainer.input_channel == 1:
            out_channels = 1
        else:
            out_channels = (3 if not learn_sigma else 6)
        self.net_model = UNetModel(image_size,
                        in_channels = input_channels,
                        model_channels = num_channels, #128
                        out_channels = out_channels,
                        num_res_blocks = num_res_blocks, #2
                        attention_resolutions = attention_resolutions,
                        dropout=0.,
                        channel_mult=(1, 1, 2, 2, 4, 4),
                        conv_resample=True,
                        dims=2,
                        num_classes=None,
                        use_checkpoint=False,
                        use_fp16=use_fp16,
                        num_heads=-1,
                        num_head_channels=num_head_channels,
                        num_heads_upsample=-1,
                        use_scale_shift_norm=use_scale_shift_norm,
                        resblock_updown=resblock_updown,
                        use_new_attention_order=False)

        ckpt_imagenet = torch.load(self.cfg.trainer.imagenet_256_ckpt)
        self.net_model.load_state_dict(ckpt_imagenet)
        self.ema_model = copy.deepcopy(self.net_model)         
        # if use_fp16:
        #     self.net_model.convert_to_fp16()
        #     self.ema_model.convert_to_fp16()
        # self.net_model = self.net_model.cuda(self.cfg.trainer.gpu)
        # self.ema_model = self.ema_model.cuda(self.cfg.trainer.gpu)
        # self.net_model = DistributedDataParallel(self.net_model, device_ids=[self.cfg.trainer.gpu])
        # self.ema_model = DistributedDataParallel(self.ema_model, device_ids=[self.cfg.trainer.gpu])

        # show model size
        model_size = 0
        for param in self.net_model.parameters():
            model_size += param.data.nelement()
        LOG.info('Model params: %.2f M' % (model_size / 1024 / 1024))
