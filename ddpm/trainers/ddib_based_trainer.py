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
import pickle
import lpips
import random

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
import numpy as np
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
from ddpm.utils.utils import image_align, compute_psnr, compute_ssim, proc_metrics
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
                        collate_fn: Optional[Callable] = None,
                        shuffle=True
                        ):

    sampler = DistributedSampler(dataset, num_replicas=world_size,shuffle=shuffle,rank=rank)
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

        LOG.info(f"{self.cfg.trainer.name}: {self.cfg.trainer.rank}, gpu: {self.cfg.trainer.gpu}")
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.makedirs(os.path.join(self.cfg.trainer.logdir, 'sample'), exist_ok=True)
        self.writer = None

        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if self.cfg.trainer.test_session:
            if self.cfg.trainer.use_wandb:
                wandb.init(project="DDIB_Training", entity=self.cfg.trainer.wandb_entity, group="Denoising_Test" ,sync_tensorboard=True)
                wandb.run.name = self.cfg.trainer.ml_exp_name
                wandb.run.save()
            self.writer = SummaryWriter(self.cfg.trainer.logdir)
        elif self.cfg.trainer.rank == 0:
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
        self.total_batch = self.cfg.trainer.world_size * self.cfg.trainer.batch_size
         
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
        self.test_datalooper = infiniteloop(self.test_dataloader)

        self.steps_in_one_epoch = len(self.train_dataset) // self.total_batch
        LOG.info(f"Number of step per epoch: {self.steps_in_one_epoch}")
        self.microbatch = self.cfg.trainer.microbatch if (self.cfg.trainer.microbatch is not None and self.cfg.trainer.microbatch > 0)  else self.cfg.trainer.batch_size

        if self.cfg.trainer.load_imagenet_256_ckpt:
            self.create_diffusion_imagenet_256()
        else:
            self.create_diffusion()

        self.schedule_sampler = create_named_schedule_sampler(self.cfg.trainer.schedule_sampler, self.diffusion)

        x_T = torch.randn(self.cfg.trainer.sample_size, self.cfg.trainer.input_channel, self.cfg.trainer.img_size, self.cfg.trainer.img_size)
        self.x_T = x_T.cuda(self.cfg.trainer.gpu)

    def _anneal_lr(self):
            if not self.lr_anneal_steps:
                return
            frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
            lr = self.cfg.trainer.lr * (1 - frac_done)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr


    def train(self,) -> None:
        self.ddp_model.train()
        self.mp_trainer.zero_grad()
        LOG.info(f"Sample step {self.cfg.trainer.sample_step}")
        for step in range(self.cfg.trainer.total_steps):
            self.ddp_model.train()
            self.mp_trainer.zero_grad()
            self.step = step + self.resume_step
            batch, _ = next(self.datalooper)
            batch = batch.cuda(self.cfg.trainer.gpu)
            number_of_accumulation = len(range(0, batch.shape[0], self.microbatch))
            total_loss = 0
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch]
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], micro.device)

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,)

                if last_batch:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()
                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )
                
                loss = (losses["loss"] * weights).mean() / number_of_accumulation
                total_loss += loss.detach().cpu().item()

                self.mp_trainer.backward(loss)
            took_step = self.mp_trainer.optimize(self.optimizer, writer=self.writer)

            if took_step:
                ema(self.net_model, self.ema_model, self.cfg.trainer.ema_decay)
            else: 
                LOG.info(f"NaN at step {self.step}")

            self.log(batch, total_loss)
            dist.barrier()
    # log
    def log(self, batch, total_loss):    
        test_batch , _ = next(self.test_datalooper) 
        test_subbatch = test_batch[:self.x_T.shape[0]]
        test_subbatch = test_subbatch.cuda(self.cfg.trainer.gpu)
        subbatch = batch[:self.x_T.shape[0]]
        if self.cfg.trainer.sample_step > 0 and self.step % self.cfg.trainer.sample_step == 0:                
            self.ddp_model.eval()
            self.ema_model.eval()
            if self.writer is not None:
                grid_ori = make_grid(subbatch) 
                img_grid_ori = wandb.Image(grid_ori.permute(1,2,0).cpu().numpy())
                wandb.log({"Original_Image": img_grid_ori}, commit=False)
                grid_test = make_grid(test_subbatch) 
                img_grid_test = wandb.Image(grid_test.permute(1,2,0).cpu().numpy())
                wandb.log({"Test_Image": img_grid_test}, commit=False)
            
            if self.step > 0 :
                
                with torch.no_grad():
                    time_start = time.time()
                    x_0 = self.diffusion.p_sample_loop(self.ddp_model, shape = self.x_T.shape,noise=self.x_T,
                                                                clip_denoised = self.cfg.trainer.clip_denoised,
                                                                progress = True)
                    grid = make_grid(x_0)
                    path = os.path.join(
                        self.cfg.trainer.logdir, 'sample', 'ddpm_%d.png' % self.step)
                    time_sampling = time.time() - time_start
                    if self.writer is not None:
                        LOG.info("logging image")
                        self.writer.add_scalar('Sampling Time', time_sampling, self.step)
                        img_grid = wandb.Image(grid.permute(1,2,0).cpu().numpy())
                        wandb.log({"Sample_DDPM": img_grid},commit=False)
                        
            if self.step >= 0:
                with torch.no_grad():
                    x_0_ddim = self.spaced_diffusion.ddim_sample_loop(self.ddp_model, shape = self.x_T.shape,noise=self.x_T,
                                                                clip_denoised=False,
                                                                progress = self.cfg.trainer.progress)
                    grid_ddim = make_grid(x_0_ddim)
                    path = os.path.join(
                        self.cfg.trainer.logdir, 'sample', 'ddim_%d.png' % self.step)
                    if self.writer is not None:
                        img_grid_ddim = wandb.Image(grid_ddim.permute(1,2,0).cpu().numpy())
                        wandb.log({"Sample_DDIM": img_grid_ddim}, commit=False)
                
                #Inversion
                with torch.no_grad():
                    x_0_ddim_reverse = self.spaced_diffusion.ddim_reverse_sample_loop(self.ddp_model, image = x_0_ddim,
                                                                clip_denoised=True,
                                                                progress = self.cfg.trainer.progress)
                    x_0_ddim_reverse_batch = self.spaced_diffusion.ddim_reverse_sample_loop(self.ddp_model, image = subbatch,
                                                                clip_denoised=True,
                                                                progress = self.cfg.trainer.progress)

                    x_0_ddim_reverse_test_batch = self.spaced_diffusion.ddim_reverse_sample_loop(self.ddp_model, image = test_subbatch,
                                                                clip_denoised=True,
                                                                progress = self.cfg.trainer.progress)

                    # reconstruction_error_latent_batch = torch.mean((subbatch - x_0_ddim_reverse_batch)**2)                                   
                    reconstruction_error_latent = torch.mean((self.x_T - x_0_ddim_reverse)**2)

                    x_0_ddim_reconstruct = self.spaced_diffusion.ddim_sample_loop(self.ddp_model, shape = self.x_T.shape,
                                                                noise=x_0_ddim_reverse,
                                                                clip_denoised=True,
                                                                progress = self.cfg.trainer.progress)
    
                    x_0_ddim_reconstruct_batch = self.spaced_diffusion.ddim_sample_loop(self.ddp_model, shape = self.x_T.shape,
                                                                noise=x_0_ddim_reverse_batch,
                                                                clip_denoised=True,
                                                                progress = self.cfg.trainer.progress)
                    
                    x_0_ddim_reconstruct_test_batch = self.spaced_diffusion.ddim_sample_loop(self.ddp_model, shape = self.x_T.shape,
                                                                noise=x_0_ddim_reverse_test_batch,
                                                                clip_denoised=True,
                                                                progress = self.cfg.trainer.progress)


                    reconstruction_error = torch.mean((x_0_ddim- x_0_ddim_reconstruct)**2)
                    reconstruction_error_batch = torch.mean((subbatch- x_0_ddim_reconstruct_batch)**2)
                    reconstruction_error_test_batch = torch.mean((test_subbatch- x_0_ddim_reconstruct_test_batch)**2)
                    grid_reverse = make_grid(x_0_ddim_reconstruct)
                    grid_reverse_original = make_grid(x_0_ddim_reconstruct_batch)
                    grid_reverse_test= make_grid(x_0_ddim_reconstruct_test_batch)

                    #### Test Inversion

                    if self.writer is not None:
                        # img_grid_ddim_noclip = wandb.Image(grid_ddim_noclip.permute(1,2,0).cpu().numpy())
                        wandb.log({"Reconstruction_error_latent_Sampled": reconstruction_error_latent.cpu().item()},commit=False)
                        wandb.log({"Reconstruction_error_Sampled": reconstruction_error.cpu().item()},commit=False)
                        # wandb.log({"Reconstruction_error_latent_Real": reconstruction_error_latent_batch.cpu().item()},commit=False)
                        wandb.log({"Reconstruction_error_Real": reconstruction_error_batch.cpu().item()},commit=False)
                        wandb.log({"Reconstruction_error_Real_Test": reconstruction_error_test_batch.cpu().item()},commit=False)

                        img_grid_ddim_reconstruction = wandb.Image(grid_reverse.permute(1,2,0).cpu().numpy())
                        wandb.log({"Reconstruction Sample": img_grid_ddim_reconstruction},commit=False)
                        img_grid_ddim_reconstruction_batch = wandb.Image(grid_reverse_original.permute(1,2,0).cpu().numpy())
                        wandb.log({"Reconstruction Real": img_grid_ddim_reconstruction_batch},commit=False)
                        img_grid_ddim_reconstruction_test_batch = wandb.Image(grid_reverse_test.permute(1,2,0).cpu().numpy())
                        wandb.log({"Reconstruction Test": img_grid_ddim_reconstruction_test_batch},commit=False)
                        
                with torch.no_grad():
                    x_0_ddim_ema = self.spaced_diffusion.ddim_sample_loop(self.ema_model, shape = self.x_T.shape,noise=self.x_T,
                                                                clip_denoised=True,
                                                                progress = self.cfg.trainer.progress)
                    grid_ddim_ema = make_grid(x_0_ddim_ema)
                    path = os.path.join(
                        self.cfg.trainer.logdir, 'sample', 'ddim_ema%d.png' % self.step)
                    if self.writer is not None:
                        img_grid_ddim_ema = wandb.Image(grid_ddim_ema.permute(1,2,0).cpu().numpy())
                        wandb.log({"Sample_DDIM_EMA": img_grid_ddim_ema},commit=False)

            self.ddp_model.train()
            self.ema_model.train()
        if self.writer is not None:
            wandb.log({"Loss": total_loss})

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
            # torch.save(ckpt, os.path.join(directory, f'ckpt_{self.step}.pt'))
   
   
    @torch.no_grad()
    def encoding_input(self, inputs):
        self.ema_model.eval()
        while len(inputs.shape)<4:
            inputs = inputs.unsqueeze(0)
        reversing_inputs = inputs.cuda(self.cfg.trainer.gpu)
        reverse_encoding = self.ode_diffusion.ddim_reverse_sample_loop_progressive(self.ema_model,reversing_inputs,
                                                                                    clip_denoised=True, eta=0.)
        ddpm_of_reencoded_tensor = []
        with torch.no_grad():
            for dic in tqdm(reverse_encoding):
                ddpm_of_reencoded_tensor.append(dic['sample'].cpu())
        
        seed = ddpm_of_reencoded_tensor[-1].cuda(self.cfg.trainer.gpu)
        return seed
    
    
    @torch.no_grad()
    def ODEdit(self, encoded_inputs , number_of_sample=2, K=100, langevin_step=100, epsilon=1e-6, temperature = 1/3):
        self.ema_model.eval()
        seed = encoded_inputs.cuda(self.cfg.trainer.gpu).repeat_interleave(number_of_sample,dim=0)
        ddpm_imagenet_sample = self.ode_diffusion.ddim_sample_loop_progressive(self.ema_model,[number_of_sample*encoded_inputs.shape[0],3,64,64], 
                                                                                clip_denoised=True,
                                                                                noise=seed,
                                                                                eta = 0., 
                                                                                langevin=True,
                                                                                add_noise=True,
                                                                                temperature=temperature, 
                                                                                epsilon=epsilon, 
                                                                                K=K, 
                                                                                langevin_step = langevin_step, 
                                                                                clip_distance=0.)
        ddpm_of_post_resampled_tensor = []
        with torch.no_grad():
            for dic in tqdm(ddpm_imagenet_sample):
                ddpm_of_post_resampled_tensor.append(dic['sample'].cpu())
        return ddpm_of_post_resampled_tensor


    @torch.no_grad()
    def run_metrics(self,number_of_test_per_corruption = 19000,random_corruption = False, batch_size = 20,number_of_sample=1,  number_of_encoding_decoding = 1):
        self.ema_model.eval()
        # loss_fn_vgg = lpips.LPIPS(net='vgg')

        corruptions_list = ['spatter', "motion_blur", "frost","speckle_noise","impulse_noise","shot_noise","jpeg_compression","pixelate","brightness",
                            "fog","saturate","gaussian_noise",'elastic_transform','snow','masking_vline_random_color','masking_gaussian','glass_blur','gaussian_blur','contrast']
        
        corruption_severity  = 4
        K_langevin_steps = 120
        langevin_interval = 24
        epsilon_langevin = 1e-6
        
        stop_iteration_at = int(number_of_test_per_corruption/(batch_size*self.cfg.trainer.world_size))+1
        
        print("stop_iteration_at", stop_iteration_at)
        add_index_per_gpu = self.cfg.trainer.gpu*stop_iteration_at*batch_size
        print(f"On GPU {self.cfg.trainer.gpu} start index:", add_index_per_gpu)
        if random_corruption:
            directory_results = f"/mnt/scitas/bastien/ODE_xp/random_corruption/metrics_new/"
            directory_reconstruction =  f"/mnt/scitas/bastien/ODE_xp/random_corruption/reconstruction_new/"
            directory_inputs = f"/mnt/scitas/bastien/ODE_xp/random_corruption/inputs_new/"
            directory_targets = f"/mnt/scitas/bastien/ODE_xp/random_corruption/targets_new/"
            os.makedirs(directory_reconstruction, exist_ok=True)
            os.makedirs(directory_inputs, exist_ok=True)
            os.makedirs(directory_results, exist_ok=True)
            os.makedirs(directory_targets, exist_ok=True)
            for i in tqdm(range(stop_iteration_at)):
                try:
                    corruption = random.choice(corruptions_list)
                    cfg = self.cfg
                    with open_dict(cfg):
                        cfg.trainer.corruption = corruption
                        cfg.trainer.corruption_severity = corruption_severity
                        cfg.split = "test"
                    _ , test_dataset = get_dataset(None, cfg)
                    test_dataloader = create_dataloader(
                        test_dataset,
                        rank=cfg.trainer.rank,
                        max_workers=cfg.trainer.num_workers,
                        world_size=cfg.trainer.world_size,
                        batch_size=batch_size,
                        shuffle = True,
                        )
                    z = np.random.randint(len(test_dataloader))
                    for k, batch in enumerate(test_dataloader):
                        if k != z:
                            continue
                        inputs, targets = batch
                        inputs_normalized = inputs/2 +0.5
                        targets_normalized = targets/2 + 0.5
                        for n in range(len(inputs_normalized)):
                            save_image(inputs_normalized[n],directory_inputs+f"input_{i*batch_size+n+add_index_per_gpu}.png")
                            save_image(targets_normalized[n],directory_targets+f"target_{i*batch_size+n+add_index_per_gpu}.png")
                        inputs = inputs.cuda(self.cfg.trainer.gpu)
                        targets = targets.cuda(self.cfg.trainer.gpu)
                        encoded_inputs = self.encoding_input(inputs)
                        ddpm_of_post_resampled_tensor = self.ODEdit(encoded_inputs , number_of_sample=number_of_sample, K=K_langevin_steps, langevin_step=langevin_interval, 
                                                                    epsilon=epsilon_langevin, temperature = 1/2)
                        results = ddpm_of_post_resampled_tensor[-1]
                        results_normalized = results/2 + 0.5
                        if number_of_sample > 1:
                            results_per_image = results_normalized.split(number_of_sample, dim=0)
                            for n, same_images in enumerate(results_per_image):
                                for image_i in range(len(same_images)):
                                    save_image(same_images[image_i], directory_reconstruction+f"reconstruction_{i*batch_size+n+add_index_per_gpu}_{image_i}.png")
                        else:
                            results_per_image = results_normalized
                            for n, same_images in enumerate(results_per_image):
                                save_image(same_images, directory_reconstruction+f"reconstruction_{i*batch_size+n+add_index_per_gpu}_{0}.png")
                        break
                except Exception as e:
                    print(f"Error in step {i}, gpu {self.cfg.trainer.gpu}")
                    print(e)
                    continue
        else:
            for corruption in corruptions_list:
                if True:
                # for corruption_severity in corruptions_severities: 
                    directory_results = f"/mnt/scitas/bastien/ODE_xp/severity_{corruption_severity}/{corruption}/metrics/"
                    directory_reconstruction =  f"/mnt/scitas/bastien/ODE_xp/severity_{corruption_severity}/{corruption}/reconstruction/"
                    directory_inputs = f"/mnt/scitas/bastien/ODE_xp/severity_{corruption_severity}/{corruption}/inputs/"
                    directory_targets = f"/mnt/scitas/bastien/ODE_xp/severity_{corruption_severity}/{corruption}/targets/"
                    os.makedirs(directory_reconstruction, exist_ok=True)
                    os.makedirs(directory_inputs, exist_ok=True)
                    os.makedirs(directory_results, exist_ok=True)
                    os.makedirs(directory_targets, exist_ok=True)
                    cfg = self.cfg
                    with open_dict(cfg):
                        cfg.trainer.corruption = corruption
                        cfg.trainer.corruption_severity = corruption_severity
                        cfg.split = "test"
                    _ , test_dataset = get_dataset(None, cfg)
                    LOG.info(f"train dataset length {len(self.test_dataset)}")
                    test_dataloader = create_dataloader(
                        test_dataset,
                        rank=cfg.trainer.rank,
                        max_workers=cfg.trainer.num_workers,
                        world_size=cfg.trainer.world_size,
                        batch_size=batch_size,
                        shuffle = False,
                        )
                    if stop_iteration_at >= len(test_dataloader):
                        stop_iteration_at = len(test_dataloader) -1
                    for i, batch in enumerate(test_dataloader):
                        if i >= stop_iteration_at:
                            break
                        inputs, targets = batch
                        inputs_normalized = inputs/2 +0.5
                        targets_normalized = targets/2 + 0.5
                        for n in range(len(inputs_normalized)):
                            save_image(inputs_normalized[n],directory_inputs+f"input_{i*batch_size+n+add_index_per_gpu}.png")
                            save_image(targets_normalized[n],directory_targets+f"target_{i*batch_size+n+add_index_per_gpu}.png")
                        inputs = inputs.cuda(self.cfg.trainer.gpu)
                        targets = targets.cuda(self.cfg.trainer.gpu)
                        encoded_inputs = self.encoding_input(inputs)
                        ddpm_of_post_resampled_tensor = self.ODEdit(encoded_inputs , number_of_sample=number_of_sample, K=K_langevin_steps, 
                                                                    langevin_step=langevin_interval, epsilon=epsilon_langevin, temperature = 1/3)
                        results = ddpm_of_post_resampled_tensor[-1]
                        results_normalized = results/2 + 0.5
                        results_per_image = results_normalized.split(number_of_sample, dim=0)
                        if number_of_sample > 1:
                            results_per_image = results_normalized.split(number_of_sample, dim=0)
                            for n, same_images in enumerate(results_per_image):
                                for image_i in range(len(same_images)):
                                    save_image(same_images[image_i], directory_reconstruction+f"reconstruction_{i*batch_size+n+add_index_per_gpu}_{image_i}.png")
                        else:
                            results_per_image = results_normalized
                            for n, same_images in enumerate(results_per_image):
                                save_image(same_images, directory_reconstruction+f"reconstruction_{i*batch_size+n+add_index_per_gpu}_{0}.png")
                        # for n, same_images in enumerate(results_per_image):
                        #     for image_i in range(len(same_images)):
                        #         save_image(same_images[image_i], directory_reconstruction+f"reconstruction_{i*batch_size+n+add_index_per_gpu}_{image_i}.png")
                        if self.cfg.trainer.gpu == 0 and i % 1000 == 0:
                            grid_targets = make_grid(targets_normalized)
                            grid_source = make_grid(inputs_normalized)
                            grid_results= make_grid(results)
                            img_grid_ddim_reconstruction = wandb.Image(grid_targets.permute(1,2,0).cpu().numpy())
                            wandb.log({"Original Sample": img_grid_ddim_reconstruction},commit=False)
                            img_grid_ddim_reconstruction_batch = wandb.Image(grid_source.permute(1,2,0).cpu().numpy())
                            wandb.log({"Source Sample": img_grid_ddim_reconstruction_batch},commit=False)
                            img_grid_ddim_reconstruction_test_batch = wandb.Image(grid_results.permute(1,2,0).cpu().numpy())
                            wandb.log({"Reconstruction Results": img_grid_ddim_reconstruction_test_batch},commit=True)


    def log_and_save_images(self, image_batch,corruption="",corruption_severity=-1, K=0, epsilon = -1,step = 0, loop = -1, gpu = 0, commit = True):
        image_batch = ((image_batch/2)+0.5)
        if K==0:
            title = os.path.join(corruption, f"Reconstruction_severity_{corruption_severity}_loop{loop}")
        else:
            title = os.path.join(corruption, f"severity_{corruption_severity}_K_{K}_Epsilon_{epsilon}_loop_{loop}_{step}")
        path_string = os.path.join(self.cfg.trainer.logdir, corruption, f"severity_{corruption_severity}", f"K_{K}", f"Epsilon_{epsilon}")
        p = Path(path_string).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        path = os.path.join(p, f"Sample_{gpu}_loop_{loop}.png")
        grid = make_grid(image_batch.cpu().detach())
        save_image(grid, path)
        img_grid = wandb.Image(grid.permute(1,2,0).numpy())
        wandb.log({title: img_grid},commit=commit)
        

    @torch.no_grad()
    def test_and_output(self,number_of_batch = 1, number_of_encoding_decoding = 3):
        # if self.cfg.trainer.gpu != 0:
        #     return
        self.ddp_model.eval()

        corruptions_list = ['glass_blur',"zoom_blur",'defocus_blur','fog','gaussian_blur',
                            'pixelate',"jpeg_compression","impulse_noise","speckle_noise","gaussian_noise",
                            'brightness','contrast','spatter',"saturate","shot_noise",
                            'elastic_transform','snow','masking_color_lines','masking_random_color',
                            'masking_gaussian','masking_hline_random_color','masking_simple','masking_line','masking_vline_random_color',]
        corruptions_severities  = [3,5]
        K_langevin_steps = [1,10,20,50]
        epsilon_langevin = [2e-6,5e-6,1e-5,2e-5,4e-5,6e-5,8e-5,1e-4]
        for corruption in corruptions_list:
            for corruption_severity in corruptions_severities:
                cfg = self.cfg
                with open_dict(cfg):
                    cfg.trainer.corruption = corruption
                    cfg.trainer.corruption_severity = corruption_severity
                    cfg.split = "test"
                self.train_dataset, self.test_dataset = get_dataset(None, cfg)
                LOG.info(f"train dataset length {len(self.test_dataset)}")

                self.test_dataloader = create_dataloader(
                    self.test_dataset,
                    rank=cfg.trainer.rank,
                    max_workers=cfg.trainer.num_workers,
                    world_size=cfg.trainer.world_size,
                    batch_size=cfg.trainer.batch_size,
                    )
                self.datalooper = infiniteloop(self.test_dataloader)
                for step in range(number_of_batch):
                    batch, original_target = next(self.datalooper)
                    original_target = ((original_target/2)+0.5)
                    grid_original_target = make_grid(original_target.cpu().detach())
                    img_grid_original_target = wandb.Image(grid_original_target.permute(1,2,0).numpy())
                    original_input = ((batch/2)+0.5)
                    grid_original_input = make_grid(original_input.cpu().detach())
                    img_grid_original_input = wandb.Image(grid_original_input.permute(1,2,0).numpy())
                    wandb.log({f"Target_{step}": img_grid_original_target},commit=True)
                    wandb.log({f"Input_{step}": img_grid_original_input},commit=True)
                    psnr = 0
                    ssim = 0
                    for i in range(batch.shape[0]):
                        original_psnr, original_ssim = proc_metrics(original_target[i].permute(1,2,0).numpy(), original_input[i].permute(1,2,0).numpy())
                        psnr += original_psnr / batch.shape[0]
                        ssim += original_ssim / batch.shape[0]
                    wandb.log({f"Original PSNR": psnr},commit=True)
                    wandb.log({f"Original_SSIM": ssim},commit=True)
                    batch = batch.cuda(cfg.trainer.gpu)
                    inputs = batch   
                    ### ENCODING - DECODING  
                    print("Reverse Encoding")   
                    reverse_encoding = self.spaced_diffusion.ddim_reverse_sample_loop_progressive(self.ddp_model,inputs ,
                                                                                    clip_denoised=True, eta=0.)
                    ddpm_of_reencoded_tensor = []
                    # ddpm_of_reencoded_attentions = []
                    with torch.no_grad():
                        for dic in tqdm(reverse_encoding):
                            ddpm_of_reencoded_tensor.append(dic['sample'].cpu())
                            # for key in dic['attention_maps']:
                            #     list_cpu = [layer.cpu() for layer in dic['attention_maps'][key]]
                            #     dic['attention_maps'][key] = list_cpu
                            # ddpm_of_reencoded_attentions.append(dic['attention_maps'])
                    
                    ddim_sample = self.spaced_diffusion.ddim_sample_loop_progressive(self.ddp_model,inputs.shape, 
                                                                    clip_denoised=True,noise=ddpm_of_reencoded_tensor[-1].cuda(cfg.trainer.gpu))
                    ddpm_of_resampled_tensor = []
                    # ddpm_of_resampled_attentions = []
                    with torch.no_grad():
                        for dic in tqdm(ddim_sample):
                            ddpm_of_resampled_tensor.append(dic['sample'].cpu())
                            # ddpm_of_resampled_attentions.append(dic['attention_maps'])
                    print("Saving Images")   
                    self.log_and_save_images(ddpm_of_resampled_tensor[-1],
                                            corruption=corruption,
                                            corruption_severity=corruption_severity,
                                            K=0, 
                                            epsilon=0,
                                            gpu = cfg.trainer.gpu, 
                                            step=step,
                                            loop=-1)
                    reconstructed_img = ((ddpm_of_resampled_tensor[-1]/2)+0.5)
                    psnr = 0
                    ssim = 0
                    for i in range(batch.shape[0]):
                        original_psnr, original_ssim = proc_metrics(original_target[i].permute(1,2,0).numpy(), reconstructed_img[i].permute(1,2,0).numpy())
                        psnr += original_psnr / batch.shape[0]
                        ssim += original_ssim / batch.shape[0]
                    wandb.log({f"Post_inversion PSNR": psnr},commit=True)
                    wandb.log({f"Post_inversion SSIM": ssim},commit=True)
                    for K in K_langevin_steps:
                        wandb.define_metric("epsilon_step")
                        # define which metrics will be plotted against it
                        for h in range(number_of_encoding_decoding):
                            wandb.define_metric(f"psnr_severity_{corruption_severity}_K_{K}_loop_{h}", step_metric="epsilon_step")
                            wandb.define_metric(f"ssim_severity_{corruption_severity}_K_{K}_loop_{h}", step_metric="epsilon_step")
                        for epsilon in epsilon_langevin:
                            latent = ddpm_of_reencoded_tensor[-1].cuda(cfg.trainer.gpu)
                            if epsilon * K > 1e-3 - 0.000001:
                                continue
                            for j in range(number_of_encoding_decoding):
                                print("Langevin Started")
                                ddim_sample = self.spaced_diffusion.ddim_sample_loop_progressive(self.ddp_model,inputs.shape, clip_denoised=True,noise=latent,
                                                                                            eta = 0., langevin=True, epsilon = epsilon, K=K)
                                ddpm_of_post_resampled_tensor = []
                                # ddpm_of_post_resampled_attentions = []
                                with torch.no_grad():
                                    for dic in tqdm(ddim_sample):
                                        ddpm_of_post_resampled_tensor.append(dic['sample'].cpu())
                                        # ddpm_of_post_resampled_attentions.append(dic['attention_maps'])
                                inputs = ddpm_of_post_resampled_tensor[-1].cuda(cfg.trainer.gpu)
                                self.log_and_save_images(inputs,
                                            corruption=corruption,
                                            corruption_severity=corruption_severity,
                                            K=K, 
                                            epsilon=epsilon,
                                            gpu = cfg.trainer.gpu,
                                            step=step, 
                                            loop=j)
                                reconstructed_img = ddpm_of_post_resampled_tensor[-1]/2 +0.5
                                psnr = 0
                                ssim = 0
                                for i in range(batch.shape[0]):
                                    original_psnr, original_ssim = proc_metrics(original_target[i].permute(1,2,0).numpy(), reconstructed_img[i].permute(1,2,0).numpy())
                                    psnr += original_psnr / batch.shape[0]
                                    ssim += original_ssim / batch.shape[0]
                                log_dict = {
                                    f"psnr_severity_{corruption_severity}_K_{K}_loop_{j}": psnr,
                                    f"ssim_severity_{corruption_severity}_K_{K}_loop_{j}": ssim,
                                    "epsilon_step": epsilon,  
                                    }
                                wandb.log(log_dict)
                                if j < number_of_encoding_decoding -1:
                                    reverse_new_encoding = self.spaced_diffusion.ddim_reverse_sample_loop_progressive(self.ddp_model,inputs ,
                                                                                clip_denoised=True, eta=0.)
                                    ddpm_of_new_reencoded_tensor = []
                                    # ddpm_of_new_reencoded_attentions = []
                                    with torch.no_grad():
                                        for dic in tqdm(reverse_new_encoding):
                                            ddpm_of_new_reencoded_tensor.append(dic['sample'].cpu())
                                            # for key in dic['attention_maps']:
                                                # list_cpu = [layer.cpu() for layer in dic['attention_maps'][key]]
                                                # dic['attention_maps'][key] = list_cpu
                                            # ddpm_of_new_reencoded_attentions.append(dic['attention_maps'])
                                    latent = ddpm_of_new_reencoded_tensor[-1].cuda(cfg.trainer.gpu)


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

        self.spaced_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.cfg.trainer.num_timesteps, timesteps_respacing),
                                betas=self.betas,
                                model_mean_type=self.model_mean_type,
                                model_var_type=self.model_var_type,
                                loss_type=self.loss_type,
                                rescale_timesteps=False,
                            )
        self.ode_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.cfg.trainer.num_timesteps, "ddim100"),
                                betas=self.betas,
                                model_mean_type=self.model_mean_type,
                                model_var_type=self.model_var_type,
                                loss_type=self.loss_type,
                                rescale_timesteps=False,)


    def create_diffusion_imagenet_256(self):
        noise_schedule = self.cfg.trainer.beta_schedule 
        diffusion_steps = 1000
        self.betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        self.diffusion = GaussianDiffusion(betas = self.betas,
                                    model_mean_type = self.model_mean_type,
                                    model_var_type = ModelVarType.LEARNED_RANGE,
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
                                loss_type=LossType.RESCALED_MSE,
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
                    self.cfg.trainer.attention_resolutions = ckpt_config.trainer.attention_resolutions
                    self.cfg.trainer.num_res_blocks = ckpt_config.trainer.num_res_blocks
                    self.cfg.trainer.dropout = ckpt_config.trainer.dropout
                    self.cfg.trainer.input_channel = ckpt_config.trainer.input_channel
                    self.cfg.trainer.kernel_size = ckpt_config.trainer.kernel_size
                    self.cfg.trainer.original_img_size = ckpt_config.trainer.original_img_size
                    self.cfg.trainer.lower_image_size = ckpt_config.trainer.lower_image_size
                    self.cfg.trainer.img_size = ckpt_config.trainer.img_size
                self.net_model_state_dict= copy.deepcopy(ckpt['net_model'])
                self.ema_model_state_dict = copy.deepcopy(ckpt['ema_model'])
                del ckpt_config
                del ckpt
                LOG.info(f"Checkpoint Loaded.")
                print(f"Checkpoint Loaded.")
            except Exception as e:
                LOG.info(f"Error {e} while trying to load the checkpoint.")
                print(f"Error {e} while trying to load the checkpoint.")
                self.net_model_state_dict = None
                self.ema_model_state_dict = None
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
                self.channel_mult = (1, 2, 2, 4, 8) #(1, 1, 2, 3, 4)
            elif self.image_size == 64:
                self.channel_mult = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {self.image_size}")
        else:
            self.channel_mult = tuple(int(ch_mult) for ch_mult in self.channel_mult.split(","))

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
        self.ddp_model = DistributedDataParallel(self.net_model, device_ids=[self.cfg.trainer.gpu])

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
        noise_schedule = self.cfg.trainer.beta_schedule 
        num_channels = 256 
        num_head_channels = 64 
        num_res_blocks = 2 
        resblock_updown = True 
        use_fp16 = self.cfg.trainer.use_fp16 
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


