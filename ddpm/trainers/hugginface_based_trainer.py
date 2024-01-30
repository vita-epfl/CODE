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
from ddpm.sde.sdelib import load_model, SDEditing
import wandb


import matplotlib.pyplot as plt
import PIL
import diffusers
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import Resize, ToTensor, ToPILImage
import importlib

from tqdm import trange
import yaml
import os
import numbers


from ddpm.datasets.celeba import CelebA 
from ddpm.ddib_diffusion import GaussianDiffusion, SpacedDiffusion, _extract_into_tensor, space_timesteps, LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
from ddpm.datasets.corruptions import *

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
                wandb.init(project="ODEditing", entity=self.cfg.trainer.wandb_entity, group="Editing" ,sync_tensorboard=True)
                wandb.run.name = self.cfg.trainer.ml_exp_name
                wandb.run.save()
            self.writer = SummaryWriter(self.cfg.trainer.logdir)
        elif self.cfg.trainer.rank == 0:
            if self.cfg.trainer.use_clearml:
                from clearml import Task
                task = Task.init(project_name="ODEditing", task_name=self.cfg.trainer.ml_exp_name)
            if self.cfg.trainer.use_wandb:
                wandb.init(project="ODEditing", entity=self.cfg.trainer.wandb_entity, sync_tensorboard=True)
                wandb.run.name = self.cfg.trainer.ml_exp_name
                wandb.run.save()
        
            self.writer = SummaryWriter(self.cfg.trainer.logdir)

        if self.cfg.trainer.gpu is not None:
            torch.cuda.set_device(self.cfg.trainer.gpu)

        # Get the pipeline, model and its parameters
        self.model_id = self.cfg.trainer.model_id
        self.ddpm = DDPMPipeline.from_pretrained(self.model_id)  
        self.model = self.ddpm.unet
        self.num_timesteps = int(ddpm.scheduler.betas.shape[0])
        self.betas = self.ddpm.scheduler.betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

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
        

        x_T = torch.randn(self.cfg.trainer.sample_size, self.cfg.trainer.input_channel, self.cfg.trainer.img_size, self.cfg.trainer.img_size)
        self.x_T = x_T.cuda(self.cfg.trainer.gpu)



    def train(self,) -> None:
        LOG.info("Huggingface models are pretrained, no need to train them")
        return 
        
    def log(self,) -> None:
        """
        Log inputs/outputs in clearml or wandb or tensorboard
        Log process 
        """
    def _threshold_sample(sample: torch.FloatTensor, dynamic_thresholding_ratio: float, sample_max_value : float = 5/3) -> torch.FloatTensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample  


    @torch.no_grad()
    def langevin_sampling(inputs, model, t, t_prev, alphas_cumprod,alphas = alphas, steps = 100, epsilon = 1e-5,min_variance = -1, 
                        device = device, denoising_step = True, clip_prev = False, clip_now = False, dynamic_thresholding = False,  power  = 0.5):
        timestep = t
        t = torch.tensor([t] * inputs.shape[0], device=device)  
        mean_coef_t = torch.sqrt(_extract_into_tensor(alphas_cumprod, t , inputs.shape))
        variance = _extract_into_tensor(1.0 - alphas_cumprod, t , inputs.shape)
        alphas = alphas.to(device)
        std = torch.sqrt(variance)
        std_epsilon = []
        mean_epsilon = []
        if t_prev is not None:
            t_prev = torch.tensor([t_prev] * inputs.shape[0], device=device)        
            mean_coef_t_prev = torch.sqrt(_extract_into_tensor(alphas_cumprod, t_prev , inputs.shape))
            variance_t_prev = _extract_into_tensor(1.0 - alphas_cumprod, t_prev , inputs.shape)
            std_prev = torch.sqrt(variance_t_prev)
            noise_estimate_t_prev = model(inputs, t_prev)['sample']
            x0_t_1 = (inputs - std_prev * noise_estimate_t_prev)/mean_coef_t_prev
            if dynamic_thresholding:
                x0_t_1 = _threshold_sample(x0_t_1, self.dynamic_threshol_ratio, self.dynamic_threshold_max)
            if clip_prev:
                x0_t_1 = x0_t_1.clamp(-1,1)
            inputs = inputs + x0_t_1 * (mean_coef_t - mean_coef_t_prev)
        inputs = inputs.to(device)
        model = model.to(device)
        if min_variance > 0:
            alpha_coef =  (variance/min_variance) * epsilon
        else:
            alpha_coef = torch.ones_like(variance) * epsilon
        with torch.no_grad():
            for i in range(steps):
                noise_estimate = model(inputs, t)['sample']
                if dynamic_thresholding:
                    x0_t = (inputs - std * noise_estimate)/mean_coef_t
                    x0_t = _threshold_sample(x0_t, self.dynamic_threshol_ratio, self.dynamic_threshold_max)
                    noise_estimate = (inputs - mean_coef_t * x0_t) / std
                elif clip_now:
                    x0_t = (inputs - std * noise_estimate)/mean_coef_t
                    x0_t = x0_t.clamp(-1,1)
                    noise_estimate = (inputs - mean_coef_t * x0_t) / std
                std_epsilon.append(noise_estimate[0].cpu().std().item())
                mean_epsilon.append(noise_estimate[0].cpu().mean().item())
                score = - noise_estimate / std
                noise = torch.randn(inputs.shape).to(device)
                if steps > 1:
                    inputs = (inputs + alpha_coef * score) + torch.pow(2*alpha_coef, power) * noise
            if denoising_step:
                noise_estimate = model(inputs, t)['sample']
                score = - noise_estimate / std
                inputs = (inputs + alpha_coef * score) 
        return inputs, alpha_coef, std_epsilon, mean_epsilon


    @torch.no_grad()
    def ddim_step(self, inputs, t, clip_denoised = False, dynamic_thresholding = True, clip_value = 1, sigma = 0, 
                forward = True, number_of_sample = 1):
        alphas_cumprod = self.alphas_cumprod
        number_of_timesteps = self.ddpm.scheduler.betas.shape[0]
        model = self.model.cuda(self.cfg.trainer.gpu)
        if t > 0 and forward:
            t_prev = torch.tensor([t-1] * inputs.shape[0]).cuda(self.cfg.trainer.gpu)
            variance_prev = _extract_into_tensor(1.0 - alphas_cumprod, t_prev , inputs.shape)
            std_prev = torch.sqrt(variance_prev)
        elif t < number_of_timesteps :
            t_prev = torch.tensor([t+1] * inputs.shape[0]).cuda(self.cfg.trainer.gpu)
            variance_prev = _extract_into_tensor(1.0 - alphas_cumprod, t_prev , inputs.shape)
            std_prev = torch.sqrt(variance_prev)
        t = torch.tensor([t] * inputs.shape[0]).cuda(self.cfg.trainer.gpu)
        variance = _extract_into_tensor(1.0 - alphas_cumprod, t , inputs.shape)
        mean_coef_t = torch.sqrt(_extract_into_tensor(alphas_cumprod, t , inputs.shape))
        std = torch.sqrt(variance)
        noise_estimate = model(inputs, t)['sample']
        std_epsilon = noise_estimate[0].cpu().std().item()
        mean_epsilon = noise_estimate[0].cpu().mean().item()
        if dynamic_thresholding:
            x0_t = (inputs - std * noise_estimate)/mean_coef_t
            x0_t = _threshold_sample(x0_t, self.dynamic_threshol_ratio, self.dynamic_threshold_max)
            noise_estimate = (inputs - mean_coef_t * x0_t) / std
            std_epsilon = noise_estimate[0].cpu().std().item()
            mean_epsilon = noise_estimate[0].cpu().mean().item()
        elif clip_denoised:
            x0_t = (inputs - std * noise_estimate)/mean_coef_t
            x0_t = x0_t.clamp(-1,1)
            noise_estimate = (inputs - mean_coef_t * x0_t) / std
            std_epsilon = noise_estimate[0].cpu().std().item()
            mean_epsilon = noise_estimate[0].cpu().mean().item()
        if sigma > 0:
            sigma_t = sigma * torch.sqrt((variance_prev)/(variance)) * torch.sqrt(1 - (1-variance)/ (1-variance_prev))
        else:
            sigma_t = 0.
        noise= torch.randn_like(inputs)
        x_prev = torch.sqrt(1-variance_prev) / torch.sqrt(1-variance) * ((inputs - std * noise_estimate)) + torch.sqrt(variance_prev - sigma_t**2)  * noise_estimate + sigma_t * noise
        return x_prev, std_epsilon, mean_epsilon

    def corrupt(number = 9999,corruption = 'spatter', random_sampling = False, random_corruption = False):
        corruptions_functions = {"shot_noise" : shot_noise, "gaussian_blur" : gaussian_blur, "spatter" : spatter, "fog":fog, "frost":frost, 
                            "snow":snow, "glass_blur":glass_blur, "elastic_transform":elastic_transform, "contrast":contrast, "brightness":brightness,
                            "gaussian_noise":gaussian_noise, "impulse_noise":impulse_noise, "masking_random_color_random":masking_random_color_random,
                            "motion_blur":motion_blur,"jpeg_compression":jpeg_compression, "pixelate":pixelate}
        if random_corruption:         
            corruption = random.choice(list(corruptions_functions.keys()))
        if random_sampling:
            number = np.random.randint(1,29999)
            img_path = f"/mnt/scitas/bastien/CelebAMask-HQ/CelebA-HQ-img/{number}.jpg"
            img_pil = PIL.Image.open(img_path).resize([256,256])

        else:
            img_path = f"/mnt/scitas/bastien/CelebAMask-HQ/CelebA-HQ-img/{number}.jpg"
            img_pil = PIL.Image.open(img_path).resize([256,256])
        corrupted_sample = PIL.Image.fromarray(corruptions_functions[corruption](img_pil, severity  = 5).astype(np.uint8))
        img_tensor = (torch.from_numpy(np.array(corrupted_sample)).permute(2,0,1)/255)*2-1
        original = (torch.from_numpy(np.array(img_pil)).permute(2,0,1)/255)*2-1
        
        return img_tensor, original

def editing_with_ode(latent_codes, model, t_start = 1000, std_div = 0.05, annealing = False, epsilon = 1e-8, steps = 20, power =0.5, 
                     min_variance = -1. , number_of_sample = 1,normalize=False, alphas_cumprod = alphas_cumprod, 
                     corrector_step = True,deep_correction = False, comparison = False):
    list_of_evolution_reverse = []
    final_samples = []
    t_valid = list(range(0, min(1000,t_start)))
    std_recon = 1
    
    with torch.no_grad():
        if normalize:
            inputs = (latent_codes[np.max(t_valid)] - latent_codes[np.max(t_valid)].mean())/latent_codes[np.max(t_valid)].std()
            batch_size = inputs.shape[0]
        else:
            inputs = latent_codes[np.max(t_valid)]
            batch_size = inputs.shape[0]
            inputs = inputs.split(len(inputs)-1, dim=0)
            inputs = [input.repeat(number_of_sample, 1, 1, 1) for input in inputs]
            inputs = torch.cat(inputs)
        edited_to_be_done = True
        if comparison:
            run_with_correction = [True,False]
        else:
            run_with_correction = [corrector_step]
        for corrector in run_with_correction:
            for t in range(1,np.max(t_valid)+1)[::-1]:
                alpha_std = alphas[t]
                counter = 0
                if std_div > 0:
                    if t>100 and t < t_start-1:
                        while std_recon < (1-std_div) or std_recon > (1+std_div):
                            inputs,alpha_coef, list_of_stds, list_of_means = langevin_sampling(inputs, model, t, None, alphas_cumprod, steps = steps, epsilon = epsilon,
                                                   min_variance = min_variance, clip_prev = False, clip_now = False,dynamic_thresholding=True, device = device, power = power)
        
                            std_recon = list_of_stds[-1]
                            counter += 1
                            if counter > 100:
                                epsilon = 2*epsilon
                            for h in range(len(inputs)):
                                list_of_evolution_reverse.append(inputs[h].cpu())
                        epsilon *= 1
                        
                elif t == t_start-1 and edited_to_be_done:
                    before = inputs.cpu()
                    if annealing>=1:
                        for j in range(int(annealing)):
                            step_per_epsilon = steps // len(range(int(annealing)))
                            inputs,alpha_coef, list_of_stds, list_of_means = langevin_sampling(inputs, model, t, None, alphas_cumprod, steps = step_per_epsilon, epsilon = epsilon,
                                                   min_variance = min_variance, clip_prev = False, clip_now = False,dynamic_thresholding=True, device = device, power = power)
                            epsilon  = epsilon / 2
                            plt.figure(figsize=[10,10])
                            plt.imshow(make_grid(torch.cat([before.squeeze(), inputs.squeeze().cpu()])).permute(1,2,0)/2+0.5)
                            plt.show()
                    else:
                        inputs,alpha_coef, list_of_stds, list_of_means = langevin_sampling(inputs, model, t, None, alphas_cumprod, steps = steps, epsilon = epsilon,
                                min_variance = min_variance, clip_prev = False, clip_now = False,dynamic_thresholding=True, device = device, power = power)  
                    std_recon = list_of_stds[-1]
                    for h in range(len(inputs)):
                            list_of_evolution_reverse.append(inputs[h].cpu())
                    edited_latents = inputs
                    edited_to_be_done = False
                elif t == t_start-1:
                    inputs = edited_latents
                if corrector:
                    if deep_correction:
                        if t < 80 and t % 19 == 0:           
                            inputs,alpha_coef, list_of_stds, list_of_means = langevin_sampling(inputs, model, t, t, alphas_cumprod, 
                                                                                steps = 100, epsilon = 5e-4,
                                                                                min_variance = min_variance, clip_prev = False,
                                                                                clip_now = False,dynamic_thresholding=True, device = device, 
                                                                                power = power)
    
                            std_recon = list_of_stds[-1]
                    else:
                            inputs,alpha_coef, list_of_stds, list_of_means = langevin_sampling(inputs, model, t, t, alphas_cumprod, 
                                                                                steps = 1, epsilon = 1e-7,
                                                                                min_variance = min_variance, clip_prev = False,
                                                                                clip_now = False,dynamic_thresholding=True, device = device, 
                                                                                power = power)
    
                            std_recon = list_of_stds[-1]
                inputs, std_epsilon, mean_epsilon = ddim_step(inputs, model, t, alphas_cumprod, sigma = 0.,
                                                          clip_denoised=True,dynamic_thresholding=True, 
                                                          device = device, forward=True, number_of_sample = number_of_sample)               
                std_recon = std_epsilon
                for h in range(len(inputs)):
                    list_of_evolution_reverse.append(inputs[h].cpu())
            for sample in list_of_evolution_reverse[-batch_size*number_of_sample:]:
                final_samples.append(sample)
        
        for h in range(len(inputs)):
            list_of_evolution_reverse.append(inputs[h].cpu())
    return list_of_evolution_reverse, final_samples


    @torch.no_grad()
    def encode_inputs(self, inputs) 
        latent_codes = []
        with torch.no_grad():
            latent_codes.append(inputs)
            for t in range(0,self.ddpm.scheduler.betas.shape[0]-1):
                inputs, std_eps, mean_eps = self.ddim_step(inputs, self.ddpm.unet, t, self.alphas_cumprod, sigma = 0.,
                                            clip_denoised=True,dynamic_thresholding = True, forward=False)
                latent_codes.append(inputs)
                list_std_encoding.append(std_eps)
                list_mean_encoding.append(mean_eps)
        return latent_codes, list_mean_encoding, list_std_encoding

    @torch.no_grad()
    def old_encoding_input(self, inputs):
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
    
    # store reconstruction par ode/sde/original/corrupted,latent/noise/img_number/samples
    @torch.no_grad()
    def run_experiments(self,number_of_image = 1000, corruptions = 'all', sde_range = [100,1001,100], 
                            ode_range = [100, 1001, 100], number_of_sample = 4, celebaHQ = True)
        if corruptions == 'all':
            corruptions_list = ['spatter', "motion_blur", "frost","speckle_noise","impulse_noise","shot_noise","jpeg_compression","pixelate","brightness",
                            "fog","saturate","gaussian_noise",'elastic_transform','snow','masking_vline_random_color','masking_gaussian','glass_blur','gaussian_blur','contrast']
        else:
            corruptions_list = [corruptions]
        if celebaHQ:
            all_numbers = range(len(self.train_dataset))
            images_numbers = np.random.choice(all_numbers, number_of_image)
        ### get subset of dataset based on the indices

        subset_dataset = Subset(self.train_dataset, list(images_numbers))
        original_indices = list(range(len(subset_dataset)))
        remaining_indices = original_indices
        directory_clean_original = f"/mnt/scitas/bastien/ODEDIT/original"
        directory_corrupted = f"/mnt/scitas/bastien/ODEDIT/corrupted"
        directory_reconstruction_sde =  f"/mnt/scitas/bastien/ODEDIT/sde"
        directory_reconstruction_ode =  f"/mnt/scitas/bastien/ODEDIT/ode"
        directory_latent = f"/mnt/scitas/bastien/ODEDIT/latent"
        os.makedirs(directory_clean_original, exist_ok=True)
        os.makedirs(directory_corrupted, exist_ok=True)
        os.makedirs(directory_reconstruction_sde, exist_ok=True)
        os.makedirs(directory_reconstruction_ode, exist_ok=True)
        
        dataloader = DataLoader(subset_dataset, batch_size=self.cfg.trainer.batch_size, sampler = SequentialSampler()
        epsilons = np.np.geomspace(1e-6,1e-2, 10)
        list_steps = [1000]
        ### IF different per latent
        dictionnary_epsilon = {100:[1e-5, 1e-6], 200:[1e-5, 1e-6], 300:[1e-5, 1e-6],400:[1e-5, 1e-6],500:[1e-5, 1e-6],
                                600:[1e-5, 1e-6],700:[1e-5, 1e-6], 800:[1e-5, 1e-6], 900:[1e-5, 1e-6], 1000:[1e-5, 1e-6]}
        dictionnary_steps = {100:[100,200], 200:[100,200], 300:[100,200],400:[100,200],500:[100,200],
                                600:[100,200],700:[100,200], 800:[100,200], 900:[100,200], 1000:[100,200]}
        for k in list(images_numbers):
            #run sde
            for latent in range(sde_range[0],sde_range[1], sde_range[2]):
                for corruption in corruptions_list:
                    directory_reconstruction_sde_latent_corruption = f"{directory_reconstruction_sde}/latent_{latent}/{corruption}/{k}"
                    img_tensor, original = corrupt(number = k, corruption = corruption, random_sampling=False, random_corruption=False)
                    directory_corrupted_latent = f"/mnt/scitas/bastien/ODEDIT/corrupted/{corruption}"
                    os.makedirs(directory_corrupted_latent, exist_ok=True)
                    save_image(img_tensor.cpu()/2+0.5,directory_corrupted_latent+f"/{k}.png")
                    save_image(original.cpu()/2+0.5,directory_clean_original+f"/{k}.png")

            #run ode
            for corruption in corruptions_list:
                img_tensor, original = corrupt(number = k, corruption = corruption, random_sampling=False, random_corruption=False)
                # TO DO encode 
                latent_codes = self.encode_inputs(img_tensor)
                for latent in range(ode_range[0], ode_range[1], ode_range[2]):
                    directory_reconstruction_ode_latent_corruption = f"{directory_reconstruction_ode}/latent_{latent}/{corruption}/{k}"
                    save_image(latent_codes[latent].cpu()/2+0.5, f"{directory_latent}/{latent}/{corruption}/{k}.png")
                    image_encoded = latent_codes[latent]
                    ## To define --> Probably fix steps with different epsilon
                    for number_of_steps in list_steps:
                        for epsilon in epsilons:
                            list_of_evolution_reverse, samples = self.editing_with_ode(latent_codes, self.ddpm.unet, t_start = latent, 
                                        std_div = -1, epsilon = epsilon, steps = number_of_steps, power =0.5, 
                     number_of_sample = number_of_sample, alphas_cumprod = self.alphas_cumprod, corrector_step = True,
                                        deep_correction=True, comparison = False)
                    for index, sample in enumerate(samples):
                        save_image(latent_codes[latent].cpu()/2+0.5, 
                                    f"{directory_reconstruction_ode_latent_corruption}/{k}/{number_of_steps}_{epsilon}/{index}.png")
        
        return


    @torch.no_grad()
    def run_sdeedit(self, number_of_test_per_corruption = 15000,random_corruption = True, batch_size = 36, celebahq = False):
        corruptions_list = ['spatter', "motion_blur", "frost","speckle_noise","impulse_noise","shot_noise","jpeg_compression","pixelate","brightness",
                            "fog","saturate","gaussian_noise",'elastic_transform','snow','masking_vline_random_color','masking_gaussian','glass_blur','gaussian_blur','contrast']
        
        corruption_severity  = 4
        stop_iteration_at = int(number_of_test_per_corruption/(batch_size*self.cfg.trainer.world_size))+1
        print("stop_iteration_at", stop_iteration_at)
        add_index_per_gpu = self.cfg.trainer.gpu*stop_iteration_at*batch_size
        print(f"On GPU {self.cfg.trainer.gpu} start index:", add_index_per_gpu)

        sample_step = 1
        if celebahq:
            model, betas, num_timesteps, logvar = load_model(device=f"cuda:{self.cfg.trainer.gpu}")
        else:
            model = self.ema_model
            betas = th.from_numpy(self.diffusion.betas)
            alphas = (1.0 - betas).numpy()
            alphas_cumprod = np.cumprod(alphas, axis=0)
            alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
            posterior_variance = betas * \
                (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            logvar = np.log(np.maximum(posterior_variance, 1e-20))
            num_timesteps = int(betas.shape[0])

        for noise_levels in range(200,800,100):
            directory_reconstruction =  f"/mnt/scitas/bastien/SDE_xp/random_corruption/reconstruction_{noise_levels}/"
            directory_inputs = f"/mnt/scitas/bastien/SDE_xp/random_corruption/inputs_{noise_levels}/"
            directory_targets = f"/mnt/scitas/bastien/SDE_xp/random_corruption/targets_{noise_levels}/"
            os.makedirs(directory_reconstruction, exist_ok=True)
            os.makedirs(directory_inputs, exist_ok=True)
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
                        results = SDEditing(inputs, betas, logvar, model, sample_step, noise_levels, n=1, huggingface = celebahq)
                        results_normalized = results / 2 + 0.5
                        for n, image in enumerate(results_normalized):
                            save_image(image, directory_reconstruction+f"reconstruction_{i*batch_size+n+add_index_per_gpu}.png")
                except Exception as e:
                    print(f"Error in step {i}, gpu {self.cfg.trainer.gpu}")
                    print(e)
                    continue
                      

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



   