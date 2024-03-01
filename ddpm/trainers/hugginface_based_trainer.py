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
from torch.utils.data import Dataset,Subset,  DataLoader, DistributedSampler, RandomSampler
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


from diffusers import UNet2DModel, DDIMScheduler, VQModel, DDIMInverseScheduler
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline,DiffusionPipeline

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
                        shuffle=True,
                        single_gpu=False
                        ):
    if single_gpu:
        sampler = RandomSampler(dataset)
    else:
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


class Hugginface_Trainer(BaseTrainer):
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

        self.root = self.cfg.trainer.log_root
        os.makedirs(self.root, exist_ok=True)

        # Get the pipeline, model and its parameters
        self.model_id = self.cfg.trainer.model_id
        self.ddpm = DDPMPipeline.from_pretrained(self.model_id).to(f"cuda:{self.cfg.trainer.gpu}")
        self.model = self.ddpm.unet
        self.num_timesteps = int(self.ddpm.scheduler.betas.shape[0])
        self.betas = self.ddpm.scheduler.betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.dynamic_threshol_ratio = self.cfg.trainer.dynamic_threshol_ratio
        self.dynamic_threshold_max = self.cfg.trainer.dynamic_threshold_max

        self.train_dataset, self.test_dataset = get_dataset(None, self.cfg)
        if not self.cfg.trainer.single_gpu:
            dist.barrier()
        self.corruptions_list = self.cfg.trainer.corruptions_list or ["motion_blur", "frost", "speckle_noise", "impulse_noise", "shot_noise", "jpeg_compression",
                                "pixelate", "brightness", "fog", "saturate", "gaussian_noise", 'elastic_transform','saturate',
                                'snow', 'masking_vline_random_color', 'spatter', 'glass_blur', 'gaussian_blur', 'contrast', 'masking_random_color']
        LOG.info(f"Corruptions considered : {self.corruptions_list}")
        self.corruptions_functions = {"shot_noise" : shot_noise, "gaussian_blur" : gaussian_blur, "spatter" : spatter, "fog":fog, "frost":frost, 
                            "snow":snow, "glass_blur":glass_blur, "elastic_transform":elastic_transform, "contrast":contrast, "brightness":brightness,
                            "gaussian_noise":gaussian_noise, "impulse_noise":impulse_noise, "masking_random_color_random":masking_random_color_random,
                            "motion_blur":motion_blur, "saturate":saturate,'masking_vline_random_color':masking_vline_random_color,
                            "jpeg_compression":jpeg_compression, "pixelate":pixelate,"speckle_noise":speckle_noise,"masking_random_color":masking_random_color}
        

        LOG.info(f"train dataset length {len(self.train_dataset)}")
        LOG.info(f"test dataset length {len(self.test_dataset)}")



    def train(self,) -> None:
        LOG.info("Huggingface models are pretrained, no need to train them")
        return 
        

    def log(self,) -> None:
        """
        Log inputs/outputs in clearml or wandb or tensorboard
        Log process 
        """
    

    def _clip_inputs(self, sample: torch.FloatTensor, t : int, number_of_stds: float = 2.):
        """
        Cliping the inputs with an confidence interval given by the diffusion schedule
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half
        alpha_t = self.alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t).item()
        one_minus_sqrt_alpha_t = torch.sqrt(1-alpha_t).item()
        confidence_interval = [-sqrt_alpha_t - number_of_stds * one_minus_sqrt_alpha_t,sqrt_alpha_t + number_of_stds * one_minus_sqrt_alpha_t]
        sample = torch.clamp(sample, confidence_interval[0], confidence_interval[1])
        sample = sample.to(dtype)

        return sample


    def _threshold_sample(self, sample: torch.FloatTensor, dynamic_thresholding_ratio: float, sample_max_value : float = 5/3) -> torch.FloatTensor:
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
    def langevin_sampling(self, inputs, t, t_prev, steps = 100, epsilon = 1e-5,min_variance = -1, 
                         denoising_step = True, clip_prev = False, clip_now = False, dynamic_thresholding = False,  power  = 0.5):


        alphas_cumprod = self.ddpm.scheduler.alphas_cumprod.cpu().numpy()
        model = self.ddpm.unet
        index = t
        t = torch.tensor([t] * inputs.shape[0]).cuda(self.cfg.trainer.gpu)   
        mean_coef_t = torch.sqrt(_extract_into_tensor(alphas_cumprod, t , inputs.shape))
        variance = _extract_into_tensor(1.0 - alphas_cumprod, t , inputs.shape)
        std = torch.sqrt(variance)
        std_epsilon = []
        mean_epsilon = []
        if t_prev is not None:
            t_prev = torch.tensor([t_prev] * inputs.shape[0]).cuda(self.cfg.trainer.gpu)     
            mean_coef_t_prev = torch.sqrt(_extract_into_tensor(alphas_cumprod, t_prev , inputs.shape))
            variance_t_prev = _extract_into_tensor(1.0 - alphas_cumprod, t_prev , inputs.shape)
            std_prev = torch.sqrt(variance_t_prev)
            if self.cfg.trainer.clip_inputs:
                inputs = self._clip_inputs(inputs, t = index, number_of_stds = self.cfg.trainer.number_of_stds)
            noise_estimate_t_prev = self.model(inputs, t_prev)['sample']
            x0_t_1 = (inputs - std_prev * noise_estimate_t_prev)/mean_coef_t_prev
            if dynamic_thresholding:
                x0_t_1 = self._threshold_sample(x0_t_1, self.dynamic_threshol_ratio, self.dynamic_threshold_max)
            if clip_prev:
                x0_t_1 = x0_t_1.clamp(-1,1)
            inputs = inputs + x0_t_1 * (mean_coef_t - mean_coef_t_prev)

        if min_variance > 0:
            alpha_coef =  (variance/min_variance) * epsilon
        else:
            alpha_coef = torch.ones_like(variance) * epsilon
        with torch.no_grad():
            for i in range(steps):
                if self.cfg.trainer.clip_inputs:
                    inputs = self._clip_inputs(inputs, t = index, number_of_stds = self.cfg.trainer.number_of_stds)
                noise_estimate = self.model(inputs, t)['sample']
                if dynamic_thresholding:
                    x0_t = (inputs - std * noise_estimate)/mean_coef_t
                    x0_t = self._threshold_sample(x0_t, self.dynamic_threshol_ratio, self.dynamic_threshold_max)
                    noise_estimate = (inputs - mean_coef_t * x0_t) / std
                elif clip_now:
                    x0_t = (inputs - std * noise_estimate)/mean_coef_t
                    x0_t = x0_t.clamp(-1,1)
                    noise_estimate = (inputs - mean_coef_t * x0_t) / std
                std_epsilon.append(noise_estimate[0].cpu().std().item())
                mean_epsilon.append(noise_estimate[0].cpu().mean().item())
                score = - noise_estimate / std
                noise = torch.randn(inputs.shape).cuda(self.cfg.trainer.gpu)
                if steps > 1:
                    inputs = (inputs + alpha_coef * score) + torch.pow(2*alpha_coef, power) * noise
            if denoising_step:
                if self.cfg.trainer.clip_inputs:
                    inputs = self._clip_inputs(inputs, t = index, number_of_stds = self.cfg.trainer.number_of_stds)
                noise_estimate = self.model(inputs, t)['sample']
                score = - noise_estimate / std
                inputs = (inputs + alpha_coef * score) 
        return inputs, alpha_coef, std_epsilon, mean_epsilon


    @torch.no_grad()
    def ddim_step(self, inputs, t, clip_denoised = False, dynamic_thresholding = True, clip_value = 1, sigma = 0, 
                forward = True, number_of_sample = 1):
        alphas_cumprod = self.alphas_cumprod.cpu().numpy()
        number_of_timesteps = self.ddpm.scheduler.betas.shape[0]
        index = t
        # model = self.model.cuda(self.cfg.trainer.gpu)
        # inputs = inputs.cuda(self.cfg.trainer.gpu)
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
        if self.cfg.trainer.clip_inputs:
            inputs = self._clip_inputs(inputs, t = index, number_of_stds = self.cfg.trainer.number_of_stds)
        noise_estimate = self.model(inputs, t)['sample']
        std_epsilon = noise_estimate[0].cpu().std().item()
        mean_epsilon = noise_estimate[0].cpu().mean().item()
        if dynamic_thresholding:
            x0_t = (inputs - std * noise_estimate)/mean_coef_t
            x0_t = self._threshold_sample(x0_t, self.dynamic_threshol_ratio, self.dynamic_threshold_max)
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


    def corrupt(self, image = None, number = 9999,corruption = 'spatter', random_sampling = False, random_corruption = False):
        img_list = []
        original_list = []
        if random_corruption:         
            corruption = random.choice(list(corruptions_functions.keys()))
        if image is None:
            if random_sampling:
                number = np.random.randint(1,29999)
                # img_path = f"{self.root}/CelebAMask-HQ/CelebA-HQ-img/{number}.jpg"
                img_path = f"{self.cfg.trainer.datapath}/{number}.jpg"
                img_pil = PIL.Image.open(img_path).resize([256,256])

            else:
                img_path = f"{self.cfg.trainer.datapath}/{number}.jpg"
                img_pil = PIL.Image.open(img_path).resize([256,256])
            corrupted_sample = PIL.Image.fromarray(self.corruptions_functions[corruption](img_pil, severity  = 5).astype(np.uint8)).resize([256,256])
            img_tensor = (torch.from_numpy(np.array(corrupted_sample)).permute(2,0,1)/255)*2-1
            original = (torch.from_numpy(np.array(img_pil)).permute(2,0,1)/255)*2-1
            img_list.append(img_tensor)
            original_list.append(original)
        else:
            if len(image.shape)>3:
                for img in image:
                    img_pil = (transforms.functional.to_pil_image(img/2+0.5)).resize([256,256])
                    corrupted_sample = PIL.Image.fromarray(self.corruptions_functions[corruption](img_pil, severity  = 5).astype(np.uint8)).resize([256,256])
                    img_tensor = (torch.from_numpy(np.array(corrupted_sample)).permute(2,0,1)/255)*2-1
                    original = (torch.from_numpy(np.array(img_pil)).permute(2,0,1)/255)*2-1
                    img_list.append(img_tensor)
                    original_list.append(original)
            else:
                img_pil = (transforms.functional.to_pil_image(image/2+0.5)).resize([256,256])
                corrupted_sample = PIL.Image.fromarray(self.corruptions_functions[corruption](img_pil, severity  = 5).astype(np.uint8)).resize([256,256])
                img_tensor = (torch.from_numpy(np.array(corrupted_sample)).permute(2,0,1)/255)*2-1
                original = (torch.from_numpy(np.array(img_pil)).permute(2,0,1)/255)*2-1
                img_list.append(img_tensor)
                original_list.append(original)

        img_tensor = torch.stack(img_list)
        original = torch.stack(original_list)
        return img_tensor, original


    def editing_with_ode(self, latent_codes, model, t_start = 1000, std_div = -1., annealing = False, epsilon = 1e-8, 
                        steps = 20, power =0.5, 
                        min_variance = -1. , number_of_sample = 1,normalize=False, normalize_mean = False,
                        corrector_step = 1,deep_correction = False, comparison = False):
        alphas_cumprod = self.ddpm.scheduler.alphas_cumprod
        list_of_evolution_reverse = []
        final_samples = []
        t_valid = list(range(0, min(len(latent_codes)+1,t_start)))
        std_recon = 1
        epsilon = epsilon
        if t_start > 99 + corrector_step and corrector_step > 1:
            correction_latents = np.linspace(99, np.max(t_valid), corrector_step).astype(int).tolist()
            epsilon_correction = np.geomspace(1,100,1000)[::-1] 
            epsilon_correction = epsilon_correction / epsilon_correction[np.max(t_valid)]
        else:
            correction_latents = [np.max(t_valid)]
        # print(f"correction_latents {correction_latents}")
        with torch.no_grad():
            if self.cfg.trainer.normalize and np.max(t_valid) == len(latent_codes) - 1:
                inputs = (latent_codes[np.max(t_valid)] - latent_codes[np.max(t_valid)].mean([1,2,3],keepdim=True))/latent_codes[np.max(t_valid)].std([1,2,3], keepdim=True)
            elif self.cfg.trainer.normalize_mean and np.max(t_valid) == len(latent_codes) - 1:
                inputs = (latent_codes[np.max(t_valid)] - latent_codes[np.max(t_valid)].mean())
            else:
                inputs = latent_codes[np.max(t_valid)]
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(0)
                batch_size = 1
            elif len(inputs.shape) == 4:
                batch_size = inputs.shape[0]
            else:
                raise NotImplementedError

            if batch_size > 1 and number_of_sample > 1:
                inputs = inputs.split(1, dim=0)
                inputs = [inp.repeat(number_of_sample, 1, 1, 1) for inp in inputs]
                inputs = torch.cat(inputs)
            elif number_of_sample > 1:
                inputs = inputs.repeat(number_of_sample, 1, 1, 1)
            else:
                inputs = inputs

            inputs = inputs.cuda(self.cfg.trainer.gpu)

            edited_to_be_done = True
            if comparison:
                run_with_correction = [True,False]
            else:
                run_with_correction = [corrector_step]
            for corrector in run_with_correction:
                for t in tqdm(t_valid[::-1]):
                    counter = 0
                    if std_div > 0:
                        if t>100 and t < t_start-1:
                            while std_recon < (1-std_div) or std_recon > (1+std_div):
                                inputs,alpha_coef, list_of_stds, list_of_means = self.langevin_sampling(inputs, t, None, steps = steps, epsilon = epsilon,
                                                    min_variance = min_variance, clip_prev = False, clip_now = False,dynamic_thresholding=self.cfg.trainer.dynamic_thresholding_langevin, power = power)
            
                                std_recon = list_of_stds[-1]
                                counter += 1
                                if counter > 100:
                                    epsilon = 2*epsilon
                                for h in range(len(inputs)):
                                    list_of_evolution_reverse.append(inputs[h].cpu())
                            epsilon *= 1
                            
                    elif t == np.max(t_valid) and edited_to_be_done:
                        before = inputs.cpu()
                        if annealing>1:
                            new_epsilon = epsilon
                            for j in range(int(annealing)):
                                step_per_epsilon = steps // len(range(int(annealing)))
                                inputs,alpha_coef, list_of_stds, list_of_means = self.langevin_sampling(inputs, t, None, steps = step_per_epsilon, epsilon = new_epsilon,
                                                    min_variance = min_variance, clip_prev = False, clip_now = False,dynamic_thresholding=self.cfg.trainer.dynamic_thresholding_langevin,power = power)
                                new_epsilon  = new_epsilon / 2
   
                        else:
                            inputs,alpha_coef, list_of_stds, list_of_means = self.langevin_sampling(inputs, t, None, steps = steps, epsilon = epsilon,
                                    min_variance = min_variance, clip_prev = False, clip_now = False,dynamic_thresholding=self.cfg.trainer.dynamic_thresholding_langevin, power = power)  
                        std_recon = list_of_stds[-1]
                        for h in range(len(inputs)):
                                list_of_evolution_reverse.append(inputs[h].cpu())
                        edited_latents = inputs
                        edited_to_be_done = False
                    elif t == np.max(t_valid):
                        inputs = edited_latents

                    elif t in correction_latents:
                        before = inputs.cpu()
                        
                        new_epsilon = epsilon * epsilon_correction[t]
                        print(f'at {t} epsilon before {epsilon}, epsilon after {new_epsilon}')
                        if annealing>1:
                            for j in range(int(annealing)):
                                step_per_epsilon = steps // len(range(int(annealing)))
                                inputs,alpha_coef, list_of_stds, list_of_means = self.langevin_sampling(inputs, t, None, steps = step_per_epsilon, epsilon = new_epsilon,
                                                    min_variance = min_variance, clip_prev = False, clip_now = False,dynamic_thresholding=self.cfg.trainer.dynamic_thresholding_langevin,power = power)
                                new_epsilon  = new_epsilon / 2
                        else:
                            inputs,alpha_coef, list_of_stds, list_of_means = self.langevin_sampling(inputs, t, None, steps = steps, epsilon = new_epsilon,
                                    min_variance = min_variance, clip_prev = False, clip_now = False,dynamic_thresholding=self.cfg.trainer.dynamic_thresholding_langevin, power = power)  
                        std_recon = list_of_stds[-1]
                        for h in range(len(inputs)):
                                list_of_evolution_reverse.append(inputs[h].cpu())

                    inputs, std_epsilon, mean_epsilon = self.ddim_step(inputs, t, sigma = 0.,
                                                            clip_denoised=False,dynamic_thresholding=self.cfg.trainer.dynamic_thresholding_ddim, forward=True, number_of_sample = number_of_sample)               
                    std_recon = std_epsilon
                    for h in range(len(inputs)):
                        list_of_evolution_reverse.append(inputs[h].cpu())
                for sample in list_of_evolution_reverse[-batch_size*number_of_sample:]:
                    final_samples.append(sample)
                    
            for h in range(len(inputs)):
                list_of_evolution_reverse.append(inputs[h].cpu())
        return list_of_evolution_reverse, final_samples


    @torch.no_grad()
    def encode_inputs(self, inputs, noisified = True): 
        latent_codes = []
        list_std_encoding = []
        list_mean_encoding = []
        if noisified:
            inputs = inputs + 0.01 * torch.randn_like(inputs)
        with torch.no_grad():
            latent_codes.append(inputs.cpu())
            for t in range(0,self.ddpm.scheduler.betas.shape[0]-1):
                inputs, std_eps, mean_eps = self.ddim_step(inputs, t,  sigma = 0.,clip_denoised=False,dynamic_thresholding = self.cfg.trainer.dynamic_thresholding_ddim, forward=False)
                latent_codes.append(inputs.cpu())
                list_std_encoding.append(std_eps)
                list_mean_encoding.append(mean_eps)
        return latent_codes, list_mean_encoding, list_std_encoding


    @torch.no_grad()
    def batch_for_single_image_experiments(self, input_image = None, number = 1, number_of_images = 4):
            # corruptions_list = ["motion_blur", "frost", "speckle_noise", "impulse_noise", "shot_noise", "jpeg_compression",
                                # "pixelate", "brightness", "fog", "saturate", "gaussian_noise", 'elastic_transform',
                                # 'snow', 'masking_vline_random_color', 'spatter', 'glass_blur', 'gaussian_blur', 'contrast', 'masking_random_color']
            # print(self.cfg.trainer.number_of_image)
            corruptions_list = self.corruptions_list
            number_of_images = self.cfg.trainer.number_of_image or number_of_images
            # print('number_of_images', number_of_images)
            img_list = []
            original_list = []
            # if number_of_images > 0:
            #     subset = np.random.choice(corruptions_list, number_of_images, replace=False)
            # else:
            #     number_of_images = len(corruptions_list)
            subset = corruptions_list
            
            # print(f'subset length {len(subset)}')
            corruptions_order = []
            for corruption in subset:
                img_tensor, original = self.corrupt(image=input_image, number = number, corruption=corruption)
                img_list.append(img_tensor.squeeze())
                original_list.append(original.squeeze())
                corruptions_order.append(corruption)

            img_tensor_batch = torch.stack(img_list)
            original_batch = torch.stack(original_list)

            # The rest of your code to log or save the results goes here...
            # For example, you can use img_tensor_batch and original_batch as needed.

            return img_tensor_batch, original_batch, corruptions_order
        

    @torch.no_grad()
    def run_qualitative_experiments(self,number_of_image = 1, corruptions = 'all', sde_range = [99,800,100], 
                            ode_range = [99, 1000, 100], number_of_sample = 3, celebaHQ = True):
        number_of_image = self.cfg.trainer.number_of_image or number_of_image
        ode_range = self.cfg.trainer.ode_range or ode_range
        sde_range = self.cfg.trainer.sde_range or sde_range
        number_of_sample = self.cfg.trainer.number_of_sample or number_of_sample
        if celebaHQ:
            if self.cfg.trainer.use_val:
                try:
                    all_numbers = range(len(self.test_dataset))
                    assert len(all_numbers)>0
                except:
                    LOG.info("Using train set instead.")
                    with open_dict(self.cfg):
                        self.cfg.trainer.use_val = False
                    all_numbers = range(len(self.train_dataset))
            else:
                all_numbers = range(len(self.train_dataset))
            images_numbers = np.random.choice(all_numbers, number_of_image)
            
        if self.cfg.trainer.image_number is not None:
            images_numbers = np.array([self.cfg.trainer.image_number])

        sde_model, sde_betas, sde_num_timesteps, sde_logvar = load_model(model_id = self.cfg.trainer.model_id, device=f"cuda:{self.cfg.trainer.gpu}")
        if self.cfg.trainer.use_val:
            self.subset_dataset = Subset(self.test_dataset, list(images_numbers))
        else:
            self.subset_dataset = Subset(self.train_dataset, list(images_numbers))
        print(f"{self.cfg.trainer.gpu} : {list(images_numbers)[:2]}")
        # print(list(images_numbers))
        LOG.info(f"SubDataset length {len(self.subset_dataset)} ")
        print("sync_key", self.cfg.trainer.sync_key)
        directory_base = f"{self.root}/ODEDIT/qualitative"
        directory_latent = f"{self.root}/ODEDIT/qualitative/latent"
        os.makedirs(directory_latent, exist_ok=True)
        os.makedirs(directory_base, exist_ok=True)

        self.dataloader = create_dataloader(self.subset_dataset,
                                        rank=self.cfg.trainer.rank,
                                        max_workers=self.cfg.trainer.num_workers,
                                        world_size=self.cfg.trainer.world_size,
                                        batch_size=1,
                                        shuffle=False,
                                        single_gpu = self.cfg.trainer.single_gpu
                                        )
        LOG.info(f"Dataloader length {len(self.dataloader)} on GPU: {self.cfg.trainer.gpu}")
        
        # for rcp in case of crash
        if os.path.exists(f"{directory_base}/checkpoint_state.p"):
            ckpt_dict = pickle.load(open(f"{directory_base}/checkpoint_state.p","rb"))
            current_index = ckpt_dict['index']
            current_epsilon = ckpt_dict['epsilon']
            run_sdedit = ckpt_dict['run_sdedit']
            ckpt_dict = {'index':current_index, 'epsilon':current_epsilon, 'run_sdedit':run_sdedit}
        else:
            current_index = 0
            current_epsilon = 0
            run_sdedit = True
            ckpt_dict = {'index':current_index, 'epsilon':current_epsilon, 'run_sdedit':run_sdedit}

        epsilons = np.geomspace(self.cfg.trainer.min_epsilon,self.cfg.trainer.max_epsilon, self.cfg.trainer.number_of_epsilons)
        list_steps = [self.cfg.trainer.number_of_steps]

        LOG.info(f"Starting Dataloader loop.")
        for k, (_, img_batch, indexes) in enumerate(self.dataloader):
            if k >= current_index:
                current_index = k
                torch.cuda.empty_cache()
                index_list = indexes.tolist()

                if self.cfg.trainer.image_number is None:
                    img_tensor, original, corruptions_order = self.batch_for_single_image_experiments(img_batch, number = k)
                else:
                    k = self.cfg.trainer.image_number
                    img_tensor, original, corruptions_order = self.batch_for_single_image_experiments(None, number = k)
                img_tensor = img_tensor.cuda(self.cfg.trainer.gpu)
                original = original.cuda(self.cfg.trainer.gpu)
                index_directory = []
                for i,corr in enumerate(corruptions_order):
                    counter = 0
                    while os.path.isdir(f"{directory_base}/{corruptions_order[i]}/{index_list[0]}_{counter}"):
                        counter +=1
                    index_directory.append(f"{directory_base}/{corruptions_order[i]}/{index_list[0]}_{counter}")
                    os.makedirs(f"{directory_base}/{corruptions_order[i]}/{index_list[0]}_{counter}", exist_ok=True)
                    os.makedirs(f"{directory_base}/{corruptions_order[i]}/{index_list[0]}_{counter}/sde", exist_ok=True)
                    os.makedirs(f"{directory_base}/{corruptions_order[i]}/{index_list[0]}_{counter}/ode", exist_ok=True)
                for i,image in enumerate(img_tensor):
                    save_image(img_tensor[i].cpu()/2+0.5,f"{index_directory[i]}/corrupted_{self.cfg.trainer.gpu}.png")
                    if i == 0:
                        save_image(original[i].cpu()/2+0.5,f"{index_directory[i]}/original_{self.cfg.trainer.gpu}.png") 
                if self.cfg.trainer.gpu == 0:
                    grid_corrupted = make_grid(img_tensor.cpu().detach())
                    grid_original = make_grid(original.cpu().detach())
                    img_grid_corrupted = wandb.Image(grid_corrupted.permute(1,2,0).numpy())
                    img_grid_original= wandb.Image(grid_original.permute(1,2,0).numpy())
                    wandb.log({f"Corruption_{[corr[:5] for corr in corruptions_order]}": img_grid_corrupted},commit=True)
                    wandb.log({f"Original": img_grid_original},commit=True)
            
                if self.cfg.trainer.run_sdedit and run_sdedit:
                    for latent in range(sde_range[0],sde_range[1], sde_range[2]):
                        sample_step = 1
                        results = SDEditing(img_tensor, sde_betas, sde_logvar, sde_model, sample_step, latent, n=number_of_sample, huggingface = True)
                        results_normalized = results / 2 + 0.5
                        samples = torch.stack(results_normalized.split(number_of_sample, dim=0))
                        for k, corruption_samples in enumerate(samples):
                            for sample_index, sample in enumerate(corruption_samples):
                                save_image(sample.cpu(), f"{index_directory[k]}/sde/{latent}_{sample_index}_{self.cfg.trainer.gpu}.png")
                        if self.cfg.trainer.gpu == 0:
                            grid_reco_sde = make_grid(results_normalized.cpu().detach())
                            img_grid_reco_sde = wandb.Image(grid_reco_sde.permute(1,2,0).numpy())
                            wandb.log({f"SDE_Reconstruction_{latent}": img_grid_reco_sde},commit=True)
                        torch.cuda.empty_cache()
                run_sdedit = False
                ckpt_dict = {'index':current_index, 'epsilon':current_epsilon, 'run_sdedit':run_sdedit}
                pickle.dump(ckpt_dict,open(f"{directory_base}/checkpoint_state.p",'wb'))

                

                #run ode
                latent_codes, _, _ = self.encode_inputs(img_tensor)
                for latent in range(ode_range[0], ode_range[1], ode_range[2]):
                    if self.cfg.trainer.gpu == 0:
                        grid_latent = make_grid(torch.clamp(latent_codes[latent].cpu().detach(),-1,1))
                        img_grid_latent = wandb.Image(grid_latent.permute(1,2,0).numpy())
                        wandb.log({f"Latent_{latent}": img_grid_latent},commit=True)

                    ## To define --> Probably fix steps with different epsilon
                    for number_of_steps in list_steps:
                        for l, epsilon in enumerate(epsilons):
                            if l >= current_epsilon or self.cfg.trainer.run_all_epsilon:
                                current_epsilon = l
                                list_of_evolution_reverse, samples = self.editing_with_ode(latent_codes, self.ddpm.unet, t_start = latent, 
                                            std_div = -1, epsilon = epsilon, steps = number_of_steps, power =0.5, 
                                            number_of_sample = number_of_sample,
                                            corrector_step = self.cfg.trainer.number_of_latents_corrected,
                                            deep_correction=False, comparison = False)
                                samples_stacked = torch.stack(samples)

                                samples = torch.stack(samples_stacked.split(number_of_sample, dim=0)) / 2 + 0.5
                                for k, corruption_samples in enumerate(samples):
                                    for sample_index, sample in enumerate(corruption_samples):
                                        save_image(sample.cpu(), f"{index_directory[k]}/ode/{sample_index}_{number_of_steps}_{round(epsilon,6)}_{self.cfg.trainer.gpu}.png")
                                if self.cfg.trainer.gpu == 0:
                                    grid_reco = make_grid(samples_stacked.cpu().detach())
                                    img_grid_reco= wandb.Image(grid_reco.permute(1,2,0).numpy())
                                    wandb.log({f"Reconstruction_l{latent}_e{round(epsilon,6)}": img_grid_reco},commit=True)
                            ckpt_dict = {'index':current_index, 'epsilon':current_epsilon, 'run_sdedit':run_sdedit}
                            pickle.dump(ckpt_dict,open(f"{directory_base}/checkpoint_state.p",'wb'))
                            
        return


    @torch.no_grad()
    def run_experiments(self,number_of_image = 4, sde_range = [99,1000,100], 
                            ode_range = [99, 1000, 100], number_of_sample = 4, celebaHQ = True):
        number_of_image = self.cfg.trainer.number_of_image or number_of_image
        corruptions_list = self.corruptions_list

        if celebaHQ:
            all_numbers = range(len(self.train_dataset))
            images_numbers = np.random.choice(all_numbers, number_of_image)

        ### get subset of dataset based on the indices

        self.subset_dataset = Subset(self.train_dataset, list(images_numbers))
        LOG.info(f"SubDataset length {len(self.subset_dataset)} ")

        directory_clean_original = f"{self.root}/ODEDIT/original"
        directory_corrupted = f"{self.root}/ODEDIT/corrupted"
        directory_reconstruction_sde =  f"{self.root}/ODEDIT/sde"
        directory_reconstruction_ode =  f"{self.root}/ODEDIT/ode"
        directory_latent = f"{self.root}/ODEDIT/latent"
        os.makedirs(directory_clean_original, exist_ok=True)
        os.makedirs(directory_corrupted, exist_ok=True)
        os.makedirs(directory_reconstruction_sde, exist_ok=True)
        os.makedirs(directory_reconstruction_ode, exist_ok=True)
        self.dataloader = create_dataloader(self.subset_dataset,
                                        rank=self.cfg.trainer.rank,
                                        max_workers=self.cfg.trainer.num_workers,
                                        world_size=self.cfg.trainer.world_size,
                                        batch_size=self.cfg.trainer.batch_size,
                                        shuffle=False
                                        )
        LOG.info(f"Dataloader length {len(self.dataloader)} on GPU: {self.cfg.trainer.gpu}")

        ## HARD CODED FOR FASTNESS BASED ON EXPERIENCE
        dictionnary_corruption_latent = {"frost": [199,999], 
                                        "speckle_noise": [199,999],
                                        'impulse_noise':[199,999], 
                                        "shot_noise":[199,999], 
                                        "fog":[999],
                                        'snow':[199,999], 
                                        'masking_vline_random_color':[199,999], 
                                        'spatter':[199,999], 
                                        'contrast':[999], 
                                        'masking_random_color':[999]}
                                        
        dictionnary_corruption_epsilon = {}

        epsilons = np.geomspace(self.cfg.trainer.min_epsilon,self.cfg.trainer.max_epsilon, self.cfg.trainer.number_of_epsilons)

        list_steps = [self.cfg.trainer.number_of_steps]
        ### IF different per latent
        # dictionnary_epsilon = {100:[1e-5, 1e-6], 200:[1e-5, 1e-6], 300:[1e-5, 1e-6],400:[1e-5, 1e-6],500:[1e-5, 1e-6],
        #                         600:[1e-5, 1e-6],700:[1e-5, 1e-6], 800:[1e-5, 1e-6], 900:[1e-5, 1e-6], 1000:[1e-5, 1e-6]}
        # dictionnary_steps = {100:[100,200], 200:[100,200], 300:[100,200],400:[100,200],500:[100,200],
        #                         600:[100,200],700:[100,200], 800:[100,200], 900:[100,200], 1000:[100,200]}
        LOG.info(f"Starting Dataloader loop.")
        for k, (_, img_batch, indexes) in enumerate(self.dataloader):
            #run sde

            index_list = indexes.tolist()
            for l, corruption in enumerate(corruptions_list):
                
                img_tensor, original = self.corrupt(image = img_batch, number = k, corruption = corruption, random_sampling=False, random_corruption=False)
                img_tensor = img_tensor.cuda(self.cfg.trainer.gpu)
                original = original.cuda(self.cfg.trainer.gpu)
                directory_corrupted_per_corruption = f"{self.root}/ODEDIT/corrupted/{corruption}"
                os.makedirs(directory_corrupted_per_corruption, exist_ok=True)
                for i,image_index in enumerate(index_list):
                    save_image(img_tensor[i].cpu()/2+0.5,directory_corrupted_per_corruption+f"/{image_index}_{self.cfg.trainer.gpu}.png")
                    save_image(original[i].cpu()/2+0.5,directory_clean_original+f"/{image_index}_{self.cfg.trainer.gpu}.png")
                if self.cfg.trainer.gpu == 0:
                    grid_corrupted = make_grid(img_tensor.cpu().detach())
                    grid_original = make_grid(original.cpu().detach())
                    img_grid_corrupted = wandb.Image(grid_corrupted.permute(1,2,0).numpy())
                    img_grid_original= wandb.Image(grid_original.permute(1,2,0).numpy())
                    wandb.log({f"Corruption_{corruption}": img_grid_corrupted},commit=True)
                    if l == 0:
                        wandb.log({f"Original": img_grid_original},commit=True)

                if self.cfg.trainer.run_sdedit:
                    for latent in range(sde_range[0],sde_range[1], sde_range[2]):
                        directory_reconstruction_sde_latent_corruption = f"{directory_reconstruction_sde}/latent_{latent}/{corruption}/{k}"
                        sample_step = 1
                        directory_reconstruction_sde_latent_corruption = f"{directory_reconstruction_sde}/latent_{latent}"
                        os.makedirs(directory_reconstruction_sde_latent_corruption, exist_ok=True)
                        results = SDEditing(img_tensor, sde_betas, sde_logvar, sde_model, sample_step, latent, n=number_of_sample, huggingface = True)
                        results_normalized = results / 2 + 0.5
                        for n, image in enumerate(results_normalized):
                            save_image(image.cpu(), directory_reconstruction_sde_latent_corruption+f"/{n}_{self.cfg.trainer.gpu}.png")
                        if self.cfg.trainer.gpu == 0:
                            grid_reco_sde = make_grid(results_normalized.cpu().detach())
                            img_grid_reco_sde = wandb.Image(grid_reco_sde.permute(1,2,0).numpy())
                            wandb.log({f"SDE_Reconstruction_{latent}_{n}": img_grid_reco_sde},commit=True)

                #run ode
                latent_codes, _, _ = self.encode_inputs(img_tensor)
                for latent in range(ode_range[0], ode_range[1], ode_range[2]):
                    directory_reconstruction_ode_latent_corruption = f"{directory_reconstruction_ode}/latent_{latent}/{corruption}"
                    os.makedirs(directory_reconstruction_ode_latent_corruption, exist_ok=True)

                    for i,image_index in enumerate(index_list):
                        # os.makedirs(f"{directory_reconstruction_ode_latent_corruption}/{image_index}", exist_ok=True)
                        save_image(latent_codes[latent][i].cpu()/2+0.5, f"{directory_reconstruction_ode_latent_corruption}/{image_index}.png")
                    if self.cfg.trainer.gpu == 0:
                        grid_latent = make_grid(torch.clamp(latent_codes[latent].cpu().detach(),-1,1))
                        img_grid_latent = wandb.Image(grid_latent.permute(1,2,0).numpy())
                        wandb.log({f"Latent_{latent}": img_grid_latent},commit=True)

                    ## To define --> Probably fix steps with different epsilon
                    for number_of_steps in list_steps:
                        for epsilon in epsilons:
                            for _, image_index in enumerate(index_list):
                                os.makedirs(f"{directory_reconstruction_ode_latent_corruption}/{image_index}/{number_of_steps}_{round(epsilon,6)}", exist_ok = True)
                            list_of_evolution_reverse, samples = self.editing_with_ode(latent_codes, self.ddpm.unet, t_start = latent, 
                                        std_div = -1, epsilon = epsilon, steps = number_of_steps, power =0.5, 
                                        number_of_sample = number_of_sample,
                                        corrector_step = False,
                                        deep_correction=False, comparison = False)
                            samples_stacked = torch.stack(samples)
                            if self.cfg.trainer.batch_size > 1:
                                samples = torch.stack(samples_stacked.split(number_of_sample, dim=0)) 
                                for sample_index, image_index in enumerate(index_list):
                                    for m in range(number_of_sample):
                                        save_image(samples[sample_index][m].cpu()/2+0.5, 
                                            f"{directory_reconstruction_ode_latent_corruption}/{image_index}/{number_of_steps}_{round(epsilon,6)}/{m}_{self.cfg.trainer.gpu}.png")
                                if self.cfg.trainer.gpu == 0:
                                    grid_reco = make_grid(samples_stacked.cpu().detach())
                                    img_grid_reco= wandb.Image(grid_reco.permute(1,2,0).numpy())
                                    wandb.log({f"Reconstruction_l{latent}_{corruption}_e{round(epsilon,6)}_{image_index}": img_grid_reco},commit=True)
                            else:
                                for sample_index, image_index in enumerate(index_list):
                                    for m in range(number_of_sample):
                                        save_image(samples_stacked[m].cpu()/2+0.5, 
                                            f"{directory_reconstruction_ode_latent_corruption}/{image_index}/{number_of_steps}_{round(epsilon,6)}/{m}_{self.cfg.trainer.gpu}.png")
                                if self.cfg.trainer.gpu == 0:
                                    grid_reco = make_grid(samples_stacked.cpu().detach())
                                    img_grid_reco= wandb.Image(grid_reco.permute(1,2,0).numpy())
                                    wandb.log({f"Reconstruction_l{latent}_{corruption}_e{round(epsilon,6)}_{image_index}": img_grid_reco},commit=True)
                             
        return


    @torch.no_grad()
    def run_sdeedit(self, number_of_test_per_corruption = 15000,random_corruption = True, batch_size = 36, celebahq = False):
        corruptions_list = ['spatter', "motion_blur", "frost","speckle_noise","impulse_noise","shot_noise","jpeg_compression","pixelate","brightness",
                            "fog","saturate","gaussian_noise",'elastic_transform','snow','masking_vline_random_color',
                            'masking_gaussian','glass_blur','gaussian_blur','contrast',"masking_random_color"]
        
        corruption_severity  = 4
        stop_iteration_at = int(number_of_test_per_corruption/(batch_size*self.cfg.trainer.world_size))+1
        print("stop_iteration_at", stop_iteration_at)
        add_index_per_gpu = self.cfg.trainer.gpu*stop_iteration_at*batch_size
        print(f"On GPU {self.cfg.trainer.gpu} start index:", add_index_per_gpu)

        sample_step = 1
        if celebahq:
            model, betas, num_timesteps, logvar = load_model(model_id = self.cfg.trainer.model_id, device=f"cuda:{self.cfg.trainer.gpu}")
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
            directory_reconstruction =  f"{self.root}/SDE_xp/random_corruption/reconstruction_{noise_levels}/"
            directory_inputs = f"{self.root}/SDE_xp/random_corruption/inputs_{noise_levels}/"
            directory_targets = f"{self.root}/SDE_xp/random_corruption/targets_{noise_levels}/"
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



   