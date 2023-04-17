from itertools import chain
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import Dict, List, Tuple, Any, Optional, Callable
import copy
import warnings 
import time
import matplotlib.pyplot as plt

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
from ddpm.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler, HeatDiffusionTrainer, HeatDiffusionSampler
from ddpm.model import UNet
from ddpm.score.both import get_inception_and_fid_score
import wandb
LOG = logging.getLogger(__name__)




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


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


class Cityscape_Trainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)


    def setup_trainer(self) -> None:
        if self.cfg.trainer.use_half_for_matmul:
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False

        if self.cfg.trainer.use_half_for_conv:
            torch.backends.cudnn.allow_tf32 = True
        else:
            torch.backends.cudnn.allow_tf32 = False

        if self.cfg.trainer.use_half_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            self.half_precision = True
        else:
            self.half_precision = False
        print("Using fp16 precision:", self.half_precision)
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
                wandb.init(project="Diffusion_Dataset", entity=self.cfg.trainer.wandb_entity, sync_tensorboard=True)
                wandb.run.name = self.cfg.trainer.ml_exp_name
                wandb.run.save()
            self.writer = SummaryWriter(self.cfg.trainer.logdir)

        if self.cfg.trainer.gpu is not None:
            torch.cuda.set_device(self.cfg.trainer.gpu)

        self.dataset, self.dataset_test = get_dataset(None, self.cfg)
        print("train dataset length",len(self.dataset))
        

        self.train_dataloader = create_dataloader(
                self.dataset,
                rank=self.cfg.trainer.rank,
                max_workers=self.cfg.trainer.num_workers,
                world_size=self.cfg.trainer.world_size,
                batch_size=self.cfg.trainer.batch_size,
            )
        self.datalooper = infiniteloop(self.train_dataloader)
        
        if self.dataset_test is not None:
            print("test dataset length",len(self.dataset_test))
            self.test_dataloader = create_dataloader(
                    self.dataset_test,
                    rank=self.cfg.trainer.rank,
                    max_workers=self.cfg.trainer.num_workers,
                    world_size=self.cfg.trainer.world_size,
                    batch_size=self.cfg.trainer.batch_size,
                )
        else:
            self.test_dataloader = self.train_dataloader
            self.dataset_test = self.dataset
        # model setup
        self.net_model = UNet(
            T=self.cfg.trainer.T, ch=self.cfg.trainer.ch, ch_mult=OmegaConf.to_object(self.cfg.trainer.ch_mult), 
                attn=OmegaConf.to_object(self.cfg.trainer.attn),
                num_res_blocks=self.cfg.trainer.num_res_blocks, dropout=self.cfg.trainer.dropout, 
                input_channel=self.cfg.trainer.input_channel, kernel_size=self.cfg.trainer.kernel_size)

        self.ema_model = copy.deepcopy(self.net_model)
        self.optimizer = torch.optim.Adam(self.net_model.parameters(), lr=self.cfg.trainer.lr)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.warmup_lr)

        if self.cfg.trainer.checkpointpath is not None:
            try:
                ckpt = torch.load(self.cfg.trainer.checkpointpath)
                self.net_model.load_state_dict(ckpt['net_model'])
                self.ema_model.load_state_dict(ckpt['ema_model'])
            except Exception as e:
                print(e)

        self.diffusion_trainer = GaussianDiffusionTrainer(
            self.net_model, self.cfg.trainer.beta_1, self.cfg.trainer.beta_T, self.cfg.trainer.T).cuda(self.cfg.trainer.gpu)
        # trainer = HeatDiffusionTrainer(
        #     net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size).to(device)

        # net_sampler = HeatDiffusionSampler(
        #     net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size).to(device)

        self.net_sampler = GaussianDiffusionSampler(
            self.net_model, self.cfg.trainer.beta_1, self.cfg.trainer.beta_T, self.cfg.trainer.T, self.cfg.trainer.img_size,
            self.cfg.trainer.mean_type, self.cfg.trainer.var_type).cuda(self.cfg.trainer.gpu)

        # net_sampler = HeatDiffusionSampler(
        #     net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size).to(device)
            
        # ema_sampler = HeatDiffusionSampler(
        #     ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size).to(device)

        self.ema_sampler = GaussianDiffusionSampler(
            self.ema_model, self.cfg.trainer.beta_1, self.cfg.trainer.beta_T, self.cfg.trainer.T, self.cfg.trainer.img_size,
            self.cfg.trainer.mean_type, self.cfg.trainer.var_type).cuda(self.cfg.trainer.gpu)

        self.diffusion_trainer = DistributedDataParallel(self.diffusion_trainer, device_ids=[self.cfg.trainer.gpu])
        self.net_sampler = DistributedDataParallel(self.net_sampler, device_ids=[self.cfg.trainer.gpu])
        self.ema_sampler = DistributedDataParallel(self.ema_sampler, device_ids=[self.cfg.trainer.gpu])

        # log setup 
        #TODO: automate the size
        x_T = torch.randn(self.cfg.trainer.sample_size, self.cfg.trainer.input_channel, self.cfg.trainer.img_size[0], self.cfg.trainer.img_size[1])
        self.x_T = x_T.cuda(self.cfg.trainer.gpu)

    
        # show model size
        model_size = 0
        for param in self.net_model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))


    def train(self,) -> None:
        self.net_model.train()
        self.optimizer.zero_grad()
        previous_loss = 1.
        for step in range(self.cfg.trainer.total_steps):
            start_time = time.time()
            x_0, _ = next(self.datalooper)
            
            loading_time = time.time() - start_time

            if self.x_T[0].shape != x_0[0].shape:
                print(f"Issue with x_T shape, {self.x_T.shape} but {x_0.shape} needed.")
                self.x_T = torch.randn_like(x_0).cuda(self.cfg.trainer.gpu)

            x_0 = x_0.cuda(self.cfg.trainer.gpu)
            start_time = time.time()

            if self.half_precision:
                with torch.cuda.amp.autocast():
                    loss = self.diffusion_trainer(x_0).mean() / self.cfg.trainer.accumulating_step
                    diffusion_time = time.time() - start_time
                self.scaler.scale(loss).backward()
            else:
                loss = self.diffusion_trainer(x_0).mean() / self.cfg.trainer.accumulating_step
                diffusion_time = time.time() - start_time
                loss.backward()

            if (previous_loss < 80*loss.data.cpu().item()) and (step >0):
                if self.writer is not None:
                    grid_ori_pb = make_grid(x_0) #(make_grid(x_0) + 1) / 2
                    img_grid_ori_pb = wandb.Image(grid_ori_pb.permute(1,2,0).cpu().numpy())
                    wandb.log({"Problem_Image": img_grid_ori_pb}) 

            previous_loss = loss.data.cpu().item()

            if (step + 1) % self.cfg.trainer.accumulating_step == 0:
                if self.half_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.sched.step()
                else:
                    torch.nn.utils.clip_grad_norm_(self.net_model.parameters(), self.cfg.trainer.grad_clip)
                    self.optimizer.step()
                    self.sched.step()
                self.optimizer.zero_grad()

            ema(self.net_model, self.ema_model, self.cfg.trainer.ema_decay)

            # log
            if self.writer is not None:
                self.writer.add_scalar('diffusion_time', diffusion_time, step)
                self.writer.add_scalar('loading_time', loading_time, step)
                self.writer.add_scalar('loss', (loss*self.cfg.trainer.accumulating_step).data.cpu().item(), step)

                if step % 1000 == 0:
                    print(f"Step_{step},  loss :",(loss*self.cfg.trainer.accumulating_step).data.cpu().item())
            # sample
            if self.cfg.trainer.sample_step > 0 and step % self.cfg.trainer.sample_step == 0:                
                self.net_model.eval()
                if self.writer is not None:
                    grid_ori = make_grid(x_0) #(make_grid(x_0) + 1) / 2
                    img_grid_ori = wandb.Image(grid_ori.permute(1,2,0).cpu().numpy())
                    wandb.log({"Original_Image": img_grid_ori}) 

                if step > 0 :
                    with torch.no_grad():
                        if self.half_precision:
                            with torch.cuda.amp.autocast():
                                x_0 = self.ema_sampler(self.x_T)
                        else:
                            x_0 = self.ema_sampler(self.x_T)
                        grid = make_grid(x_0)
                        path = os.path.join(
                            self.cfg.trainer.logdir, 'sample', '%d.png' % step)
                        if self.writer is not None:
                            save_image(grid, path)
                            # self.writer.add_image('Sample', grid, step)
                            img_grid = wandb.Image(grid.permute(1,2,0).cpu().numpy())
                            wandb.log({"Sample_Grid": img_grid})
                self.net_model.train()

            # save
            if self.cfg.trainer.save_step > 0 and step % self.cfg.trainer.save_step == 0:
                ckpt = {
                    'net_model': self.net_model.state_dict(),
                    'ema_model': self.ema_model.state_dict(),
                    'sched': self.sched.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'step': step,
                    'x_T': self.x_T,
                }
                torch.save(ckpt, os.path.join(self.cfg.trainer.logdir, f'ckpt_{step}.pt'))

            # evaluate
            if self.cfg.trainer.eval_step > 0 and step % self.cfg.trainer.eval_step == 0 and step > 0:
                net_IS, net_FID, _ = self.evaluate(self.net_sampler, self.net_model)
                ema_IS, ema_FID, _ = self.evaluate(self.ema_sampler, self.ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                # pbar.write(
                #     "%d/%d " % (step, self.cfg.trainer.total_steps) +
                #     ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                if self.writer is not None:
                    for name, value in metrics.items():
                        self.writer.add_scalar(name, value, step)
                    self.writer.flush()
                with open(os.path.join(self.cfg.trainer.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")

        if self.writer is not None:
            self.writer.close()

    def eval(self):
        # model setup
        model = UNet(
            T=self.cfg.trainer.T, ch=self.cfg.trainer.ch, ch_mult=self.cfg.trainer.ch_mult, attn=self.cfg.trainer.attn,
            num_res_blocks=self.cfg.trainer.num_res_blocks, dropout=self.cfg.trainer.dropout)
        sampler = GaussianDiffusionSampler(
            model, self.cfg.trainer.beta_1, self.cfg.trainer.beta_T, self.cfg.trainer.T, img_size=self.cfg.trainer.img_size,
            mean_type=self.cfg.trainer.mean_type, var_type=self.cfg.trainer.var_type).cuda(self.cfg.trainer.gpu)
        
        sampler = DistributedDataParallel(sampler, device_ids=[self.cfg.trainer.gpu])

        # load model and evaluate
        ckpt = torch.load(os.path.join(self.cfg.trainer.logdir, 'ckpt.pt'))
        model.load_state_dict(ckpt['net_model'])
        (IS, IS_std), FID, samples = self.evaluate(sampler, model)
        print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        save_image(
            torch.tensor(samples[:256]),
            os.path.join(self.cfg.trainer.logdir, 'samples.png'),
            nrow=16)

        model.load_state_dict(ckpt['ema_model'])
        (IS, IS_std), FID, samples = self.evaluate(sampler, model)
        print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        save_image(
            torch.tensor(samples[:256]),
            os.path.join(self.cfg.trainer.logdir, 'samples_ema.png'),
            nrow=16)

    def warmup_lr(self, step):
        return min(step, self.cfg.trainer.warmup) / self.cfg.trainer.warmup


    def evaluate(self, sampler, model):
        model.eval()
        with torch.no_grad():
            images = []
            desc = "generating images"
            for i in trange(0, self.cfg.trainer.num_images, self.cfg.trainer.batch_size, desc=desc):
                batch_size = min(self.cfg.trainer.batch_size, self.cfg.trainer.num_images - i)
                x_T = torch.randn((batch_size, 3, self.cfg.trainer.img_size, self.cfg.trainer.img_size))
                batch_images = sampler(x_T.cuda(self.cfg.trainer.gpu)).cpu()
                images.append((batch_images + 1) / 2)
            images = torch.cat(images, dim=0).numpy()
        model.train()
        (IS, IS_std), FID = get_inception_and_fid_score(
            images, self.cfg.trainer.fid_cache, num_images=self.cfg.trainer.num_images,
            use_torch=self.cfg.trainer.fid_use_torch, verbose=True)
        return (IS, IS_std), FID, images

