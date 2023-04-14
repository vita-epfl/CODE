from itertools import chain
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import Dict, List, Tuple, Any, Optional
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
from torch.distributions.categorical import Categorical
from torch.nn.utils import clip_grad_value_
from torch.nn.parallel import DistributedDataParallel

import torchaudio
import torchaudio.transforms as T
from torchaudio.functional import mu_law_encoding, mu_law_decoding, DB_to_amplitude

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
from ddpm.datasets.audio import create_dataloader, AudioDataset
from ddpm.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler, HeatDiffusionTrainer, HeatDiffusionSampler
from ddpm.model import UNet
from ddpm.score.both import get_inception_and_fid_score
import wandb
LOG = logging.getLogger(__name__)


def infiniteloop(dataloader):
    while True:
        for x, y, z in iter(dataloader):
            yield x


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


class AudioTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def setup_trainer(self) -> None:
        LOG.info(f"AudioTrainer: {self.cfg.trainer.rank}, gpu: {self.cfg.trainer.gpu}")
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.makedirs(os.path.join(self.cfg.trainer.logdir, 'sample'), exist_ok=True)
        if self.cfg.trainer.platform == "slurm":
            torchaudio.set_audio_backend('soundfile')
        self.writer = None
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

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

        if self.cfg.trainer.rank == 0:
            if self.cfg.trainer.use_clearml:
                from clearml import Task
                task = Task.init(project_name="AudioDiffusion", task_name=self.cfg.trainer.ml_exp_name)
            if self.cfg.trainer.use_wandb:
                wandb.init(project="AudioDiffusion", entity=self.cfg.trainer.wandb_entity, sync_tensorboard=True)
                wandb.run.name = self.cfg.trainer.ml_exp_name
                wandb.run.save()
            self.writer = SummaryWriter(self.cfg.trainer.logdir)
        self.sample_rate = self.cfg.dataset.sample_rate
        if self.cfg.trainer.gpu is not None:
            torch.cuda.set_device(self.cfg.trainer.gpu)

        #TODO: Fix get_dataset function for cfg usage
        self.dataset, _ = get_dataset(None, self.cfg)
        print("dataset length",len(self.dataset))
        self.dataset_test = copy.deepcopy(self.dataset)
        self.train_dataloader = create_dataloader(
                self.dataset,
                subset="train",
                eq_mode='train',
                rank=self.cfg.trainer.rank,
                max_workers=self.cfg.trainer.num_workers,
                world_size=self.cfg.trainer.world_size,
                batch_size=self.cfg.trainer.batch_size,
            )

        self.test_dataloader = create_dataloader(
                self.dataset_test,
                subset="test",
                eq_mode="test",
                rank=self.cfg.trainer.rank,
                max_workers=self.cfg.trainer.num_workers,
                world_size=self.cfg.trainer.world_size,
                batch_size=self.cfg.trainer.batch_size,
            )
        # dataloader = torch.utils.data.DataLoader(
        #     dataset, batch_size=FLAGS.batch_size, shuffle=True,
        #     num_workers=FLAGS.num_workers, drop_last=True)
        self.datalooper = infiniteloop(self.train_dataloader)
        #TODO get image size automatically from Spectogram specs
        
        # model setup
        self.net_model = UNet(
            T=self.cfg.trainer.T, ch=self.cfg.trainer.ch, ch_mult=self.cfg.trainer.ch_mult, attn=self.cfg.trainer.attn,
            num_res_blocks=self.cfg.trainer.num_res_blocks, dropout=self.cfg.trainer.dropout, input_channel=self.cfg.trainer.input_channel)
        self.ema_model = copy.deepcopy(self.net_model)
        self.optimizer = torch.optim.Adam(self.net_model.parameters(), lr=self.cfg.trainer.lr)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.warmup_lr)

        if self.cfg.trainer.checkpointpath is not None:
            try:
                ckpt = torch.load(self.cfg.trainer.checkpointpath)
                self.net_model.load_state_dict(ckpt['net_model'])
                self.ema_model.load_state_dict(ckpt['ema_model'])
                # self.optim.load_state_dict(ckpt['optim'])
                # self.sched.load_state_dict(ckpt['sched'])
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
        x_T = torch.randn(self.cfg.trainer.sample_size, 1, 256, 512)
        self.x_T = x_T.cuda(self.cfg.trainer.gpu)
        self.n_fft = 510
        self.win_length = None
        self.hop_length = 128
        # self.n_mels = 128
            # show model size

        self.spectrogram = T.Spectrogram(
                    n_fft= self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    center=True,
                    pad_mode="reflect",
                    power=self.cfg.trainer.spectrogram_power,
                    onesided=True,
            )

        self.griffin_lim = T.GriffinLim(
                 n_fft= self.n_fft,
                 n_iter = 32,
                 win_length = self.win_length,
                 hop_length = self.hop_length,
                 power = self.cfg.trainer.spectrogram_power,
                 length = self.cfg.trainer.audio_timesteps,
                #  normalized = True,
                 #norm="slaney",
                 momentum = 0.99,
                 rand_init= True).cuda(self.cfg.trainer.gpu)


        self.power_to_db = T.AmplitudeToDB()
        self.power_to_db.top_db = 80.0
        self.power_to_db.amin = 1e-6

        model_size = 0
        for param in self.net_model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))

    def db_to_power(self, x):
        return DB_to_amplitude(x, ref = 1., power = self.cfg.trainer.spectrogram_power/2.)

    def train(self) -> None:
        with trange(self.cfg.trainer.total_steps, dynamic_ncols=True) as pbar:
            for step in pbar:
                # train
                self.optimizer.zero_grad()
                start_time = time.time()
                if self.cfg.trainer.unique_img:
                    x_0 = dataset[0][0].unsqueeze(0)
                else:
                    x_0 = next(self.datalooper)
                shape = x_0.shape
                
                loading_time = time.time() - start_time
                start_time = time.time()
                if self.cfg.trainer.use_spectrogram:
                    x_spec = self.spectrogram(x_0.squeeze(1)).unsqueeze(1)
                    x_0 = self.power_to_db(x_spec)
                if self.x_T.shape != x_0.shape:
                    self.x_T = torch.randn_like(x_0)
                x_0 = x_0.cuda(self.cfg.trainer.gpu)

                spect_time = time.time() - start_time
                start_time = time.time()

                if self.half_precision:
                    with torch.cuda.amp.autocast(enabled = True):
                        loss = self.diffusion_trainer(x_0).mean()
                        diffusion_time = time.time() - start_time
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.sched.step()
                else:
                    loss = self.diffusion_trainer(x_0).mean()
                    diffusion_time = time.time() - start_time
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.net_model.parameters(), self.cfg.trainer.grad_clip)
                    self.optimizer.step()
                    self.sched.step()
            
                ema(self.net_model, self.ema_model, self.cfg.trainer.ema_decay)

                # log
                if self.writer is not None:
                    self.writer.add_scalar('spect_time', spect_time, step)
                    self.writer.add_scalar('diffusion_time', diffusion_time, step)
                    self.writer.add_scalar('loading_time', loading_time, step)
                    self.writer.add_scalar('loss', loss, step)
                pbar.set_postfix(loss='%.3f' % loss)

                # sample
                
                if self.cfg.trainer.sample_step > 0 and step % self.cfg.trainer.sample_step == 0:
                    inputs_original_img = x_0.transpose(1,3).transpose(1,2).cpu().numpy()
                    power_spectrogram = self.db_to_power(x_0)
                    audio_original = self.griffin_lim(power_spectrogram)

                    self.net_model.eval()
                    with torch.no_grad():
                        if self.half_precision:
                            with torch.cuda.amp.autocast():
                                x_0 = self.ema_sampler(self.x_T)
                                print(x_0)
                        else:
                            x_0 = self.ema_sampler(self.x_T)
                        x_spec = self.db_to_power(x_0[0])
                        audio_reconstructed = self.griffin_lim(x_spec)
                        if self.writer is not None:
                            self.writer.add_audio(
                            f"Audio_inputs_{step}",
                            audio_original[0].squeeze().detach().cpu(), 
                            sample_rate=self.sample_rate
                        )
                            self.writer.add_audio(
                                f"Audio_Generated_{step}",
                                audio_reconstructed.squeeze().detach().cpu(), 
                                sample_rate=self.sample_rate
                                )
                        x_0 = x_0.transpose(1,3).transpose(1,2).cpu()
                        path = os.path.join(
                            self.cfg.trainer.logdir, 'sample', '%d.png' % step)
                        plt.imshow(x_0[0].numpy())
                        plt.savefig(path)

                        if self.cfg.trainer.use_wandb and self.cfg.trainer.rank == 0 and self.cfg.trainer.rank == 0:
                            images_orig = wandb.Image(inputs_original_img[0])
                            spec_ori = wandb.Image(power_spectrogram.transpose(1,3).transpose(1,2).cpu().numpy()[0])
                            wandb.log({"Original_Inputs": images_orig}) 
                            wandb.log({"Original_Spec": spec_ori})
                            image_sample = wandb.Image(x_0.numpy()[0])
                            wandb.log({"One_Sample": image_sample})                                
                    self.net_model.train()
                    print("Logging samples finished.")
                dist.barrier()

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
                    pbar.write(
                        "%d/%d " % (step, self.cfg.trainer.total_steps) +
                        ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
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