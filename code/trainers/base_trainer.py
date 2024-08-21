from abc import abstractmethod
import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda import init
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torchaudio
import hashlib
import subprocess
import submitit
from tqdm import tqdm
import time
import datetime
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from code.utils.utils import fix_random_seeds

LOG = logging.getLogger(__name__)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_local_single_gpu_mode(cfg: DictConfig) -> DictConfig:
    with open_dict(cfg):
        cfg.trainer.rank = 0
        cfg.trainer.world_size = 1
        cfg.trainer.gpu = 0 
        torch.cuda.set_device(cfg.trainer.gpu)
    return cfg


def init_local_distributed_mode(cfg: DictConfig) -> DictConfig:
    with open_dict(cfg):  # add to config
        # launched with torch.distributed.launch locally
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            cfg.trainer.rank = int(os.environ["RANK"])
            cfg.trainer.world_size = int(os.environ["WORLD_SIZE"])
            cfg.trainer.gpu = int(os.environ["LOCAL_RANK"])
        elif torch.cuda.is_available():
            LOG.info("Will run the code on one GPU.")
            # Naive launch
            # we manually add MASTER_ADDR and MASTER_PORT to env variables
            cfg.trainer.rank, cfg.trainer.gpu, cfg.trainer.world_size = 0, 0, 1

            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29400"
        else:
            LOG.error("Does not support training without GPU.")
            sys.exit(1)

    # LOG.info(f'Init process group: {cfg.trainer}')
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=cfg.trainer.world_size,
            timeout=datetime.timedelta(seconds=1800),
            rank=cfg.trainer.rank,
        )
    except:
        os.environ["MASTER_PORT"] = "29401"
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=cfg.trainer.world_size,
            timeout=datetime.timedelta(seconds=1800),
            rank=cfg.trainer.rank,
        )

    torch.cuda.set_device(cfg.trainer.gpu)
    # LOG.info('| distributed init (rank {}): {}'.format(cfg.trainer.rank, cfg.trainer.gpu))
    dist.barrier()
    return cfg


class BaseTrainer(object):
    current_epoch = 0
    model = None
    optimizer = None

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def setup_local(self) -> None:
        torchaudio.set_audio_backend("sox_io")
        self.cfg = init_local_distributed_mode(self.cfg)

    def setup_local_single_gpu(self) -> None:
        self.cfg = init_local_single_gpu_mode(self.cfg)

    def run(self) -> None:
        # self.train()
        if self.cfg.trainer.mode == "eval":
            self.eval()
        elif self.cfg.trainer.mode == "train":
            self.train()
        else:
            raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def setup_trainer(self) -> None:
        pass

    def setup_platform(self) -> None:
        if self.cfg.trainer.random_seed:
            seed = np.random.randint(10000)
            fix_random_seeds(seed)
            with open_dict(self.cfg):
                self.cfg.trainer.seed = seed
        else:
            fix_random_seeds(self.cfg.trainer.seed)
        if self.cfg.trainer.platform == "local":
            LOG.info(f"Training platform : {self.cfg.trainer.platform}")
            if self.cfg.trainer.single_gpu:
                self.setup_local_single_gpu()
            else:
                self.setup_local()
        elif self.cfg.trainer.platform == "slurm":
            LOG.info(f"Training platform : {self.cfg.trainer.platform}")
            self.setup_slurm()
        else:
            raise NotImplementedError("Unknown platform (valid value are local or slurm)")

    @abstractmethod
    def eval(self) -> None:
        pass

    # https://github.com/facebookincubator/submitit/blob/main/docs/checkpointing.md
    def __call__(self) -> None:
        self.setup_platform()
        self.setup_trainer()
        self.run()

    def setup_slurm(self) -> None:
        # torchaudio.set_audio_backend("soundfile")
        torchaudio.set_audio_backend("sox_io")
        job_env = submitit.JobEnvironment()
        with open_dict(self.cfg):
            self.cfg.trainer.job_id = job_env.job_id
            self.cfg.trainer.gpu = job_env.local_rank
            self.cfg.trainer.rank = job_env.global_rank
            self.cfg.trainer.world_size = job_env.num_tasks
        LOG.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        LOG.error(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        ### Master address & Port
        if "MASTER_PORT" in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            # from PyTorch Lightning
            default_port = os.environ.get("SLURM_JOB_ID")
            job_id = default_port
            if default_port:
                # use the last 4 numbers in the job id as the id
                default_port = default_port[-4:]
                # all ports should be in the 10k+ range
                default_port = int(default_port) + 15000
            else:
                default_port = 12910
            os.environ["MASTER_PORT"] = str(default_port)

        # Master Addr
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(self.cfg.trainer.world_size)  # str(ntasks)
        os.environ["LOCAL_RANK"] = str(self.cfg.trainer.gpu)  # str(proc_id % num_gpus)
        os.environ["RANK"] = str(self.cfg.trainer.rank)  # str(proc_id)
        print("WORLD SIZE :", self.cfg.trainer.world_size)
        print("LOCAL_RANK :", self.cfg.trainer.gpu)
        print("RANK :", self.cfg.trainer.rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.cfg.trainer.world_size,
            rank=self.cfg.trainer.rank,
        )
        torch.cuda.set_device(self.cfg.trainer.gpu)
        dist.barrier()


    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        print("Requeuing SLURM job", OmegaConf.to_yaml(self.cfg))
       
        if (self.model is not None) and (self.optimizer is not None):
            self.checkpoint_dump(
                checkpoint_path=self.cfg.trainer.checkpointpath,
                epoch=self.current_epoch,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
            )
        else:
            pass
        empty_trainer = type(self)(self.cfg)
        print("Sending Delayed Submission...")
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def checkpoint_dump(
        self,
        checkpoint_path: str = None,
        epoch: int = 0,
        model_state_dict: Dict = None,
        optimizer_state_dict: Dict = None,
        **kwargs,
    ) -> None:
        if (model_state_dict is None) and (self.model is not None):
            model_state_dict = self.model.state_dict()

        if (optimizer_state_dict is None) and (self.optimizer is not None):
            optimizer_state_dict = self.optimizer.state_dict()

        if checkpoint_path is None:
            prefix = self.cfg.trainer.output_dir
            # if self.cfg.trainer.platform == "local": # Hydra changes base directory
            #     prefix = ''
            checkpoint_path = os.path.join(prefix, "default_checkpoint.pt")
            # if epoch == 0:
            #     checkpoint_path = os.path.join(prefix, "default_checkpoint.pt")
            # else:
            #     checkpoint_path = os.path.join(prefix, f"checkpoint_epoch_{str(epoch)}.pt")
        
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                **kwargs
            },
            checkpoint_path,
        )
        with open_dict(self.cfg):
            self.cfg.trainer.checkpointpath = checkpoint_path

    def checkpoint_load(self, checkpoint_path: Union[str, Path]) -> Optional[Dict]:
        if not checkpoint_path:
            return None 
        if not Path(checkpoint_path).exists():
            return None
        return torch.load(str(checkpoint_path))

    def find_latest_checkpoint_path(self, output_dir: Union[str, Path]) -> Optional[Path]:
        p = Path(output_dir)
        if not p.exists():
            return None
        checkpoints = [x for x in p.iterdir() if x.is_file() and x.suffix == '.pt']
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]), reverse=True)
        return checkpoints[0]
