import pathlib
import urllib
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torchaudio.transforms as T
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor
from torch.utils.data import Sampler, WeightedRandomSampler
import torch.distributed as dist
import gc
import math

class WeightedDistributedSampler(Sampler):
    def __init__(self, dataset,weights = None, targets = None, nb_labels = None,  
                num_replicas=None, rank=None, drop_last = False, shuffle : bool = True, seed: int = 0, replacement=True):   
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed
        self.replacement = replacement
        self.weights = weights
        self.targets = targets
        self.nb_labels = nb_labels
        if targets is not None:
            self.use_targets = True
            assert len(self.targets) == len(dataset)
        elif weights is not None:
            self.use_targets = False
            assert len(self.weights) == len(dataset)
        # else:
        #     raise NotImplementedError

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        self.indices = indices[self.rank:self.total_size:self.num_replicas]
        targets = self.dataset.get_labels()
        weights, num_classes = self.calculate_weights(targets)
        self.targets = targets[self.indices]
        self.weights = weights[self.indices]
        self.num_classes = num_classes
        # else:
        #     weights = self.weights[indices]

        assert len(self.weights) == len(self.indices)
        assert len(self.indices) == self.num_samples

        # self.weights = weights

    def calculate_weights(self, targets = None):

        # print("Shape targets",targets[0])
        nb_per_class = targets.sum(0)
        # print("nb shape", nb_per_class)
        weights_per_class = 1/nb_per_class
        # print("weights_per_class", weights_per_class)
        weights_per_class = torch.nan_to_num(weights_per_class, nan=0.0, posinf=0.0, neginf=0.0)
        num_classes = (weights_per_class > 0).sum()
        # print("weights_per_class", weights_per_class)
        weights = torch.matmul(targets.float(), weights_per_class.float())
        
        # print("Weights", weights)   
        return weights , num_classes

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement)
        yield from iter(rand_tensor.tolist())
    # def __iter__(self):
    #     return iter(torch.multinomial(self.weights, 1, self.replacement).tollist())
    #     # return iter(indices)
    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def get_output_dir(cfg, job_id=None) -> Path:
    out_dir = Path(cfg.trainer.logdir)
    exp_name = str(cfg.trainer.ml_exp_name)
    folder_name = exp_name+'_'+str(job_id)
    p = Path(out_dir).expanduser()
    if job_id is not None:
        # p = p / str(job_id)
        p = p / folder_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def add_key_value_to_conf(cfg: DictConfig, key: Any, value: Any) -> DictConfig:
    with open_dict(cfg):
        cfg[key] = value
    return cfg


def fix_random_seeds(seed: int = 31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def file_uri_to_path(file_uri: str, path_class=Path) -> Path:
    # https://stackoverflow.com/questions/5977576/is-there-a-convenient-way-to-map-a-file-uri-to-os-path
    """
    This function returns a pathlib.PurePath object for the supplied file URI.

    :param str file_uri: The file URI ...
    :param class path_class: The type of path in the file_uri. By default it uses
        the system specific path pathlib.PurePath, to force a specific type of path
        pass pathlib.PureWindowsPath or pathlib.PurePosixPath
    :returns: the pathlib.PurePath object
    :rtype: pathlib.PurePath
    """
    windows_path = isinstance(path_class(), pathlib.PureWindowsPath)
    file_uri_parsed = urllib.parse.urlparse(file_uri)
    file_uri_path_unquoted = urllib.parse.unquote(file_uri_parsed.path)
    if windows_path and file_uri_path_unquoted.startswith("/"):
        result = path_class(file_uri_path_unquoted[1:])
    else:
        result = path_class(file_uri_path_unquoted)
    if result.is_absolute() == False:
        raise ValueError("Invalid file uri {} : resulting path {} not absolute".format(file_uri, result))
    return result


def print_stats(waveform: Tensor, sample_rate: int = None, src: str = None) -> None:
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    w = waveform.float()
    print(f" - Max:     {w.max().item():6.3f}")
    print(f" - Min:     {w.min().item():6.3f}")
    print(f" - Mean:    {w.mean().item():6.3f}")
    print(f" - Std Dev: {w.std().item():6.3f}")
    print()
    print(waveform)
    print()


def play_audio_jupyter(waveform: Tensor, sample_rate: int) -> Any:
    from IPython.display import Audio, display

    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


def play_audio_vscode(waveform: Tensor, sample_rate: int) -> Any:
    # https://github.com/microsoft/vscode-jupyter/issues/1012
    import json

    import IPython.display
    import numpy as np

    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        channels = [waveform.tolist()]
    else:
        channels = waveform.tolist()

    return IPython.display.HTML(
        """
        <script>
            if (!window.audioContext) {
                window.audioContext = new AudioContext();
                window.playAudio = function(audioChannels, sr) {
                    const buffer = audioContext.createBuffer(audioChannels.length, audioChannels[0].length, sr);
                    for (let [channel, data] of audioChannels.entries()) {
                        buffer.copyToChannel(Float32Array.from(data), channel);
                    }
            
                    const source = audioContext.createBufferSource();
                    source.buffer = buffer;
                    source.connect(audioContext.destination);
                    source.start();
                }
            }
        </script>
        <button onclick="playAudio(%s, %s)">Play</button>
    """
        % (json.dumps(channels), sample_rate)
    )

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True

def zero_grad(model):
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None

def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()

def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_state(model):
    return sum(s.numel() for s in model.state_dict().values())