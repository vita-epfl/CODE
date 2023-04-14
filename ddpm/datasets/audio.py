import logging
import os
from pathlib import Path
from syslog import LOG_LOCAL0
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union
import copy
import time

import math
from unittest import result
import numpy as np
from pandas._config.config import options
import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.functional import norm
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from numpy.random import SeedSequence, default_rng
from ddpm.datasets.transform import signal

from tqdm import tqdm

LOG = logging.getLogger(__name__)


@dataclass
class AudioClip:
    waveform: Tensor
    sample_rate: int
    num_channels: int
    path: str
    name: str
    frame_offset: int = 0  # original frame offset before resampling
    num_frames: int = -1  # original num_frames before resampling


    def pin_memory(self):
        if self.waveform is not None:
            self.waveform = self.waveform.pin_memory()
        return self


def split_dataset(
    indices: List[int], train_perc: float, validate_perc: float, rng: np.random.Generator
) -> Tuple[List[int], List[int], List[int]]:
    assert 0 < train_perc < 1
    assert 0 <= validate_perc < 1
    assert train_perc >= validate_perc
    assert train_perc + validate_perc < 1

    rng.shuffle(indices)

    train_idx = int(train_perc * len(indices))
    valid_idx = int((1 - validate_perc) * len(indices))
    train, test, validate = np.split(indices, [train_idx, valid_idx])
    return train, validate, test

def split_dataframe(
    df: pd.DataFrame, train_perc: float, validate_perc: float, rng: np.random.Generator
) -> Tuple[List[int], List[int], List[int]]:

    train, validate, test = split_dataset(df.index.values, train_perc, validate_perc, rng)
    df.insert(loc=0, column="subset", value="")
    df.loc[train, "subset"] = "train"
    df.loc[validate, "subset"] = "validate"
    df.loc[test, "subset"] = "test"
    df.reset_index(inplace=True)
    return df


def load_audio_clip(
    filepath: str, mono: bool = False, frame_offset: int = 0, num_frames: int = -1, normalize: bool = True
) -> Optional[AudioClip]:
    """
    Loads a file from the dataset and returns the raw waveform
    as a Torch Tensor, its sample rate as an integer, and its
    name a string.
    """
    try:
        start_time = time.time()
        waveform, sample_rate = torchaudio.load(filepath, frame_offset, num_frames, normalize)
        # print(sample_rate)
        # waveform, sample_rate = torchaudio.load(filepath, False)
        # print('waveform', waveform.shape)
        # print('loading time first',time.time()-start_time )
        # waveform = waveform[:,frame_offset:frame_offset+num_frames]
        if mono and waveform.shape[0] != 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
    except RuntimeError as e:
        LOG.error(f"{e} - {filepath}")
        return None

    name = Path(filepath).name
    clip = AudioClip(waveform, sample_rate, waveform.shape[0], filepath, name, frame_offset, waveform.shape[1])
    return clip


def get_audio_metadata(path: str, slow_mode: bool = False) -> Optional[torchaudio.backend.common.AudioMetaData]:
    metadata = None
    try:
        metadata = torchaudio.info(path)
        if slow_mode:
            waveform, _ = torchaudio.load(path)
            metadata.num_frames = waveform[0].size(0)
    except RuntimeError as e:
        LOG.error(e)
    return metadata


def convert_metadata_to_dict(path: str, metadata: Any) -> Dict:
    row = {
        "name": Path(path).stem,
        "sample_rate": metadata.sample_rate,
        "num_frames": metadata.num_frames,
        "num_channels": metadata.num_channels,
        "path": path,
    }
    return row


def find_audio_files(
    root_dir: str,
    allowed_ext: str,
    skip_dir: str = None,
    match_sample_rate: int = None,
    slow_mode: bool = False
) -> Optional[pd.DataFrame]:

    """
    Return a DataFrame of audio info
    """
    data = []
    if not root_dir:
        return None

    root_dir = Path(root_dir).expanduser()

    # If skip dir is a child of the root dir
    skip_dir = Path(skip_dir).expanduser()
    skip_name = None
    if skip_dir.parent == root_dir:
        skip_name = skip_dir.stem

    for root, dirs, files in tqdm(os.walk(root_dir, topdown=True)):
        dirs[:] = [d for d in dirs if d != skip_name]
        for name in files:
            (_, ext) = os.path.splitext(name)  # split base and extension
            if ext[1:] not in allowed_ext:  # check the extension
                continue

            fullpath = os.path.join(root, name)  # create full path
            metadata = get_audio_metadata(fullpath, slow_mode)
            if not metadata:
                continue

            if match_sample_rate:
                if metadata.sample_rate == match_sample_rate:
                    row = convert_metadata_to_dict(fullpath, metadata)
                    data.append(row)
            else:
                row = convert_metadata_to_dict(fullpath, metadata)
                data.append(row)
    if not data:
        return None

    df = pd.DataFrame.from_records(data)
    df.insert(loc=1, column="duration", value=0)
    df["duration"] = df["num_frames"] / df["sample_rate"]
    return df


def read_collated_file(path: str) -> Optional[pd.DataFrame]:
    p = Path(path).expanduser()
    if not p.exists():
        LOG.info(f"Collated file {path} doesn't exist, creating...")
        return None

    df = None
    try:
        df = pd.read_csv(p, sep=";")
        for c in df.columns:
            if c.startswith("eq"):
                df[c] = df[c].astype("string")
    except Exception as e:
        LOG.error(f"Error reading {path}, {e}")
    return df


def get_equalizer_names(cfg: DictConfig) -> Optional[List[str]]:
    for mode in ['train', 'test', 'validation']:
        return [e.name + '_' + mode for e in cfg.dataset.low_quality_effect[mode]['effects']]


def should_index_slowly(cfg):
    return cfg.trainer.platform == "slurm"  # torch audio metadata bug on slurm..


def match_lq_files(df: pd.DataFrame, cfg: DictConfig) -> Optional[pd.DataFrame]:
    def check_valid_duration(num_frames: int, target_metadata: Any):
        return target_metadata['num_frames'] == num_frames
        # return target_metadata.num_frames == num_frames
        # # print(f"Target duration : {target_duration}.")
        # if target_duration - 0.1 <= source_duration <= target_duration + 0.1:
        #     return True
        # return False

    def check_exists_valid(path: Path, source_duration: float):
        # print(f"Source duration : {int(source_duration)}")
        if Path(path).exists():
            metadata = get_audio_metadata(path, should_index_slowly(cfg))
            metadata = convert_metadata_to_dict(path, metadata)
            if check_valid_duration(source_duration, metadata):
                return path
            else:
                LOG.warning(f"Invalid duration for matched LQ file: {path}")
        return None

    def lq_path_tester(lq_folder: Path, name: str, num_frames: int, eq_names: List[str]) -> Optional[dict]:
        results = dict()

        test_file = lq_folder / (name + cfg.dataset.user_lq_suffix + "." + cfg.dataset.audio_format)
        fullpath = check_exists_valid(test_file, num_frames)
        if fullpath:
            results["eq_user"] = str(fullpath)

        for eq in eq_names:
            test_file = lq_folder / (name + "_" + eq + "." + cfg.dataset.audio_format)
            fullpath = check_exists_valid(test_file, num_frames)
            if fullpath:
                results[eq] = str(fullpath)

        return results

    data = []
    hq_folder = Path(cfg.dataset.high_quality_path).expanduser()
    lq_folder = Path(cfg.dataset.low_quality_path).expanduser()

    if not lq_folder.exists():
        return None

    eqs = get_equalizer_names(cfg)

    found_a_file = False
    for _, row in df.iterrows():
        # check if files exists with same name under the root LQ folder
        results = lq_path_tester(lq_folder, row["name"], row["num_frames"], eqs)
        if results:
            found_a_file = True
            data.append(results)
            continue

        # check if file exists under the same hierachy structure
        skip_char = len(str(hq_folder)) + 1  # account for '/'
        relative_path = row["hq_path"][skip_char:]

        if relative_path:
            test_folder = (lq_folder / relative_path).parent
            results = lq_path_tester(test_folder, row["name"], row["num_frames"], eqs)
            if results:
                found_a_file = True

        if not results:
            LOG.warning(f"Could not find matching LQ file for {row['name']}")
        data.append(results)

    if not found_a_file:
        return None

    lq_df = pd.DataFrame.from_records(data)
    df = df.join(lq_df)

    for c in df.columns:
        if c.startswith("eq"):
            df[c] = df[c].astype("string")
    return df.fillna("")


class AudioDataset(Dataset):
    df = None
    cfg = None
    subset_view = None
    chosen_set = None
    rnd = None
    chunks = None
    subset_mode = None
    eq_mode = None

    def __init__(self, cfg: DictConfig, subset: str = None, df: pd.DataFrame = None) -> None:
        super(AudioDataset).__init__()
        self.cfg = cfg
        self.subset_mode = subset
        self.seed_rng_generator()
        if df is not None:
            self.df = df
        else:
            # Use collated file if possible
            if cfg.dataset.use_collated_file:
                self.df = read_collated_file(cfg.dataset.collated_path)
                if self.df is not None:  # all good
                    self.subset_view = self.df

        self.change_subset(subset)

    def seed_rng_generator(self, worker_id=0):
        seed_sequence = SeedSequence(self.cfg.trainer.seed + worker_id)
        self.rnd = default_rng(seed_sequence)

    def reseed_random_generator(self):
        self.seed_sequence = self.seed_sequence.spawn(1)
        self.rnd = default_rng(self.seed_sequence)

    def persist_collated(self):
        self.df.to_csv(self.cfg.dataset.collated_path, sep=";", index=False)

    def should_build_dataset(self):
        return self.df is None

    def build_collated(self):
        self.subset_view = None
        sr = self.cfg.dataset.sample_rate
        if self.cfg.dataset.match_sample_rate:
            sr = self.cfg.dataset.sample_rate
        else:
            sr = False
        # No collated file or regenerate
        if self.df is None:
            print('Looking for audio files.....')
            self.df = find_audio_files(
                self.cfg.dataset.high_quality_path,
                self.cfg.dataset.audio_format,
                self.cfg.dataset.low_quality_path,
                match_sample_rate = sr,
                slow_mode = should_index_slowly(self.cfg)
            )

            if self.df is None or len(self.df) == 0:
                LOG.error(
                    f"No {self.cfg.dataset.audio_format} files found under directory: {self.cfg.dataset.high_quality_path}"
                )
                return
        else: # building from existing df
            # drop lq columns
            eq_columns = list(filter(None, map(lambda c: c if c.startswith("eq") else None, self.df.columns)))
            self.df.drop(columns=eq_columns, axis=1, inplace=True)

        LOG.debug(f"Found {len(self.df)} files under directory: {self.cfg.dataset.high_quality_path}")
        if 'path' in self.df.columns:
            self.df.rename(columns={"path": "hq_path"}, inplace=True)

        # Find corresponding LQ files
        match_df = match_lq_files(self.df, self.cfg)
        if match_df is not None:
            self.df = match_df
        else:
            if self.cfg.dataset.use_user_lq_files:
                LOG.error(
                    f"Could not find user LQ files in {self.cfg.dataset.low_quality_path} with suffix {self.cfg.dataset.user_lq_suffix}"
                )
                return
            else:
                # generate from EQ
                pass
        
        if 'subset' not in self.df.columns:
            self.df = split_dataframe(self.df, self.cfg.trainer.training_perc, self.cfg.trainer.validation_perc, self.rnd)

        # Write to file
        self.persist_collated()
        self.change_subset(self.subset_mode)

    def change_subset(self, subset: str = None) -> None:
        assert subset is None or subset in ["train", "validate", "test",], (
            "When `subset` is not None, it must be either " + "{'train', 'validate', 'test'}."
        )
        self.subset_mode = subset
        self.subset_view = None
        self.chunks = None
        self.chosen_set = subset
        if self.df is None:
            return

        if not subset:
            self.subset_view = self.df
        else:
            if "subset" in self.df:
                s = self.df["subset"] == subset
                if s.any():
                    self.subset_view = self.df.loc[s]
        self.subset_mode = subset #train, validation, test
        return self.subset_view

    def __len__(self) -> int:
        if self.subset_view is None:
            return 0
        return len(self.subset_view)

    def get_sample(self, filepath: str, frame_offset: int = 0, num_frames: int = -1, normalize_gain: bool = True) -> Optional[AudioClip]:
        start_time = time.time()
        clip = load_audio_clip(filepath, mono=self.cfg.dataset.mono, frame_offset=frame_offset, num_frames=num_frames)
        # print("waveform",clip.waveform.shape)
        if not clip:  # error loading the file
            return None
        ### to see why  it is happening
        if clip.waveform.numpy().size == 0:
            raise StopIteration

        if clip.sample_rate != self.cfg.dataset.sample_rate:
            waveform, sample_rate = signal.resample_signal(
                clip.waveform, clip.sample_rate, self.cfg.dataset.sample_rate
            )
            ####TODO : check only in the case of shorter file
            if waveform.shape[-1] < self.get_random_sample_duration() * sample_rate:
                # print("before padding", waveform.shape)
                pad = (0,self.get_random_sample_duration() * sample_rate - waveform.shape[-1])
                waveform = torch.nn.functional.pad(waveform, pad, mode='constant', value=0)
                # print("after padding", waveform.shape)
            clip.waveform = waveform
            clip.sample_rate = sample_rate

        if normalize_gain:
            sox = signal.SoxEffectTransform()
            sox.normalize_gain()
            clip.waveform, clip.sample_rate = sox.apply_tensor(clip.waveform, self.cfg.dataset.sample_rate)

        return clip

    def get_hq_sample(self, n: int, frame_offset: int = 0, num_frames: int = -1) -> Optional[AudioClip]:
        if n >= len(self.subset_view):
            LOG.error(f"Cannot get HQ sample at index {n}, only {len(self.subset_view)} files")
            return None

        filepath = self.subset_view.iloc[n]["hq_path"]
        
        return self.get_sample(filepath, frame_offset, num_frames)

    def get_random_hq_sample(self, n: int, duration: float = 1) -> Optional[AudioClip]:
        if n >= len(self.subset_view):
            LOG.error(f"Cannot get HQ sample at index {n}, only {len(self.subset_view)} files")
            return None
        serie = self.subset_view.iloc[n][["hq_path", "sample_rate", "num_frames"]]

        sample_rate = serie.sample_rate
        filepath = serie.hq_path
        total_frames = serie.num_frames

        if duration * sample_rate >= total_frames:
            LOG.warning(f"Requested random sample length is longer than overall sample")
            return self.get_hq_sample(n)
        # if self.cfg.trainer.audio_timesteps >= total_frames:
        #     LOG.warning(f"Requested random sample length is longer than overall sample")
        #     return self.get_hq_sample(n)

        frames = duration * sample_rate
        # frames = self.cfg.trainer.audio_timesteps
        high_bound = total_frames - frames - 1
        frame_offset = self.rnd.integers(0, high_bound)
        return self.get_sample(filepath, frame_offset, frames)

    def get_lq_sample(
        self,
        n: int,
        eq: str = "eq_user",
        frame_offset: int = 0,
        num_frames: int = -1,
        hq_clip: AudioClip = None,
        persist_eq: bool = False,
    ) -> Optional[AudioClip]:

        if n >= len(self.subset_view):
            LOG.error(f"Cannot get LQ sample at index {n}, only {len(self.subset_view)} files")
            return None

        serie = self.subset_view.iloc[n].fillna("")
        lq_path = None
        if hq_clip:
            num_frames = hq_clip.num_frames
            frame_offset = hq_clip.frame_offset
        
        # LQ file on disk exists and has the same sample rate
        if eq == "eq_user":
            if "eq_user" in serie and serie.eq_user and Path(serie.eq_user).exists():
                lq_path = serie.eq_user
                return self.get_sample(lq_path, frame_offset, num_frames)
            else:
                if not self.cfg.dataset.fallback_generated_eqs:
                    LOG.error(
                        f'User defined low-quality sample for \
                        sample {serie.name}: {serie.hq_path} does not exist. \
                        Set "cfg.dataset.use_user_lq_files" to False and/or fallback_generated_eqs to True or index the LQ file.'
                    )
                return None

        # Load LQ from disk
        if eq in serie:
            lq_path = serie[eq]
            if lq_path and Path(lq_path).exists():
                print("Loaded LQ from the disk...")
                return self.get_sample(lq_path, frame_offset, num_frames)

        # Generate EQ
        if self.eq_mode == 'random':
            eq_transform = [signal.SoxEffectTransform.random_effects(self.cfg)]
            eq = "random"
        else:
            transforms = signal.SoxEffectTransform.from_config(self.cfg)
            eq_transform = list(filter(lambda x: x.name == eq, transforms[self.eq_mode]))

        if not eq_transform:
            LOG.error(f"Low-quality sample with name {eq} cannot be found or generated")
            return None

        eq_transform = eq_transform[0]
        # Generate from input HQ clip
        if hq_clip:
            lq_wave, lq_sample_rate = eq_transform.apply_tensor(hq_clip.waveform, self.cfg.dataset.sample_rate)
            lq_wave = lq_wave[:,:hq_clip.waveform.size(1)]
            clip = AudioClip(
                lq_wave,
                lq_sample_rate,
                lq_wave.shape[0],
                "",
                serie["name"] + "_" + eq,
                hq_clip.frame_offset,  # original offset before resampling
                lq_wave.shape[1],
            )
            # LOG.info("LQ generated on the fly ....")
            return clip

        # if we work on the whole file
        if frame_offset == 0 and num_frames == -1:
            if persist_eq:  # if we should save the lq version
                lq_folder = Path(self.cfg.dataset.low_quality_path).expanduser()
                lq_path = eq_transform.process_file(serie["hq_path"], lq_folder)
                return self.get_sample(lq_path)
            else:
                lq_wave, lq_sample_rate = eq_transform.apply_file(serie["hq_path"])
                clip = AudioClip(
                    lq_wave,
                    lq_sample_rate,
                    lq_wave.shape[0],
                    "",
                    serie["name"] + "_" + eq,
                    frame_offset,
                    lq_wave.shape[1],
                )
                return clip

        # We generate EQ from a excerpt of the HQ file
        hq_clip = self.get_sample(serie["hq_path"], frame_offset, num_frames)
        if not hq_clip:
            LOG.error(f"Cannot load HQ sample {serie['hq_path']}")
            return None

        lq_wave, lq_sample_rate = eq_transform.apply_tensor(hq_clip.waveform, hq_clip.sample_rate)
        clip = AudioClip(
            lq_wave, lq_sample_rate, lq_wave.shape[0], "", serie["name"] + "_" + eq, frame_offset, lq_wave.shape[1]
        )
        return clip

    def get_random_eq_name(self) -> Optional[str]:
        if self.subset_mode is not None:
            names = list(map(lambda x: x.name, self.cfg.dataset.low_quality_effect[self.eq_mode]['effects']))
        else:
            names = list(map(lambda x: x.name, self.cfg.dataset.low_quality_effect['train']['effects']))
        if not names:
            LOG.error("No low quality effects found in config file")
            return None

        total_eqs = len(names)
        if self.subset_mode == 'train':
            # Potentially limit the total number of eqs
            if 'max_training_effects' in self.cfg.dataset:
                total_eqs = self.cfg.dataset.max_training_effects
            
            if total_eqs > 0:
                total_eqs = min(total_eqs, len(names))
            else:
                total_eqs = len(names)
            
        eq = self.rnd.choice(names[:total_eqs])
        return eq

    def get_eq_indices(self, eq_names: Union[List, str]) -> Optional[List[int]]:
        """Returns eq_name index in config, eq_user is always the first 0 if we use the user_eq,
        the rest of the eqs will be shifted in position"""
        def find_eq(name):
            if self.cfg.dataset.use_user_lq_files and name == 'eq_user':
                return 0

            idx = None
            for i, e in enumerate(self.cfg.dataset.low_quality_effect[self.subset_mode or 'train']['effects']):
                if e.name == name:
                    idx = i
                    break

            if self.cfg.dataset.use_user_lq_files and idx:
                idx += 1
            return idx

        if not eq_names:
            return None

        if isinstance(eq_names, str):
            eq_names = [eq_names]

        results = [find_eq(eq) for eq in eq_names]
        return results

    def get_random_sample_duration(self) -> int:
        duration = 1
        if self.subset_mode == 'train':
            duration = self.cfg.trainer.train_clip_duration
        elif self.subset_mode == "validation":
            duration = self.cfg.trainer.validation_clip_duration or self.cfg.trainer.train_clip_duration
        elif self.subset_mode == "test":
            duration = self.cfg.trainer.test_clip_duration or self.cfg.trainer.train_clip_duration
        # duration = self.cfg.trainer.audio_timesteps
        return duration

    def get_random_clips(self, n: int) -> Tuple[Optional[AudioClip], Optional[AudioClip], Optional[str]]:
        # print(f"Get item: {n}, {self.subset_view.iloc[n]['hq_path']}")
        # LOG.info(f"Get item: {n}, {self.subset_view.iloc[n]['hq_path']}\n")
        hq_clip = self.get_random_hq_sample(n, self.get_random_sample_duration())
        duration = hq_clip.waveform.shape[-1]
        while 2 * duration <= self.cfg.trainer.audio_timesteps:
            hq_clip.waveform = F.pad(hq_clip.waveform, (0,duration), 'constant')
        if duration < self.cfg.trainer.audio_timesteps:
            hq_clip.waveform = F.pad(hq_clip.waveform, (0,self.cfg.trainer.audio_timesteps - duration), 'constant')
        else:
            hq_clip.waveform = hq_clip.waveform[:,:self.cfg.trainer.audio_timesteps]
        if not hq_clip:
            return None, None, None
        
        eq_name = None
        lq_clip = None

        if self.cfg.dataset.no_lq_file:
            return hq_clip, hq_clip, 'no_eq'

        if self.cfg.dataset.use_user_lq_files:
            eq_name = "eq_user"
            lq_clip = self.get_lq_sample(
                n,
                eq_name,
                hq_clip.frame_offset,
                hq_clip.num_frames,
                hq_clip,
                self.cfg.dataset.persist_low_quality_files,
            )

        if not lq_clip:
            # LOG.debug(f"No LQ_FILE with EQ {eq_name}.")
            if self.cfg.dataset.fallback_generated_eqs:
                if self.eq_mode == 'random':
                    eq_name = 'random'
                else:
                    eq_name = self.get_random_eq_name()  # from defined eqs

                # LOG.debug(f"LQ_FILE with EQ {eq_name} from {mode} config instead.")
                # LOG.info(f"LQ_FILE with EQ {eq_name} from {mode} config instead.")
                assert eq_name
                lq_clip = self.get_lq_sample(
                    n,
                    eq_name,
                    hq_clip.frame_offset,
                    hq_clip.num_frames,
                    hq_clip,
                    self.cfg.dataset.persist_low_quality_files,
                )
            else:
                LOG.error("Cannot find user-defined low-quality file")

        if not lq_clip:
            lq_clip = None
            eq_name = None

        return lq_clip, hq_clip, eq_name

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor, str]:
        lq_clip, hq_clip, eq_name = self.get_random_clips(n)
        if not lq_clip:
            raise RuntimeError(f"Invalid LQ clip for index {n}")
        if not hq_clip:
            raise RuntimeError(f'Invalid HQ clip for index {n}')
        
        return lq_clip.waveform, hq_clip.waveform, eq_name

    def get_max_workers(self, desired_workers):
        chunks = np.array_split(self.subset_view.index, desired_workers)
        chunks = list(filter(lambda x: not x.empty, chunks))
        return len(chunks)



def worker_fn(worker_id):
    """Define a `worker_init_fn` that configures each dataset copy differently"""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    dataset.seed_rng_generator(worker_id)


def create_dataloader(
    dataset: AudioDataset,
    subset: str = None,
    eq_mode: str = None,
    rank: int = 0,
    world_size: int = 1,
    max_workers: int = 0,
    batch_size: int = 1,
):
    if eq_mode is None:
        eq_mode = subset
    if eq_mode not in ['train', 'test', 'validate', 'random', None]:
        raise KeyError
    dataset.eq_mode = eq_mode # train, test, validate, None
    dataset.change_subset(subset)
    actual_workers = max_workers
    # if max_workers > 1:
    #     actual_workers = dataset.get_max_workers(max_workers)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=None,
        sampler=sampler,
        # shuffle= True,
        num_workers=actual_workers,
        worker_init_fn=worker_fn,
        pin_memory=True,
        prefetch_factor=2
    )

    return loader
