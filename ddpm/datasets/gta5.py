import torch
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import os
import PIL
from collections import namedtuple
from omegaconf import OmegaConf
from .corruptions import *
from PIL import Image
from .vision import VisionDataset
from .utils import download_file_from_google_drive, check_integrity
from torchvision.datasets.utils import extract_archive, iterable_to_str, verify_str_arg
import torchvision
import torchvision.transforms as transforms
import logging

LOG = logging.getLogger(__name__)

class GTA_Pretraining_Dataset(VisionDataset):
    def __init__(self, cfg, root: Optional[str] = None, 
                            split: Optional[str] = None, 
                            corruption: Optional[str] = None, 
                            corruption_severity: Optional[int] = None,
                            transform: Optional[Callable] = None,
                            inv_transform: Optional[Callable] = None,
                            image_folder: str = 'images',):
        
        self.cfg = cfg
        self.lower_image_size = OmegaConf.to_object(self.cfg.trainer.lower_image_size)
        self.img_size = OmegaConf.to_object(self.cfg.trainer.img_size)
        self.image_folder = image_folder
        self.labels_folder = image_folder

        if root is not None:
            self.root = root
        else:
            self.root = cfg.trainer.datapath

        super(GTA_Pretraining_Dataset, self).__init__(self.root)

        if split is not None:
            self.split = split
        else:
            self.split = cfg.trainer.split
        
        if corruption is not None:
            self.corruption = corruption
        else:
            self.corruption = cfg.trainer.corruption

        if corruption_severity is not None:
            self.corruption_severity = corruption_severity
        else:
            self.corruption_severity = cfg.trainer.corruption_severity

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
        
        self.images = []

        valid_splits = ("train", "test", "val", "train_val", "all")
        assert self.split in valid_splits
        LOG.warning("No split when using GTA")

        self.images_dir = os.path.join(self.root,self.image_folder)
        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))


    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """

        X = Image.open(self.images[index]).convert("RGB")
        X_original = X

        if self.corruption is not None:         
            if self.corruption == 'gaussian_blur':
                X = PIL.Image.fromarray(gaussian_blur(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "glass_blur":
                X = PIL.Image.fromarray(glass_blur(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "defocus_blur":
                X = PIL.Image.fromarray(defocus_blur(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "fog":
                X = PIL.Image.fromarray(fog(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "brightness":
                X = PIL.Image.fromarray(brightness(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "contrast":
                X = PIL.Image.fromarray(contrast(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "spatter":
                X = PIL.Image.fromarray(spatter(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "elastic_transform":
                X = PIL.Image.fromarray(elastic_transform(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "snow":
                X = PIL.Image.fromarray(snow(X, severity  = self.corruption_severity).astype(np.uint8))

        if self.transform is not None:
            image = self.transform(X)
            image_original = self.transform(X_original)
        else:
            image = X
            image_original = X_original
        
        return image, image_original


    def __len__(self) -> int:
        return len(self.images)
        