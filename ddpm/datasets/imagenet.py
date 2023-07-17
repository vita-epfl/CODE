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




class Imagenet_Dataset(VisionDataset):
    def __init__(self, cfg, root: Optional[str] = None, 
                            split: Optional[str] = None, 
                            corruption: Optional[List[str]] = None, 
                            corruption_severity: Optional[List[int]] = None,
                            transform: Optional[Callable] = None,
                            open_ai_normalization: bool = False,
                            inv_transform: Optional[Callable] = None):
        self.cfg = cfg
        self.open_ai_normalization = open_ai_normalization or self.cfg.trainer.open_ai_normalization
        # self.lower_image_size = OmegaConf.to_object(self.cfg.trainer.lower_image_size)
        # self.img_size = OmegaConf.to_object(self.cfg.trainer.img_size)

        if root is not None:
            self.root = root
        else:
            self.root = cfg.trainer.datapath

        super(Imagenet_Dataset, self).__init__(self.root)

        if split is not None:
            self.split = split
        else:
            self.split = cfg.trainer.split
        
        if corruption is not None:
            self.corruption = corruption
        else:
            self.corruption = OmegaConf.to_object(cfg.trainer.corruption)
            if len(self.corruption) == 0:
                print("No corruption")
                self.corruption = None

        if corruption_severity is not None:
            self.corruption_severity = corruption_severity
        else:
            self.corruption_severity = OmegaConf.to_object(cfg.trainer.corruption_severity)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
            
        self.images = []
        self.original_images = []
        valid_splits = ("train",  "val", "all")
        assert self.split in valid_splits

        if self.corruption is None:
            if self.split == "all":
                for split in ["train", "val"]:
                    self.images_dir = os.path.join(self.root, split)
                    for subdir in os.listdir(self.images_dir):
                        img_subdir = os.path.join(self.images_dir, subdir)
                        if os.path.isdir(img_subdir):
                            for file_name in os.listdir(img_subdir):
                                self.images.append(os.path.join(img_subdir, file_name))
                            
            else:
                self.images_dir = os.path.join(self.root, split)
                for subdir in os.listdir(self.images_dir):
                    img_subdir = os.path.join(self.images_dir, subdir)
                    for file_name in os.listdir(img_subdir):
                        self.images.append(os.path.join(img_subdir, file_name))
        else:
            self.original_image_dir = os.path.join(self.root,'val')
            self.image_c_dir = os.path.join(self.root, "imagenet-c")
            for corruption in self.corruption:
                corrupted_image_dir = os.path.join(self.image_c_dir, corruption)
                for severity in self.corruption_severity:
                    severity_subdir = os.path.join(corrupted_image_dir, str(severity))
                    for img_subdir in os.listdir(severity_subdir): 
                        img_subdir_path = os.path.join(severity_subdir, img_subdir)
                        original_subdir_path = os.path.join(self.original_image_dir, img_subdir)
                        for file_name in os.listdir(img_subdir_path):
                            self.images.append(os.path.join(img_subdir_path, file_name))
                            self.original_images.append(os.path.join(original_subdir_path, file_name))


    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """
        if self.open_ai_normalization:
            X = Image.open(self.images[index])
            X.load()
            X = X.convert("RGB")
            if self.corruption is not None:
                X_original = Image.open(self.original_images[index])
                X_original.load()
                X_original = X_original.convert("RGB")
            else:
                X_original = Image.open(self.images[index])
                X_original.load()
                X_original = X_original.convert("RGB")

            X = np.array(X)
            X_original = np.array(X_original)
            X = X.astype(np.float32) / 127.5 - 1
            X_original = X_original.astype(np.float32) / 127.5 - 1
        else:
            X = Image.open(self.images[index]).convert("RGB")
            if self.corruption is not None:
                X_original = Image.open(self.original_images[index]).convert("RGB")
            else:
                X_original = Image.open(self.images[index]).convert("RGB")


        if self.transform is not None:
            image = self.transform(X)
            image_original = self.transform(X_original)
        else:
            image = X
            image_original = X_original
        
        return image, image_original


    def __len__(self) -> int:
        return len(self.images)