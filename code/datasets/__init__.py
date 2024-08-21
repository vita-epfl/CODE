import os
import torch
import numbers
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from code.datasets.celebahq import CelebAHQ
from code.datasets.lsun import LSUN
from code.datasets.faces import Faces
from torch.utils.data import Subset
import copy
from omegaconf import DictConfig, open_dict
import numpy as np


def get_dataset(args, cfg):
    if args is None:
        args = cfg.trainer
    dataset_name = args.dataset
    datapath = args.datapath
    split = args.split
    random_flip = args.random_flip
    try:
        lower_image_size = args.lower_image_size
        original_img_size = OmegaConf.to_object(args.original_img_size)
    except:
        lower_image_size = None
        original_img_size = None
    image_size = args.img_size
    
    if random_flip is False:
        train_transform = test_transform = transforms.Compose(
            [transforms.Resize(image_size), transforms.ToTensor()]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(image_size), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    if dataset_name == "CELEBAHQ":
        dataset = CelebAHQ(
                root=datapath,
                split="train",
                transform=transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Resize(image_size),
                        transforms.CenterCrop([cfg.trainer.img_size,cfg.trainer.img_size]), 
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                            )
                )
        test_dataset = CelebAHQ(
                root=datapath,
                split="test",
                transform=transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Resize(image_size),
                        transforms.CenterCrop([cfg.trainer.img_size,cfg.trainer.img_size]), 
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                            )
        )

    elif dataset_name == "FACES":
        dataset = Faces(
                celeba_root=cfg.trainer.celeba_root,
                ffhq_root=cfg.trainer.ffhq_root,
                split="all",
                transform=transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Resize(image_size),
                        transforms.CenterCrop([cfg.trainer.img_size,cfg.trainer.img_size]), 
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                            )
                )
        test_dataset = None


    elif dataset_name == "LSUN":
        train_folder = "{}_train".format(cfg.trainer.lsun_category)
        val_folder = "{}_val".format(cfg.trainer.lsun_category)
        if cfg.trainer.random_flip:
            dataset = LSUN(
                root=cfg.trainer.datapath, #os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(cfg.trainer.img_size),
                        transforms.CenterCrop([cfg.trainer.img_size,cfg.trainer.img_size]),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                ),
            )
        else:
            dataset = LSUN(
                root= cfg.trainer.datapath, #os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(cfg.trainer.img_size),
                        transforms.CenterCrop([cfg.trainer.img_size,cfg.trainer.img_size]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                ),
            )

        test_dataset = LSUN(
            root= cfg.trainer.datapath ,#os.path.join(args.exp, "datasets", "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(cfg.trainer.img_size),
                    transforms.CenterCrop([cfg.trainer.img_size,cfg.trainer.img_size]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            ),
        )

    return dataset, test_dataset




