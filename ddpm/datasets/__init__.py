import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from ddpm.datasets.cityscape import Cityscapes_Pretraining_Dataset, Cityscapes
from ddpm.datasets.celeba import CelebA
from ddpm.datasets.ffhq import FFHQ
from ddpm.datasets.lsun import LSUN
from ddpm.datasets.audio import AudioDataset
from ddpm.datasets.gta5 import GTA_Pretraining_Dataset
from torch.utils.data import Subset
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, cfg):
    if args is None:
        args = cfg.trainer
    dataset_name = args.dataset
    datapath = args.datapath
    corruption_severity = args.corruption_severity
    corruption = args.corruption
    random_flip = args.random_flip
    lower_image_size = args.lower_image_size
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

    if dataset_name == "CIFAR10":
        dataset = CIFAR10(
            os.path.join('./data', "datasets", "cifar10"),
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = CIFAR10(
            os.path.join('./data', "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif dataset_name == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if random_flip:
            dataset = CelebA(
                root=datapath,
                split="train",
                corruption=corruption,
                corruption_severity=corruption_severity,
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                ),
                download=False,
            )

        else:
            dataset = CelebA(
                root=datapath,#os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                corruption=corruption,
                corruption_severity=corruption_severity,
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                ),
                download=False,
            )

        test_dataset = CelebA(
            root=datapath, #os.path.join(args.exp, "datasets", "celeba"),
            split="test",
            corruption=corruption,
            corruption_severity=corruption_severity,
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            ),
            download=False,
        )
    elif dataset_name == "MAESTRO":
        dataset = AudioDataset(cfg)
        if dataset.should_build_dataset():
            dataset.build_collated()
        test_dataset = None

    elif dataset_name == "CITYSCAPES_ORIGINAL":
        if cfg.dataset.mode == "fine":
            dataset = Cityscapes(root=cfg.dataset.root_dir, split='train', mode=cfg.dataset.mode,
                     target_type='semantic', transforms=transforms.Compose([transforms.ToTensor(),]),)  
            val_dataset = Cityscapes(root=cfg.dataset.root_dir, split='val', mode=cfg.dataset.mode,
                     target_type='semantic', transforms=transforms.Compose([transforms.ToTensor(),]),)
            
            dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

            test_dataset = Cityscapes(root=cfg.dataset.root_dir, split='test', mode=cfg.dataset.mode,
                     target_type='semantic', transforms=transforms.Compose([transforms.ToTensor(),]),)

        elif cfg.dataset.mode == "coarse":
            raise NotImplementedError

    elif dataset_name == "CITYSCAPES_PRETRAINING":
        print("dataset_name :", dataset_name)
        if cfg.trainer.random_flip:
            train_transform =  transforms.Compose(
                [
                    transforms.Resize(lower_image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(image_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            train_transform =  transforms.Compose(
                [
                    transforms.Resize(lower_image_size),
                    transforms.RandomCrop(image_size),
                    transforms.ToTensor(),
                ]
            )

        test_transform = transforms.Compose(
            [transforms.Resize(lower_image_size), 
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            ]
        )
        
        dataset = Cityscapes_Pretraining_Dataset(cfg, transform = train_transform)
        test_dataset = Cityscapes_Pretraining_Dataset(cfg,transform = test_transform, split = "test")
    
    elif dataset_name == "CITYSCAPES":
        dataset = Cityscapes_Pretraining_Dataset(cfg)
        test_dataset = Cityscapes_Pretraining_Dataset(cfg, split = "test")

    elif dataset_name == "GTA":
        print("dataset_name :", dataset_name)
        if cfg.trainer.random_flip:
            train_transform =  transforms.Compose(
                [
                    transforms.Resize(lower_image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(image_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            train_transform =  transforms.Compose(
                [
                    transforms.Resize(lower_image_size),
                    transforms.RandomCrop(image_size),
                    transforms.ToTensor(),
                ]
            )
        dataset = GTA_Pretraining_Dataset(cfg, transform = train_transform)
        test_dataset = None

    elif dataset_name == "LSUN":
        train_folder = "{}_train".format(config.data.category)
        val_folder = "{}_val".format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                    ]
                ),
            )
        else:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
            )

        test_dataset = LSUN(
            root=os.path.join(args.exp, "datasets", "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.CenterCrop(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
        )

    elif dataset_name == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
                ),
                resolution=config.data.image_size,
            )
        else:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.ToTensor(),
                resolution=config.data.image_size,
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.99)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    
    else:
        LOG.warning(f"No dataset named {dataset_name}")
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
