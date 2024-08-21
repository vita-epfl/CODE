import torch
import os
import PIL
from PIL import ImageEnhance
from .corruptions import *
from .vision import VisionDataset
import pandas


class CelebAHQ(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    base_folder = "CelebAMask-HQ"


    def __init__(self, root = "",
                 split="train",
                 test_split=0.1,
                 transform=None, target_transform=None,):

        
        super(CelebAHQ, self).__init__(root)
        self.split = split
        if self.split is None:
            self.split = "all"
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.all_length = len([name for name in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, name))])
        if self.split == "all" :
            self.length = self.all_length
        elif self.split == 'train':
            self.length = int((1-test_split)*self.all_length)
        elif self.split == 'test':
            self.length = int(test_split*self.all_length)

    def __getitem__(self, index):
        if self.split == "train" or self.split == "all":
            img_path = f"{self.root}/{index}.jpg"

        elif self.split == "test":
            new_index = self.all_length - index
            img_path = f"{self.root}/{new_index}.jpg"
        X = PIL.Image.open(img_path)
        X_original = X
        
        if self.transform is not None:
            X = self.transform(X)
            X_original = self.transform(X_original)

        if self.target_transform is not None:
            target = self.target_transform(target)

        indexes = torch.ones_like(X_original) * torch.tensor(index)
        indexes = torch.tensor(index).int()
        return X, X_original, indexes

    def __len__(self):
        return self.length

