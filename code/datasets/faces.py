import torch
import os
import PIL
from PIL import ImageEnhance
from .vision import VisionDataset


def is_png_file(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        return False
    
    # Check the file extension
    return file_path.lower().endswith('.png')

class Faces(VisionDataset):
    """ Face Dataset mixing ffhq and celebahq
    Args:
    """

    base_folder = "CelebAMask-HQ"

    def __init__(self, celeba_root = "/home/datasets/CelebAMask-HQ/CelebA-HQ-img",
                 ffhq_root="/home/datasets/images1024x1024/",
                 split="train",
                 transform=None, target_transform=None,):

        super(Faces, self).__init__(root="/home/datasets")
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.celeba_root = celeba_root
        self.celeba_length = len([name for name in os.listdir(self.celeba_root) if os.path.isfile(os.path.join(self.celeba_root, name))])
        print("Celeba nb of items:", self.celeba_length)
        self.ffhq_root = ffhq_root
        self.ffhq_length = len([name for name in os.listdir(self.ffhq_root) if  is_png_file(os.path.join(self.ffhq_root, name))]) #os.path.isfile(os.path.join(self.ffhq_root, name)) and ])
        print("ffhq nb of items:", self.ffhq_length)

    def __getitem__(self, index):
        if index < self.celeba_length:
            img_path = f"{self.celeba_root}/{index}.jpg"
            X = PIL.Image.open(img_path)
            X_original = X
        else:
            new_index = str(index - self.celeba_length)
            while len(new_index) < 5:
                new_index = '0'+new_index
            img_path = f"{self.ffhq_root}/{new_index}.png"
            X = PIL.Image.open(img_path)
            X_original = X
        if self.target_transform is not None:
            target = self.target_transform(X_original)
        if self.transform is not None:
            X = self.transform(X)
            X_original = self.transform(X_original)
        indexes = torch.tensor(index).int()
        return X, X_original, indexes

    def __len__(self):
        return self.celeba_length + self.ffhq_length

