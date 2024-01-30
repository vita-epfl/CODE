import torch
import os
import PIL
from PIL import ImageEnhance
from .corruptions import *
from .vision import VisionDataset
from .utils import download_file_from_google_drive, check_integrity


class CelebAHQ(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "CelebAMask-HQ"


    def __init__(self, root,
                 split="train",
                 transform=None, target_transform=None,
                 corruption=None,
                 datapath = "/mnt/scitas/bastien/CelebAMask-HQ/CelebA-HQ-img",
                 corruption_severity=5):
        import pandas
        super(CelebA, self).__init__(root)
        self.split = split
        self.corruption = corruption
        self.corruption_severity = corruption_severity
        self.transform = transform
        self.target_transform = target_transform
        self.datapath = datapath

        
        # mask = (splits[1] == split)
        # self.filename = splits[mask].index.values

    def __getitem__(self, index):
        img_path = f"{self.datapath}/{index}.jpg"
        X = PIL.Image.open(img_path)
        X_original = X
        
        if self.corruption is not None:
            if self.corruption == "black_and_white":
                filter = ImageEnhance.Color(X)
                X = filter.enhance(0)
            elif self.corruption == "speckle_noise":
                X = PIL.Image.fromarray(speckle_noise(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "gaussian_noise":
                X = PIL.Image.fromarray(gaussian_noise(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "shot_noise":
                X = PIL.Image.fromarray(shot_noise(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "impulse_noise":
                X = PIL.Image.fromarray(impulse_noise(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "spatter":
                X = PIL.Image.fromarray(spatter(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "zoom_blur":
                X = PIL.Image.fromarray(zoom_blur(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "elastic_transform":
                X = PIL.Image.fromarray(elastic_transform(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "jpeg_compression":
                X = PIL.Image.fromarray(jpeg_compression(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "pixelate":
                X = PIL.Image.fromarray(pixelate(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "saturate":
                X = PIL.Image.fromarray(saturate(X, severity  = self.corruption_severity).astype(np.uint8)) 
            elif self.corruption == "masking_random_color_random":
                X = PIL.Image.fromarray(masking_random_color_random(X, severity  = self.corruption_severity).astype(np.uint8)) 
            elif self.corruption == "masking_hline_random_color":
                X = PIL.Image.fromarray(masking_hline_random_color(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "masking_vline_random_color":
                X = PIL.Image.fromarray(masking_vline_random_color(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "masking_random_color":
                X = PIL.Image.fromarray(masking_random_color(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "masking_simple":
                X = PIL.Image.fromarray(masking_simple(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "masking_color_lines":
                X = PIL.Image.fromarray(masking_color_lines(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "masking_line":
                X = PIL.Image.fromarray(masking_line(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "masking_gaussian":
                X = PIL.Image.fromarray(masking_gaussian
                (X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == 'gaussian_blur':
                X = PIL.Image.fromarray(gaussian_blur(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == 'motion_blur':
                X = PIL.Image.fromarray(motion_blur(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "glass_blur":
                X = PIL.Image.fromarray(glass_blur(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "defocus_blur":
                X = PIL.Image.fromarray(defocus_blur(X, severity  = self.corruption_severity).astype(np.uint8))
            elif self.corruption == "frost":
                X = PIL.Image.fromarray(frost(X, severity  = self.corruption_severity).astype(np.uint8))
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
            X = self.transform(X)
            X_original = self.transform(X_original)

        if self.target_transform is not None:
            target = self.target_transform(target)

        indexes = torch.ones_like(X_original) * index
        return X, X_original, indexes

    def __len__(self):
        # TO DO 
        # return len(self.attr)
        return 30000

