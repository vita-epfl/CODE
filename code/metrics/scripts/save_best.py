import sys
sys.path.append("/home/CODE")

import os
import cv2
import torch
import fnmatch
import pandas as pd
import numpy as np

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from torchvision import transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from concurrent.futures import ProcessPoolExecutor

t = transforms.Compose([transforms.ToTensor(), transforms.Resize([256], antialias = True), transforms.CenterCrop([256, 256])])
psnr_tgt_metric = PeakSignalNoiseRatio()

def int_(x):
    try:
        return int(x)
    except:
        return 0

def to_pil(img):
    return Image.fromarray((img * 255).permute(1, 2, 0).numpy().astype(np.uint8))

def save_top_k_by_algo(df, k, ascending, metric_name, col_to_delete=None, other_cols_to_group=[], file_suffix=None):
    df = df.drop(["Unnamed: 0.1","Unnamed: 0"] , axis = 1)
    print(df.latent.unique())
    df_with_metrics_filtered = (
        df[~df['algo'].isin(["none", "reconstruction"])]
        .groupby(['img_id', 'corruption', 'algo', 'clipping'] + other_cols_to_group, observed=False, dropna=False)
        .apply(lambda x: x.sort_values(metric_name, ascending=ascending).head(k))
    )

    df_with_metrics_filtered = pd.concat([
        df_with_metrics_filtered,
        df[df['algo'].isin(['none', 'reconstruction'])]
    ], axis=0)


    print(df_with_metrics_filtered.latent.unique())

    df_with_metrics_filtered.to_csv(f"/home/metrics_csv/top_{k}_imgs_by_{metric_name}{'_' + file_suffix if file_suffix else ''}.csv")


df = pd.read_csv("/home/metrics_csv/df_with_metrics_test.csv")
df['epsilon'] =  df["path"].apply(lambda x :  str(x).split("/")[-1].split("_")[1] if str(x).split("/")[-3] == "ode" else np.nan)
df['epsilon'] = df['epsilon'].astype(np.float64)
df['latent'] = df['latent'].apply(int_)
print(df.latent.unique())
print(f"Number of images to evaluate: {len(df)}")

print("SAVING TOP 1 IMAGES BY METRIC...")
save_top_k_by_algo(df, 1, False, "psnr_to_target", )
print("SAVING TOP 4 IMAGES BY METRIC...")
save_top_k_by_algo(df, 4, False, "psnr_to_target", )
print("SAVING TOP 1 IMAGES BY METRIC...")
save_top_k_by_algo(df, 1, False, "psnr_to_target", other_cols_to_group=["latent", "epsilon"], file_suffix="by_latent_epsilon")
print("SAVING TOP 4 IMAGES BY METRIC...")
save_top_k_by_algo(df, 4, False, "psnr_to_target", other_cols_to_group=["latent", "epsilon"], file_suffix="by_latent_epsilon")