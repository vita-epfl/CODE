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

t = transforms.Compose([transforms.ToTensor(), transforms.Resize([256]), transforms.CenterCrop([256, 256])])

def to_pil(img):
    return Image.fromarray((img * 255).permute(1, 2, 0).numpy().astype(np.uint8))


psnr_tgt_metric = PeakSignalNoiseRatio()
def compute_psnr(img_path, algo):
    tgt_path = Path(img_path).parents[0 if algo == "none" else 2] / "original.png"
    tgt = t(Image.open(tgt_path))
    img = t(Image.open(img_path))
    return psnr_tgt_metric(img, tgt).item()

def process_image(img_info):
    img_path, algo, gpu_id = img_info
    device = 'cpu'
    try:

        
        tgt_path = Path(img_path).parents[0 if algo == "none" else 2] / "original.png"
        
        tgt = t(Image.open(tgt_path)).to(device)
        img = t(Image.open(img_path)).to(device)

        return {
            "psnr_to_target": psnr_tgt_metric(img, tgt).item(),
        }
    except Exception as e:
        print(f"error with image {img_path} on GPU {gpu_id} - {e}")
        return None

def get_filtering_metrics(df):
    num_gpus = torch.cuda.device_count()
    metrics = []
    img_info_list = [(row["path"], row["algo"], 0) for i, row in df.iterrows()]
    
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_image, img_info_list), total=len(img_info_list)):
            if result is not None:
                metrics.append(result)
    
    return pd.DataFrame(metrics)

#####################################################################

print("SCANNING FILES")

files = []
path_list = [
    "",
]

def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)  # see below for Python 2.x
        else:
            yield entry.path

max_image= 50000
for p in path_list:
    print(f"Looking in path: {p}")
    files.extend(tqdm(scantree(p)))
    if len(files) > max_image:
        break

print("DONE")
print("Number of files found:", len(files))


files = pd.Series(files)
files = files[files.str.endswith('png')]
print(files)

#####################################################################

print("BUILDING FILE DATAFRAME...", end="")
info = []
for f in files:
    f_split = str(f).split("/")
    if f.name in ["corrupted.png", "original.png"]:
        info.append({
            "path": f,
            "corruption": f_split[-3],
            "img_id": f_split[-2],
            "filename": f_split[-1],
            "algo": "none",
            "clipping": "none",
            "timestamp": f_split[-4],
            "latent": "none",
            "epsilon": np.nan,
        })
    else:
        info.append({
            "path": f,
            "corruption": f_split[-5],
            "img_id": f_split[-4],
            "filename": f_split[-1],
            "algo": f_split[-3],
            "clipping": f_split[-2],
            "timestamp": f_split[-6],
            "latent": f_split[-1].split("_")[0] if f_split[-3] == "sde" or f_split[-3] == "ode" else "none",
            "epsilon": f_split[-1].split("_")[1] if f_split[-3] == "ode" else np.nan
        })

df = pd.DataFrame(info)

# specific for these two folders, ode ran with clipping but got saved in the non_clipped folder
df.loc[
    (
        df['path'].apply(lambda x: 'fid' in str(x)) &
        df['timestamp'].isin(['2024-05-22_19-52', '2024-05-22_19-58']) &
        (df['algo'] == 'ode')
    ),
    'clipping'
] = 'clipped'

df['epsilon'] = pd.cut(
    df['epsilon'].astype(np.float64),
    bins=[1e-7, 5e-7, 1e-6, 5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 5e-2, 1e-1],
    labels=[1e-7, 5e-7, 1e-6, 5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 5e-2],
    right=False
)
print("DONE")
print(df.head(2))
print("\n\n")

#####################################################################

print("PRINTING THE NUMBER OF UNIQUE IMAGES PER ALGORITHM")
print("\n")
print("CODE w/o clipping")
print(df[(df['algo'] == "ode") & (df['clipping'] == 'non_clipped')].groupby("corruption")['img_id'].unique().apply(len))
print("\n")
print("CODE w/ clipping")
print(df[(df['algo'] == "ode") & (df['clipping'] == 'clipped')].groupby("corruption")['img_id'].unique().apply(len))
print("\n")
print("SDE latent 299")
print(df[(df['algo'] == "sde") & (df['latent'] == "299")].groupby("corruption")['img_id'].unique().apply(len))
print("\n")
print("SDE latent 699")
print(df[(df['algo'] == "sde") & (df['latent'] == "699")].groupby("corruption")['img_id'].unique().apply(len))
print("\n")
print("Corruption list:")
print(df['corruption'].unique())
print("\n")

#####################################################################

print("GETTING FILTERING METRICS...", end="")
print(df.head())
df = pd.concat([df, get_filtering_metrics(df)], axis=1)
Path("../metrics_csv/").mkdir(exist_ok=True)
df.to_csv("../metrics_csv/df_with_metrics_test.csv")
print("DONE")


#####################################################################

def save_top_k_by_algo(df, k, ascending, metric_name, col_to_delete=None, other_cols_to_group=[], file_suffix=None):
    df_with_metrics_filtered = (
        df[~df['algo'].isin(["none", "reconstruction"])]
        .groupby(['img_id', 'corruption', 'algo', 'clipping'] + other_cols_to_group, observed=False)
        .apply(lambda x: x.sort_values(metric_name, ascending=ascending).head(k))
    )

    df_with_metrics_filtered = pd.concat([
        df_with_metrics_filtered,
        df[df['algo'].isin(['none', 'reconstruction'])]
    ], axis=0)

    df_with_metrics_filtered.to_csv(f"../metrics_csv/top_{k}_imgs_by_{metric_name}{'_' + file_suffix if file_suffix else ''}.csv")


print("SAVING TOP K IMAGES BY METRIC...", end="")
save_top_k_by_algo(df, 1, False, "psnr_to_target", )
save_top_k_by_algo(df, 4, False, "psnr_to_target", )

save_top_k_by_algo(df, 1, False, "psnr_to_target", other_cols_to_group=["latent", "epsilon"], file_suffix="by_latent_epsilon")
save_top_k_by_algo(df, 4, False, "psnr_to_target", other_cols_to_group=["latent", "epsilon"], file_suffix="by_latent_epsilon")

print("DONE")