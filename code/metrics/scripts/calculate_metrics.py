import sys
sys.path.append("/home/CODE")

import torch
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from torchvision import transforms
from argparse import ArgumentParser
from code.datasets.celebahq import CelebAHQ
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)



USE_TOP_K = 4
USE_METRIC = "psnr_to_target"

t =  transforms.Compose([transforms.ToTensor(), transforms.Resize([256]), transforms.CenterCrop([256, 256])])

def int_(x):
    try:
        return int(x)
    except:
        return x


def get_metrics(df, fid_metric, other_groups=[], device=None):
    metrics = []
    
    for g in tqdm(df.groupby(["corruption"]+other_groups, dropna=False)):
        c = g[0]    
        psnr_src_metric = PeakSignalNoiseRatio().to(device)
        psnr_tgt_metric = PeakSignalNoiseRatio().to(device)
        ssim_src_metric = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device)
        ssim_tgt_metric = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device)
        lpips_src_metric = LearnedPerceptualImagePatchSimilarity().to(device)
        lpips_tgt_metric = LearnedPerceptualImagePatchSimilarity().to(device)
        l2_src_metric = []
        l2_tgt_metric = []
        fid_metric.reset()
        
        list_imgs = list(g[1].iterrows())
        for i, img in list_imgs:
            try:
                src = t(Image.open(Path(img["path"]).parents[0 if img["algo"] == "none" else 2] / "corrupted.png")).to(device)
                tgt = t(Image.open(Path(img["path"]).parents[0 if img["algo"] == "none" else 2] / "original.png")).to(device)
                img = t(Image.open(img["path"])).to(device)
                
                psnr_src_metric.update(img, src)
                psnr_tgt_metric.update(img, tgt)
                ssim_src_metric.update(img.unsqueeze(0), src.unsqueeze(0))
                ssim_tgt_metric.update(img.unsqueeze(0), tgt.unsqueeze(0))
                lpips_src_metric.update((img * 2 - 1).unsqueeze(0), (src * 2 - 1).unsqueeze(0))
                lpips_tgt_metric.update((img * 2 - 1).unsqueeze(0), (tgt * 2 - 1).unsqueeze(0))
                l2_src_metric.append(torch.linalg.norm((img - src).reshape(-1)))
                l2_tgt_metric.append(torch.linalg.norm((img - tgt).reshape(-1)))
                fid_metric.update((img * 255).to(torch.uint8).unsqueeze(0), real=False)
            except Exception as e:
                print(f"error with image {i} - {e}")
    
        psnr_to_source = psnr_src_metric.compute()
        psnr_to_target = psnr_tgt_metric.compute()
        ssim_to_source = ssim_src_metric.compute()
        ssim_to_target = ssim_tgt_metric.compute()
        l2_to_source = torch.stack(l2_src_metric).mean()
        l2_to_target = torch.stack(l2_tgt_metric).mean()
        lpips_to_source = lpips_src_metric.compute()
        lpips_to_target = lpips_tgt_metric.compute()
        fid = fid_metric.compute()
        dic_metrics = {
            "corruption": c[0],
            "psnr_to_input": psnr_to_source.item(),
            "psnr_to_target": psnr_to_target.item(),
            "ssim_to_input": ssim_to_source.item(),
            "ssim_to_target": ssim_to_target.item(),
            "l2_to_input": l2_to_source.item(),
            "l2_to_target": l2_to_target.item(),
            "lpips_to_input": lpips_to_source.item(),
            "lpips_to_target": lpips_to_target.item(),
            "fid": fid.item(),
            "num_imgs": len(list(g[1].iterrows()))
        }
        for nb, col in enumerate(other_groups):
            dic_metrics[col] = c[nb+1]
        metrics.append(dic_metrics)
        
    return pd.DataFrame(metrics)

#####################################################################

def run_job(k, device):
    df = pd.read_csv(f'/home/metrics_csv/top_{USE_TOP_K}_imgs_by_{USE_METRIC}.csv')
    df_2 = pd.read_csv(f'/home/metrics_csv/top_{USE_TOP_K}_imgs_by_{USE_METRIC}_by_latent_epsilon.csv')
    print(f"Legnth du df {len(df)}")
    dfs = {
        "reconstruction_without_clipping": {
            "df": df[(df["algo"] == "reconstruction") & (df["clipping"] == "non_clipped")].copy().drop(columns=["timestamp"]).copy(),
            "df_2": df_2[(df_2["algo"] == "reconstruction") & (df_2["clipping"] == "non_clipped")].copy().drop(columns=["timestamp"]).copy(),
        },
        "reconstruction_with_clipping": {
            "df": df[(df["algo"] == "reconstruction") & (df["clipping"] == "clipped")].copy().drop(columns=["timestamp"]).copy(),
            "df_2": df_2[(df_2["algo"] == "reconstruction") & (df_2["clipping"] == "clipped")].copy().drop(columns=["timestamp"]).copy()
        },
        "ode_without_clipping": {
            "df": df[(df["algo"] == "ode") & (df["clipping"] == "non_clipped")].copy().drop(columns=["timestamp"]).copy(),
            "df_2": df_2[(df_2["algo"] == "ode") & (df_2["clipping"] == "non_clipped")].copy().drop(columns=["timestamp"]).copy()
        },
        "ode_with_clipping": {
            "df": df[(df["algo"] == "ode") & (df["clipping"] == "clipped")].copy().drop(columns=["timestamp"]).copy(),
            "df_2": df_2[(df_2["algo"] == "ode") & (df_2["clipping"] == "clipped")].copy().drop(columns=["timestamp"]).copy()
        },
        "sde_without_clipping": {
            "df": df[(df["algo"] == "sde") & (df["clipping"] == "non_clipped")].copy().drop(columns=["timestamp"]).copy(),
            "df_2": df_2[(df_2["algo"] == "sde") & (df_2["clipping"] == "non_clipped")].copy().drop(columns=["timestamp"]).copy()
        },
        "original": {
            "df": df[(df["algo"] == "none") & (df["filename"] == "original.png")].copy().drop(columns=["timestamp"]).copy(),
            "df_2": df_2[(df_2["algo"] == "none") & (df_2["filename"] == "original.png")].copy().drop(columns=["timestamp"]).copy()
        },
        "corrupted": {
            "df": df[(df["algo"] == "none") & (df["filename"] == "corrupted.png")].copy().drop(columns=["timestamp"]).copy(),
            "df_2": df_2[(df_2["algo"] == "none") & (df_2["filename"] == "corrupted.png")].copy().drop(columns=["timestamp"]).copy()
        },
    }
    
    print("INITIALIZING FID WITH TRUE IMAGES")

    device = f"cuda:{device}"
    
    celebahq_fid = FrechetInceptionDistance(reset_real_features=False).to(device)
    
    class ADataset(Dataset):
        def __init__(self, ds):
            self.ds = ds
            
        def __len__(self):
            return len(self.ds)
        
        def __getitem__(self, idx):
            return self.ds[idx][0]
    
    celeba_dataset = CelebAHQ(root="/home/datasets/CelebAMask-HQ/CelebA-HQ-img/", transform=t)
    loader = DataLoader(ADataset(celeba_dataset), batch_size=16, num_workers=4, pin_memory=True)
    
    for batch in tqdm(loader):
        celebahq_fid.update((batch * 255).to(torch.uint8).to(device), real=True)
    
    print("DONE")
    
    Path("../metrics_per_corruption").mkdir(exist_ok=True)
    Path("../metrics_per_epsilon-latent").mkdir(exist_ok=True)
    Path("../metrics_aggregated").mkdir(exist_ok=True)
    
    v = dfs[k]
    
    print(k)
    
    if False:
        print("Per corruption")
        print(f"Computing for {k}")
        metrics_per_corruption = get_metrics(v["df"].copy(), celebahq_fid, device=device)
        metrics_per_corruption.to_csv(f"../metrics_per_corruption/{k}_top_{USE_TOP_K}_{USE_METRIC}.csv")

        v["metrics_per_corruption"] = metrics_per_corruption

    print("Per corruption and epsilon-latent")

    _fake_df = v["df_2"].copy()
    _fake_df['latent'] = _fake_df['latent'].apply(int_)

    metrics_per_corruption_and_epsilon_and_latent = get_metrics(_fake_df.copy(), celebahq_fid, ['latent', 'epsilon'], device=device)
    metrics_per_corruption_and_epsilon_and_latent.to_csv(f"../metrics_per_epsilon-latent/{k}_top_{USE_TOP_K}_{USE_METRIC}_bis.csv")
    
    v["metrics_per_corruption_and_epsilon_and_latent"] = metrics_per_corruption_and_epsilon_and_latent

    if False:
        print("Per aggregated")
        _fake_df = v["df"].copy()
        _fake_df["corruption"] = "all"

        metrics_aggregated = get_metrics(_fake_df.copy(), celebahq_fid, device=device)
        metrics_aggregated.to_csv(f"../metrics_aggregated/{k}_top_{USE_TOP_K}_{USE_METRIC}.csv")
        
        v["metrics_aggregated"] = metrics_aggregated

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('algo')
    parser.add_argument('device')
    args = parser.parse_args()
    
    run_job(args.algo, args.device)