
# credits to https://github.com/ermongroup/SDEdit/blob/main/colab_utils/utils.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline,DiffusionPipeline
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")
device = 'cuda'

@torch.no_grad()
def _clip_inputs(sample: torch.FloatTensor, t : int, sde_betas, sde_alphas, sde_alphas_cumprod, number_of_stds: float = 2., original_img = None, previous_x = None):
    """
    Cliping the inputs with an confidence interval given by the diffusion schedule
    """
    dtype = sample.dtype
    batch_size, channels, *remaining_dims = sample.shape
    if dtype not in (torch.float32, torch.float64):
        sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half
    alphas_cumprod = sde_alphas_cumprod
    alphas = sde_alphas
    alpha_t = sde_alphas_cumprod[t]
    sqrt_alpha_local_t = torch.sqrt(alphas[t])
    one_minus_alphas_local_t = torch.sqrt(1-alphas[t]).item()
    sqrt_alpha_t = torch.sqrt(alpha_t).item()
    one_minus_sqrt_alpha_t = torch.sqrt(1-alpha_t).item()
    if original_img is not None and previous_x is not None and t > 0:
        mean = original_img * torch.sqrt(alphas_cumprod[t-1]).item() * (sde_betas[t]).item() / (1 - alphas[t].item())
        mean += previous_x * torch.sqrt(alphas[t]).item() * (1 - alphas_cumprod[t-1].item()) / (1-alphas_cumprod[t].item())
        plt.imshow(mean[0].permute(1,2,0).cpu()/2+0.5)
        plt.show()
        std = betas[t].item() * (1-alphas_cumprod[t-1].item())/(1-alphas_cumprod[t].item())
        confidence_interval = [mean - number_of_stds * std, mean + number_of_stds * std]
    elif original_img is None:
        confidence_interval = [-sqrt_alpha_t - number_of_stds * one_minus_sqrt_alpha_t, sqrt_alpha_t + number_of_stds * one_minus_sqrt_alpha_t]
    else:
        confidence_interval = [sqrt_alpha_t * original_img - number_of_stds * one_minus_sqrt_alpha_t, 
                            sqrt_alpha_t * original_img + number_of_stds * one_minus_sqrt_alpha_t]
    sample = torch.clamp(sample, confidence_interval[0], confidence_interval[1])
    sample = sample.to(dtype)
    return sample

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *,
                                               model,
                                               logvar,
                                               betas, huggingface = False):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    if huggingface:
        model_output = model(x, t)[0]
    else:
        model_output, attn_maps = model(x, t.int())
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


def load_model(model_id = "google/ddpm-celebahq-256", device = 'cuda'):

    ddim = DDIMPipeline.from_pretrained(model_id)
    model = ddim.unet
    model = model.to(device)
    model.eval()
    scheduler = ddim.scheduler
    print("Model loaded")
    betas = ddim.scheduler.betas 
    num_timesteps = betas.shape[0]
    alphas = (1.0 - betas).numpy()
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    posterior_variance = betas * \
        (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    logvar = np.log(np.maximum(posterior_variance, 1e-20))

    return model, betas, num_timesteps, logvar


def imshow(img, title=""):
    img = img.to("cpu")
    img = img.permute(1, 2, 0, 3)
    img = img.reshape(img.shape[0], img.shape[1], -1)
    img = img / 2 + 0.5     # unnormalize
    img = torch.clamp(img, min=0., max=1.)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def SDEditing(img_tensor, betas, logvar, model, sample_step, total_noise_levels, n=4, huggingface = False, clip_input = False, number_of_stds = 2, verbose = False, device = 'cuda'):
    # print("Start sampling")
    sde_alphas = 1.0 - betas
    sde_alphas_cumprod = np.cumprod(sde_alphas, axis=0)
    with torch.no_grad():
        img = img_tensor.to(device)
        img = img_tensor / 2 + 0.5
        img = img.to(device)
        if len(img.shape) < 4:
            img = img.unsqueeze(dim=0)
        else:
            img = img
        inputs = img.split(1, dim=0)
        inputs = [inp.repeat(n, 1, 1, 1) for inp in inputs]
        img = torch.cat(inputs)
        # img = img.repeat(n, 1, 1, 1)
        mask = torch.zeros_like(img[0])
        mask = mask.to(device)
        x0 = img
        x0 = (x0 - 0.5) * 2.
        if verbose:
            imshow(x0, title="Initial input")

        for it in range(sample_step):
            e = torch.randn_like(x0)
            a = (1 - betas).cumprod(dim=0).to(device)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            if clip_input and total_noise_levels > 1:
                before = x
                x = _clip_inputs(x,total_noise_levels - 1, betas,sde_alphas, sde_alphas_cumprod, number_of_stds)
                # print('diff', (before - x).pow(2).mean())
            if verbose:
                imshow(x, title="Perturb with SDE")

            with tqdm(total=total_noise_levels, desc="Iteration {}".format(it)) as progress_bar:
                for i in reversed(range(total_noise_levels)):
                    # t = torch.ones_like(x) * i
                    t = (torch.ones(x.shape[0]) * i).to(device)
                    x_ = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                                                                    logvar=logvar,
                                                                    betas=betas, huggingface = huggingface)
                    x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                    if clip_input and i > 1:
                        before = x
                        x = _clip_inputs(x,i - 1, betas,sde_alphas, sde_alphas_cumprod, number_of_stds)
                        # print('diff', (before - x).pow(2).mean())
                    x[:, (mask != 1.)] = x_[:, (mask != 1.)]
                    # added intermediate step vis
                    # if (i - 99) % 100 == 0:
                    #     imshow(x, title="Iteration {}, t={}".format(it, i))
                    progress_bar.update(1)

            x0[:, (mask != 1.)] = x[:, (mask != 1.)]
            if verbose:
                imshow(x)
    return x