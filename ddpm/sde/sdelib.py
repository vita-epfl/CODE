import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline,DiffusionPipeline
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
device = 'cuda'

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


def SDEditing(img_tensor, betas, logvar, model, sample_step, total_noise_levels, n=4, huggingface = False,  verbose = False, device = 'cuda'):
    # print("Start sampling")
    with torch.no_grad():
        # [mask, img] = torch.load("colab_demo/{}.pth".format(name))
        # img = PIL.Image.open("/mnt/scitas/bastien/CelebAMask-HQ/CelebA-HQ-img/1008.jpg").resize([256,256])
        # img = img.convert('RGB')
        # img = (torch.from_numpy(np.array(img))/255).permute(2,0,1)
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
                    x[:, (mask != 1.)] = x_[:, (mask != 1.)]
                    # added intermediate step vis
                    # if (i - 99) % 100 == 0:
                    #     imshow(x, title="Iteration {}, t={}".format(it, i))
                    progress_bar.update(1)

            x0[:, (mask != 1.)] = x[:, (mask != 1.)]
            if verbose:
                imshow(x)
    return x