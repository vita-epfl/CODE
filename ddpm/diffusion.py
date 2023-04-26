import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
import numpy as np
from torchvision.utils import make_grid, save_image
import os
import math
import enum


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        output = self.model(x_t, t)
        loss = F.mse_loss(output, noise, reduction='none')
        return loss

class HeatDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_dim = 128):
        super().__init__()

        self.model = model
        self.T = T
        self.img_dim = img_dim
        self.sigma_blur_max = 10

        # compute frequencies
        self.freqs = torch.pi * torch.linspace(0, self.img_dim-1, self.img_dim) / self.img_dim
        # self.labda = torch.square(self.freqs[None, :, None, None]) + torch.square(self.freqs[None, None, :, None])
        self.labda = torch.square(self.freqs[None, None, :, None]) + torch.square(self.freqs[None, None, None, :])
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def get_frequency_scaling(self, t, min_scale=0.001):
        # compute dissipation time
        sigma_t = self.sigma_blur_max * torch.sin(t * torch.pi / 2).pow(2)
        
        dissipation_time = sigma_t.pow(2) / 2
        dissipation_time = dissipation_time.unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
        # compute scaling for frequencies
        scaling = torch.exp(-self.labda.cuda() * dissipation_time) * (1 - min_scale) 
        scaling = scaling + min_scale
        return scaling

    def get_noise_scaling_cosine(self, t, logsnr_min=-10, logsnr_max=10): 
        limit_max = torch.arctan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
        limit_min = torch.arctan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - limit_max
        logsnr = -2 * torch.log(torch.tan(limit_min * t + limit_max))
        # Transform logsnr to a, sigma.
        return torch.sqrt(torch.sigmoid(torch.tensor(logsnr))), torch.sqrt(torch.sigmoid(-torch.tensor(logsnr)))

    def get_alpha_sigma(self, t):
        freq_scaling = self.get_frequency_scaling(t)
        # print('freq_scaling',freq_scaling[0],freq_scaling[0].shape)
        # print('freq_scaling',freq_scaling)
        a, sigma = self.get_noise_scaling_cosine(t)
        a = a.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        alpha = a * freq_scaling # Combine dissipation and scaling. 
        return alpha , sigma

    def diffuse(self, x, t):
        x_freq = dct.dct(x, norm = 'ortho')
        x_freq = dct.dct(x_freq.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2)
        alpha, sigma = self.get_alpha_sigma(t) 
        eps = torch.randn_like(x)
        sigma = sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # Since we chose sigma to be a scalar, eps does not need to be # passed through a DCT/IDCT in this case.
        x_freq = alpha.to("cuda") * x_freq.to("cuda")
        x_freq = dct.idct(x_freq, norm='ortho')
        z_t = dct.idct(x_freq.permute(0,1,3,2), norm='ortho').permute(0,1,3,2) + sigma * eps
        # z_t = dct.idct(alpha.to("cuda") * x_freq.) + sigma * eps
        # grid = (make_grid(z_t) + 1) / 2
        # path = os.path.join(
        #     './logs/DDPM_CIFAR10_EPS', 'sample', '%d_noise.png' % 0)
        # save_image(grid, path)
        return z_t , eps

    def loss(self, x):
        # t = torch.tensor(np.random.uniform())
        t = torch.randint(self.T, size=(x.shape[0], ), device=x.device)
        z_t, noise = self.diffuse(x, t/self.T)
        loss = F.mse_loss(self.model(z_t, t), noise, reduction='none')
        return loss.mean()

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        # t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        # noise = torch.randn_like(x_0)
        # x_t = (
        #     extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
        #     extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return self.loss(x_0)

class HeatDiffusionSampler(nn.Module):
    
    def __init__(self, model, beta_1, beta_T, T, img_dim=32):

        super().__init__()
        self.model = model
        self.T = T
        self.img_dim = img_dim
        self.sigma_blur_max = 10
        # compute frequencies
        self.freqs = torch.pi * torch.linspace(0, self.img_dim-1, self.img_dim) / self.img_dim
        # self.labda = torch.square(self.freqs[None, :, None, None]) + torch.square(self.freqs[None, None, :, None])
        self.labda = torch.square(self.freqs[None, None, :, None]) + torch.square(self.freqs[None, None, None, :])
   
    def get_frequency_scaling(self, t, min_scale=0.001):
        # compute dissipation time
        sigma_t = self.sigma_blur_max * torch.sin(t * torch.pi / 2).pow(2)
        
        dissipation_time = sigma_t.pow(2) / 2
        dissipation_time = dissipation_time.unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
        # compute scaling for frequencies
        scaling = torch.exp(-self.labda.cuda() * dissipation_time) * (1 - min_scale) 
        scaling = scaling + min_scale
        return scaling

    def get_noise_scaling_cosine(self, t, logsnr_min=-10, logsnr_max=10): 
        limit_max = torch.arctan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
        limit_min = torch.arctan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - limit_max
        logsnr = -2 * torch.log(torch.tan(limit_min * t + limit_max))
        # Transform logsnr to a, sigma.
        return torch.sqrt(torch.sigmoid(torch.tensor(logsnr))), torch.sqrt(torch.sigmoid(-torch.tensor(logsnr)))

    def get_alpha_sigma(self, t):
        freq_scaling = self.get_frequency_scaling(t)
        # print('freq_scaling',freq_scaling[0],freq_scaling[0].shape)
        # print('freq_scaling',freq_scaling)
        a, sigma = self.get_noise_scaling_cosine(t)
        a = a.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        alpha = a * freq_scaling # Combine dissipation and scaling. 
        return alpha , sigma
    
    def denoise(self, z_t, t, delta=1e-8):
        alpha_s, sigma_s = self.get_alpha_sigma((t - 1 / self.T).cuda()) 
        alpha_t , sigma_t = self.get_alpha_sigma(t.cuda())
        alpha_s = alpha_s.squeeze()
        alpha_t = alpha_t.squeeze()
        alpha_ts = alpha_t / alpha_s
        sigma2_ts = sigma_t**2 - alpha_ts**2 * sigma_s**2
        sigma_ts = torch.sqrt(sigma_t**2 - alpha_ts**2 * sigma_s**2)

        sigma_t_to_s_square = 1 / torch.clip((1/ torch.clip(sigma_s**2, min=delta) + alpha_ts**2 / torch.clip(sigma2_ts, min=delta)), min = delta)
        
        # Estimate epsilon in visual domain then map it back to Freq Space
        estimate_eps = self.model(z_t, (t*self.T).int())
        estimate_eps_freq = dct.dct(estimate_eps, norm = "ortho").cuda()
        estimate_eps_freq = dct.dct(estimate_eps_freq.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2).cuda()

        # DCT to Freq Space
        u_t = dct.dct(z_t, norm = "ortho").cuda()
        u_t = dct.dct(u_t.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2).cuda()

        u_t_bis = dct.dct(z_t - estimate_eps * sigma_t, norm = "ortho").cuda()
        u_t_bis = dct.dct(u_t_bis.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2).cuda()

        #TOCHECK
        u_x_bis = u_t_bis / alpha_t
        u_x = (u_t - estimate_eps_freq * sigma_t) / alpha_t
        print("Check difference bis u_x versions :", (u_x - u_x_bis).mean())
        mu_t_to_s = sigma_t_to_s_square * ((alpha_ts / sigma2_ts) * u_t + (alpha_s / sigma_s**2) * u_x_bis)

        # Sample new random noise
        eps = torch.randn_like(mu_t_to_s)
        eps_freq = dct.dct(eps, norm = "ortho").cuda()
        eps_freq = dct.dct(eps_freq.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2).cuda()

        # Sample z from the newly define Gaussian "s"
        u_s = mu_t_to_s + eps_freq * torch.sqrt(sigma_t_to_s_square)
        
        # Proj back to Visual Space
        z_s = dct.idct(u_s, norm = "ortho")
        z_s = dct.idct(z_s.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2)

        z_x = dct.idct(u_x, norm = "ortho")
        z_x = dct.idct(z_x.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2)

        return {"sample": z_s, "pred_xstart": z_x}

    def predict_xstart_from_eps(self, x, t, eps):

        alpha_t , sigma_t = self.get_alpha_sigma(t/self.T)
        sigma_t = sigma_t.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        eps_freq = dct.dct(eps, norm = "ortho").cuda()
        eps_freq = dct.dct(eps_freq.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2).cuda()

        # x_t = dct.dct(x - sigma_t * eps, norm = "ortho").cuda()
        x_t = dct.dct(x, norm = "ortho").cuda()
        x_t = dct.dct(x_t.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2).cuda()
        # x_start_freq = x_t / alpha_t
        x_start_freq = (x_t - sigma_t * eps_freq) / alpha_t

        x_start = dct.idct(x_start_freq, norm = "ortho")
        x_start = dct.idct(x_start.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2)

        return x_start


    def ddim_sample(
            self,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            x_orig = None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        alpha_prev, sigma_prev = self.get_alpha_sigma(((t - 1) / self.T).cuda()) 
        alpha_t , sigma_t = self.get_alpha_sigma(t.cuda() /  self.T)
        sigma_t = sigma_t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        sigma_prev = sigma_prev.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        eps = self.model(x, (t).int())

        sigma = (
                eta 
                * (sigma_prev/sigma_t)
                * (sigma_t/alpha_prev)
        )

        x_start = self.predict_xstart_from_eps(x, t, eps)
        x_start_freq = dct.dct(x_start, norm = "ortho").cuda()
        x_start_freq = dct.dct(x_start_freq.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2).cuda()
        # Equation 12.
        
        noise = torch.randn_like(x)
        mean_pred_freq = x_start_freq * alpha_prev
        mean_pred = dct.idct(mean_pred_freq, norm = "ortho").cuda()
        mean_pred = dct.idct(mean_pred.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2)

        mean_pred = mean_pred + torch.sqrt(sigma_prev ** 2 - sigma ** 2) * eps
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        # sample = dct.idct(sample, norm = "ortho").cuda()
        # sample = dct.idct(sample.permute(0,1,3,2), norm = "ortho").permute(0,1,3,2)
        return {"sample": sample, "pred_xstart": x_start}


    def ddim_sample_loop(
            self,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            x_orig = None,
            eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                x_orig = x_orig,
                eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
            self,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            x_orig = None,
            eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(self.model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.T))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdms
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    x_orig = x_orig,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def ddim_reverse_sample(
            self,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            x_orig = None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        XS:
            Modified this function to include classifier guidance (i.e. condition_score).
            Note that the (source) label information is included in model_kwargs.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            x,
            t,
            x_orig = x_orig
            # model,
            # clip_denoised=clip_denoised,
            # denoised_fn=denoised_fn,
            # model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        # XS: the following is the same as _predict_eps_from_xstart
        # print("pred=orig", out["pred_xstart"] == x_orig)
        eps = (extract(self.sqrt_recip_alphas_bar, t, x.shape) * x
                      - out["pred_xstart"]
              ) / extract(self.sqrt_recipm1_alphas_bar, t, x.shape)
        alpha_bar_next = extract(self.alphas_bar_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
                out["pred_xstart"] * torch.sqrt(alpha_bar_next)
                + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample_loop(
            self,
            image,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            x_orig = None,
            eta=0.0,
    ):
        """
        XS: Encode image into latent using DDIM ODE.
        """
        final = None
        for sample in self.ddim_reverse_sample_loop_progressive(
                image,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                x_orig = x_orig,
                eta=eta,
        ):
            final = sample
        return final["sample"]


    def ddim_reverse_sample_loop_progressive(
            self,
            image,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            x_orig = None,
            eta=0.0,
    ):
        """
        XS: Use DDIM to perform encoding / inference, until isotropic Gaussian.
        """
        if device is None:
            device = next(self.model.parameters()).device
        shape = image.shape
        indices = list(range(self.T))

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_reverse_sample(
                    image,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    x_orig = x_orig,
                    eta=eta,
                )
                yield out
                image = out["sample"]

    def forward(self, z_T):
        z_t = z_T
        for t in reversed(range(1,self.T)):
            t = torch.tensor(t, device=z_T.device).unsqueeze(0)
            out = self.denoise(z_t, t/self.T)
            z_t = out['sample']
        return torch.clip(z_t, -10, 10)


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='epsilon', var_type='fixedlarge'):
                 
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        alphas_bar_next = F.pad(alphas_bar, [0, 1], value=0)[1:]
        sqrt_alphas_bar = torch.sqrt(alphas_bar)
        sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
        log_one_minus_alphas_bar = torch.log(1.0 - alphas_bar)



        self.register_buffer(
            'alphas', alphas)
        self.register_buffer(
            'alphas_bar', alphas_bar)
        self.register_buffer(
            'alphas_bar_prev', alphas_bar_prev)
        self.register_buffer(
            'alphas_bar_next', alphas_bar_next)
        self.register_buffer(
            'sqrt_alphas_bar', sqrt_alphas_bar)
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', sqrt_one_minus_alphas_bar)
        self.register_buffer(
            'log_one_minus_alphas_bar', log_one_minus_alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, torch.exp(posterior_log_var_clipped), posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                       extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
                       - pred_xstart
               ) / extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape)

    def p_mean_variance(self, x_t, t, x_orig = None):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)
        model_variance = torch.exp(model_log_var)

        # Mean parameterization
        if x_orig is not None:
            x_0 = x_orig
            model_mean, _, _ = self.q_mean_variance(x_orig, x_t, t)
        elif self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_var,
            "pred_xstart": x_0,
        }
        # model_mean, model_log_var
    

    def ddim_sample(
            self,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            x_orig = None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        # if x_orig is not None:
        out = self.p_mean_variance(
            x,
            t,
            x_orig = x_orig
            # model,
            # clip_denoised=clip_denoised,
            # denoised_fn=denoised_fn,
            # model_kwargs=model_kwargs,
        )

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = extract(self.alphas_bar, t, x.shape)
        alpha_bar_prev = extract(self.alphas_bar_prev, t, x.shape)
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


    def ddim_sample_loop(
            self,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            x_orig = None,
            eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                x_orig = x_orig,
                eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
            self,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            x_orig = None,
            eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(self.model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.T))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdms
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    x_orig = x_orig,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def ddim_reverse_sample(
            self,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            x_orig = None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        XS:
            Modified this function to include classifier guidance (i.e. condition_score).
            Note that the (source) label information is included in model_kwargs.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            x,
            t,
            x_orig = x_orig
            # model,
            # clip_denoised=clip_denoised,
            # denoised_fn=denoised_fn,
            # model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        # XS: the following is the same as _predict_eps_from_xstart
        # print("pred=orig", out["pred_xstart"] == x_orig)
        eps = (extract(self.sqrt_recip_alphas_bar, t, x.shape) * x
                      - out["pred_xstart"]
              ) / extract(self.sqrt_recipm1_alphas_bar, t, x.shape)
        alpha_bar_next = extract(self.alphas_bar_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
                out["pred_xstart"] * torch.sqrt(alpha_bar_next)
                + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample_loop(
            self,
            image,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            x_orig = None,
            eta=0.0,
    ):
        """
        XS: Encode image into latent using DDIM ODE.
        """
        final = None
        for sample in self.ddim_reverse_sample_loop_progressive(
                image,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                x_orig = x_orig,
                eta=eta,
        ):
            final = sample
        return final["sample"]


    def ddim_reverse_sample_loop_progressive(
            self,
            image,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            x_orig = None,
            eta=0.0,
    ):
        """
        XS: Use DDIM to perform encoding / inference, until isotropic Gaussian.
        """
        if device is None:
            device = next(self.model.parameters()).device
        shape = image.shape
        indices = list(range(self.T))

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_reverse_sample(
                    image,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    x_orig = x_orig,
                    eta=eta,
                )
                yield out
                image = out["sample"]

    def forward(self, x_T, clip_value = 1., clip_sampling = False):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            out = self.p_mean_variance(x_t=x_t, t=t)
            mean = out['mean']
            log_var = out['log_variance']
            # no noise when' t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
            if clip_sampling:
                x_t = torch.clip(x_t, -clip_value, clip_value)
        x_0 = x_t
        return torch.clip(x_0, -clip_value, clip_value)
