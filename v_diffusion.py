import torch
import torch.nn.functional as F

class UniformDiffusion:

    def __init__(self, net, betas, time_scale=1, sampler="ddpm"):
        super().__init__()
        self.net_ = net
        self.time_scale = time_scale
        betas = betas.type(torch.float64)
        self.num_timesteps = int(betas.shape[0])

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64, device=betas.device), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.posterior_variance = posterior_variance
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)

        if sampler == "ddpm":
            self.p_sample = self.p_sample_ddpm
        else:
            self.p_sample = self.p_sample_clipped

    def inference(self, x, t, extra_args):
        return self.net_(x, t * self.time_scale, **extra_args)

    def p_loss(self, x_0, t, extra_args, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_t, sigma_t = self.get_alpha_sigma(x_0, t)
        z = alpha_t * x_0 + sigma_t * noise
        v_recon = self.inference(z.float(), t.float(), extra_args)
        v = alpha_t * noise - sigma_t * x_0
        return F.mse_loss(v_recon, v.float())

    def q_posterior(self, x_0, x_t, t):
        mean = E_(self.posterior_mean_coef1, t, x_t.shape) * x_0 \
               + E_(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var = E_(self.posterior_variance, t, x_t.shape)
        log_var_clipped = E_(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_mean_variance(self, x, t, extra_args, clip_denoised):
        v = self.inference(x.float(), t.float(), extra_args).double()
        alpha_t, sigma_t = self.get_alpha_sigma(x, t)
        x_recon = alpha_t * x - sigma_t * v
        if clip_denoised:
            x_recon = x_recon.clamp(min=-1, max=1)
        mean, var, log_var = self.q_posterior(x_recon, x, t)
        return mean, var, log_var

    def p_sample_ddpm(self, x, t, extra_args, clip_denoised=True, **kwargs):
        mean, _, log_var = self.p_mean_variance(x, t, extra_args, clip_denoised)
        noise = torch.randn_like(x)
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape)
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    def p_sample_clipped(self, x, t, extra_args, eta=0, clip_denoised=True, clip_value=3):
        v = self.inference(x.float(), t, extra_args)
        alpha, sigma = self.get_alpha_sigma(x, t)
        pred = (x * alpha - v * sigma)
        if clip_denoised:
            pred = pred.clip(-clip_value, clip_value)
        eps = (x - alpha * pred) / sigma
        if clip_denoised:
            eps = eps.clip(-clip_value, clip_value)

        t_mask = (t > 0)
        if t_mask.any().item():
            if not t_mask.all().item():
                raise Exception()
            alpha_, sigma_ = self.get_alpha_sigma(x, (t - 1).clip(min=0))
            ddim_sigma = eta * (sigma_ ** 2 / sigma ** 2).sqrt() * \
                         (1 - alpha ** 2 / alpha_ ** 2).sqrt()
            adjusted_sigma = (sigma_ ** 2 - ddim_sigma ** 2).sqrt()
            pred = pred * alpha_ + eps * adjusted_sigma
            if eta:
                pred += torch.randn_like(pred) * ddim_sigma
        return pred

    @torch.no_grad()
    def p_sample_loop(self, x, extra_args, eta=0):
        mode = self.net_.training
        self.net_.eval()
        for i in reversed(range(self.num_timesteps)):
            x = self.p_sample(
                x,
                torch.full((x.shape[0],), i, dtype=torch.int64).to(x.device),
                extra_args,
                eta=eta,
            )
        self.net_.train(mode)
        return x

    def get_alpha_sigma(self, x, t):
        alpha = E_(self.sqrt_alphas_cumprod, t, x.shape)
        sigma = E_(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return alpha, sigma


class GaussianDiffusionDefault(GaussianDiffusion):
    def __init__(self, net, betas, time_scale=1, gamma=0.3):
        super().__init__(net, betas, time_scale)
        self.gamma = gamma

    def uniform_sample(self, shape, device):
        return torch.rand(shape, device=device) * 2 - 1  # Uniform distribution in the range [-1, 1]

    def distill_loss(self, student_diffusion, x, t, extra_args, eps=None, student_device=None):
        if eps is None:
            eps = self.uniform_sample(x.shape, x.device)  # Sample from a uniform distribution

        with torch.no_grad():
            alpha, sigma = self.get_alpha_sigma(x, t + 1)
            z = alpha * x + sigma * eps
            alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // 2)
            alpha_1, sigma_1 = self.get_alpha_sigma(x, t)
            v = self.inference(z.float(), t.float() + 1, extra_args).double()
            rec = (alpha * z - sigma * v).clip(-1, 1)
            z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
            v_1 = self.inference(z_1.float(), t.float(), extra_args).double()
            x_2 = (alpha_1 * z_1 - sigma_1 * v_1).clip(-1, 1)
            eps_2 = (z - alpha_s * x_2) / sigma_s
            v_2 = alpha_s * eps_2 - sigma_s * x_2
            if self.gamma == 0:
                w = 1
            else:
                w = torch.pow(1 + alpha_s / sigma_s, self.gamma)
        v = student_diffusion.net_(z.float(), t.float() * self.time_scale, **extra_args)
        my_rec = (alpha_s * z - sigma_s * v).clip(-1, 1)
        return F.mse_loss(w * v.float(), w * v_2.float())
