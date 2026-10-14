import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDiffusion(nn.Module):
    def __init__(self, model, T, schedule):
        super().__init__()
        self.visual = False
        if self.visual:
            self.num = 0
        self.model = model
        self.T = T
        self.schedule = schedule
        betas = self.schedule.get_betas()
        self.register_buffer("betas", betas.float())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)  # \bar\alpha_t

        # alphas_bar_prev: padded version shifting right with 1 at index 0 (alpha_bar_{-1}=1)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1.0)[:T]

        gammas = alphas_bar  # keep original name (you used gammas for alphas_bar)

        # coefficients used by original DDPM formulas
        self.register_buffer('coeff1', torch.sqrt(1.0 / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1.0 - alphas) / torch.sqrt(1.0 - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))

        # store schedules
        self.register_buffer("gammas", gammas)
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)

        # sqrt terms
        self.register_buffer("sqrt_one_minus_gammas", torch.sqrt(1.0 - gammas))
        self.register_buffer("sqrt_gammas", torch.sqrt(gammas))
        self.register_buffer("sqrt_alphas_bar_prev", torch.sqrt(alphas_bar_prev))
        self.register_buffer("one_minus_alphas_bar_prev", (1.0 - alphas_bar_prev))

    def extract_(self, a, t, x_shape):
        # legacy helper in your code: gather by t where a is a cpu Tensor maybe
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def extract(self, v, t, x_shape):
        """
        Extract some coefficients at specified timesteps, then reshape to
        [batch_size, 1, 1, 1, ...] for broadcasting.
        """
        out = torch.gather(v, index=t, dim=0).float()
        return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return self.extract(self.coeff1, t, x_t.shape) * x_t - self.extract(self.coeff2, t, x_t.shape) * eps

    def predict_eps_from_x0(self, x_t, t, x_0):
        # eps = (x_t - sqrt(alpha_bar_t) * x_0) / sqrt(1 - alpha_bar_t)
        return (x_t - self.extract(self.sqrt_gammas, t, x_t.shape) * x_0) / self.extract(
            self.sqrt_one_minus_gammas, t, x_t.shape
        )

    def x0_p_mean_variance(self, x_t, cond_, t):
        # posterior variance used in DDPM step
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = self.extract(var, t, x_t.shape)

        x0_pred = self.model(torch.cat((x_t, cond_), dim=1), t)
        eps = self.predict_eps_from_x0(x_t, t, x0_pred)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def noisy_image(self, t, y):
        noise = torch.randn_like(y)
        sqrt_gammas = self.sqrt_gammas.to(t.device)
        sqrt_one_minus_gammas = self.sqrt_one_minus_gammas.to(t.device)

        y_noisy = (
            self.extract_(sqrt_gammas, t, y.shape) * y
            + self.extract_(sqrt_one_minus_gammas, t, noise.shape) * noise
        )
        return y_noisy, noise

    def ddpm_sample(self, x_T, cond, pred_en, img_d):
        """
        Classic DDPM sampling (randomized) — similar to your original forward.
        """
        x_t = x_T
        cond_ = cond
        pred_en_ = pred_en
        img_d_ =img_d
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0],], dtype=torch.long) * time_step
            mean, var = self.x0_p_mean_variance(x_t=x_t, t=t, cond_=torch.cat((cond_, pred_en_, img_d_), dim=1))
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return x_0


    def ddim_sample(self, x_T, cond, pred_en, img_d, eta: float = 0.0, timesteps: torch.Tensor = None):
        """
        DDIM sampling.

        Args:
            x_T: initial noise at time T (shape [b, c, h, w])
            cond, pred_en: conditioning tensors concatenated inside
            eta: noise scale for stochasticity (eta=0 -> deterministic DDIM)
            timesteps: optional 1D tensor of timesteps to use (increasing order).
                       If None, uses full [0..T-1].
        Returns:
            x_0: sampled image
        """
        device = x_T.device
        b = x_T.shape[0]
        x_t = x_T

        # timesteps as a sequence of integer timesteps we will traverse backwards
        if timesteps is None:
            # use all timesteps by default
            ts = torch.arange(self.T, dtype=torch.long, device=device)
        else:
            ts = torch.arange(timesteps, dtype=torch.long, device=device)

        # ensure we iterate in reverse order
        seq = list(reversed(ts.tolist()))

        for time_step in seq:
            t = x_t.new_ones([b,], dtype=torch.long) * time_step

            # predict x0 from model
            x0_pred = self.model(torch.cat((x_t, torch.cat((cond, pred_en, img_d), dim=1)), dim=1), t)

            # compute epsilon (pred noise)
            eps = self.predict_eps_from_x0(x_t, t, x0_pred)

            # gather alpha_bar_t and alpha_bar_{t-1}
            alpha_bar_t = self.extract(self.gammas, t, x_t.shape)  # \bar\alpha_t
            alpha_bar_prev = self.extract(self.alphas_bar_prev, t, x_t.shape)  # \bar\alpha_{t-1}

            # sigma_t for stochastic DDIM (from DDIM paper)
            # sigma_t = eta * sqrt( (1 - alpha_bar_prev)/(1 - alpha_bar_t) ) * sqrt(1 - alpha_bar_t/alpha_bar_prev )
            # compute factor safely (clamp for numerical stability)
            one_minus_ab_t = (1.0 - alpha_bar_t)
            one_minus_ab_prev = (1.0 - alpha_bar_prev)

            # avoid division by zero; clamp ratio
            frac = (one_minus_ab_prev / (one_minus_ab_t + 1e-8)).clamp(min=1e-12)
            temp = (1.0 - (alpha_bar_t / (alpha_bar_prev + 1e-8))).clamp(min=0.0)

            sigma_t = eta * torch.sqrt(frac * temp)

            # the deterministic part scale
            # coef_x0 = sqrt(alpha_bar_prev)
            coef_x0 = torch.sqrt(alpha_bar_prev)
            # coef_eps = sqrt(1 - alpha_bar_prev - sigma_t^2)
            # ensure non-neg inside sqrt
            coef_eps_sq = (1.0 - alpha_bar_prev - sigma_t ** 2).clamp(min=0.0)
            coef_eps = torch.sqrt(coef_eps_sq)

            # compute x_{t-1}
            x_prev = coef_x0 * x0_pred + coef_eps * eps

            if eta > 0.0 and time_step > 0:
                noise = torch.randn_like(x_t)
                x_prev = x_prev + sigma_t * noise

            # set for next iter
            x_t = x_prev
            # sanity check
            assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor at step {time_step}."

        x_0 = x_t
        return x_0


    def forward(self, x_T, cond, pred_en, img_d, timesteps: torch.Tensor = None, eta: float = 0.0, use_ddpm: bool = False):
        """
        Default forward uses DDIM (eta=0 -> deterministic). Set use_ddpm=True to use original DDPM.
        Set eta>0 to add stochasticity in DDIM.
        timesteps: optional sequence of timesteps (in increasing order) to use for sampling. If None, uses all.
        """
        if use_ddpm:
            return self.ddpm_sample(x_T, cond, pred_en, img_d)
        else:
            return self.ddim_sample(x_T, cond, pred_en, img_d, eta=eta, timesteps=timesteps)
