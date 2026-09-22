import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8):
        super().__init__()

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)


def extract(a, t, x_shape: list):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, 1, 1)


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.arange(steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_conditioning(x, conditions, action_dim: int):
    x[:, :conditions.shape[0], action_dim:] = conditions

    return x


class WeightedStateLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ, masks: torch.Tensor):
        loss = self._loss(pred, targ)
        if masks is not None:
            loss = loss * masks[:, :, None].float()
        weights = self.weights[None].expand_as(loss)
        if masks is not None:
            weights = weights * masks[:, :, None]
        weighted_loss = (loss * weights).sum() / weights.sum().clamp_min(1)
        return weighted_loss, {'a0_loss': weighted_loss}


class WeightedStateL2(WeightedStateLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'state_l2': WeightedStateL2,
}


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
            Conv1dBlock(out_channels, out_channels, kernel_size, mish),
        ])

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            dim=128,
            dim_mults=(1, 2, 4),
            returns_condition=True,
            condition_dropout=0.1,
            calc_energy=False,
            kernel_size=5,
            num_categories=0,
            category_dropout=0.15,
            prompt_dim=3,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        self.num_categories = num_categories
        self.category_dropout = category_dropout
        self.category_embedding = nn.Embedding(num_categories + 1, dim)
        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        self.returns_mlp = nn.Sequential(
            nn.Linear(prompt_dim, dim),
            act_fn,
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        embed_dim = 2 * dim
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size,
                                      mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size,
                                      mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon,
                                                kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon,
                                                kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon,
                                      kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size,
                                      mish=mish),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time, returns: torch.Tensor = torch.ones(1, 1), use_dropout: bool = True,
                force_dropout: bool = False, category_ids=None, force_category_dropout=False):

        x = torch.permute(x, (0, 2, 1))

        # print(returns.shape)
        t = self.time_mlp(time)
        if category_ids is None:
            category_ids = torch.full((x.shape[0],), self.num_categories, device=x.device, dtype=torch.long)
        category_ids = category_ids.to(device=x.device, dtype=torch.long)
        if category_ids.shape != (x.shape[0],):
            raise ValueError("category_ids must have shape [batch]")
        category_ids = torch.where((category_ids >= 0) & (category_ids < self.num_categories),
                                   category_ids, self.num_categories)
        if force_category_dropout:
            category_ids = torch.full_like(category_ids, self.num_categories)
        elif use_dropout and self.training:
            category_ids = torch.where(torch.rand(x.shape[0], device=x.device) < self.category_dropout,
                                       self.num_categories, category_ids)
        # [B, D] industry embedding is added to diffusion-time embedding; the
        # result is concatenated with the return/CPA prompt and injected in every block.
        t = t + self.category_embedding(category_ids)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout and self.training:
                keep = torch.rand(returns_embed.shape[0], 1, device=x.device) >= self.condition_dropout
                returns_embed = returns_embed * keep
            if force_dropout:
                returns_embed = 0 * returns_embed

            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for models in self.downs:
            resnet = models[0]
            resnet2 = models[1]
            downsample = models[2]
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for models in self.ups:
            resnet = models[0]
            resnet2 = models[1]
            upsample = models[2]
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = torch.permute(x, (0, 2, 1))

        return x


class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
                 clip_denoised=False, predict_epsilon=True, hidden_dim=256,
                 loss_discount=1.0, returns_condition=False,
                 condition_guidance_w=0.1, category_guidance_w=1.0):
        super().__init__()

        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.inv_model = nn.Sequential(
            nn.Linear(4 * self.observation_dim + 3 + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        self.inv_category_embedding = nn.Embedding(model.num_categories + 1, 16)
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.category_guidance_w = category_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)

    def get_loss_weights(self, discount):

        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.matmul(discounts[:, None], dim_weights[None, :])

        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):

        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns, category_ids=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself

            epsilon_uncond = self.model(x, cond, t, returns, use_dropout=False,
                                        force_dropout=True, force_category_dropout=True)
            epsilon_prompt = self.model(x, cond, t, returns, use_dropout=False,
                                        force_category_dropout=True)
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False, category_ids=category_ids)
            epsilon = (epsilon_uncond + self.condition_guidance_w * (epsilon_prompt - epsilon_uncond)
                       + self.category_guidance_w * (epsilon_cond - epsilon_prompt))
        else:
            epsilon = self.model(x, cond, t, returns, use_dropout=False, category_ids=category_ids)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, cond, t, returns, category_ids=None):
        with torch.no_grad():
            b, _, _ = x.shape
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns, category_ids=category_ids)
            noise = torch.randn_like(x, device=x.device)
            nonzero_mask = (1 - (t == 0).float()).reshape(b, 1, 1)
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, shape, cond, returns, category_ids=None):
        with torch.no_grad():
            batch_size = shape[0]
            x = torch.randn(shape[0], shape[1], shape[2], device=cond.device)

            x = apply_conditioning(x, cond, 0)

            for i in range(self.n_timesteps - 1, -1, -1):
                timesteps = torch.ones(batch_size,
                                       device=cond.device) * i
                x = self.p_sample(x, cond, timesteps, returns, category_ids)

                x = apply_conditioning(x, cond, 0)

            return x

    def conditional_sample(self, cond, returns, category_ids=None):
        shape = (1, self.horizon, self.observation_dim)
        return self.p_sample_loop(shape, cond, returns, category_ids)

    def forward(self, cond, returns, category_ids=None):
        return self.conditional_sample(cond, returns, category_ids)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(t.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(t.device)
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None, masks=None, category_ids=None):
        noise = torch.randn_like(x_start, device=x_start.device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # Train on random observed prefixes, matching receding-horizon inference.
        lengths = masks.sum(-1)
        prefix_lengths = (torch.rand(len(lengths), device=x_start.device) * lengths).long() + 1
        known = torch.arange(x_start.shape[1], device=x_start.device)[None] < prefix_lengths[:, None]
        x_noisy = torch.where(known[:, :, None], x_start, x_noisy)
        x_noisy = x_noisy * masks[:, :, None]
        prediction_masks = masks & ~known
        t = t.to(x_noisy.device)
        x_recon = self.model(x_noisy, cond, t, returns, category_ids=category_ids)

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise, prediction_masks)
        else:
            loss, info = self.loss_fn(x_recon, x_start, prediction_masks)

        return loss, info

    def loss(self, x, cond, returns, masks, category_ids, q=None, exploration=None):

        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns, masks, category_ids)
        # Calculating inv loss
        states = x[:, :, self.action_dim:] * masks[:, :, None]
        # [B,H,S] -> [B,H,4S]: previous one, previous two, current, next.
        prev1 = F.pad(states[:, :-1], (0, 0, 1, 0))
        prev2 = F.pad(states[:, :-2], (0, 0, 2, 0))
        successor = F.pad(states[:, 1:], (0, 0, 0, 1))
        x_comb_t = torch.cat([prev1, prev2, states, successor], dim=-1)[masks]
        a_t = x[:, :, :self.action_dim][masks]
        labels = returns[:, None, :].expand(-1, x.shape[1], -1)[masks]
        categories = category_ids[:, None].expand(-1, x.shape[1])[masks]
        categories = torch.where(torch.rand_like(categories, dtype=torch.float32) < self.model.category_dropout,
                                 self.model.num_categories, categories)
        # -1 denotes unknown compliance (used for local counterfactual targets).
        inverse_labels = labels.clone()
        unknown = torch.rand(len(labels), device=x.device) < self.model.condition_dropout
        inverse_labels[unknown, 2] = -1.
        if a_t.numel() == 0:
            inv_loss = sum(p.sum() * 0 for p in self.inv_model.parameters())
        else:
            # Keep a clean expert anchor even when pseudo-label proposals are accepted.
            pred_a_t = self.inverse(x_comb_t, inverse_labels, categories)
            inv_loss = F.mse_loss(pred_a_t, a_t)
            if q is not None:
                from .exploration import propose_actions
                targets, pseudo_labels, accepted = propose_actions(
                    q, x_comb_t[:, 2 * self.observation_dim:3 * self.observation_dim],
                    a_t, category_ids[:, None].expand(-1, x.shape[1])[masks], labels, **(exploration or {}))
                info['accepted_proposals'] = accepted.sum().detach()
                if accepted.any():
                    pred_aug = self.inverse(x_comb_t[accepted], pseudo_labels[accepted], categories[accepted])
                    inv_loss = inv_loss + 0.25 * F.mse_loss(pred_aug, targets[accepted])
        loss = (1 / 2) * (diffuse_loss + inv_loss)

        return loss, info, (diffuse_loss, inv_loss)


    def inverse(self, states, prompts, categories):
        return self.inv_model(torch.cat([states, prompts, self.inv_category_embedding(categories)], dim=-1))


class DFUSER(nn.Module):
    """State-sequence diffusion with a conditional inverse bidding model."""
    def __init__(self, dim_obs=16, dim_actions=1, lr=1e-4, network_random_seed=200,
                 ACTION_MAX=10., ACTION_MIN=0., step_len=48, n_timesteps=10,
                 num_categories=0, category_dropout=0.15, condition_dropout=0.15,
                 condition_guidance_w=1.2, category_guidance_w=1.0, model_dim=128,
                 metadata=None):
        super().__init__()
        if n_timesteps < 1 or model_dim < 8 or model_dim % 8:
            raise ValueError("n_timesteps must be positive and model_dim a multiple of 8")
        if step_len < 4 or step_len % 4:
            raise ValueError("step_len must be a positive multiple of 4")
        if not 0 <= category_dropout <= 1 or not 0 <= condition_dropout <= 1:
            raise ValueError("Dropout probabilities must be in [0, 1]")
        if ACTION_MIN < 0 or ACTION_MAX <= ACTION_MIN:
            raise ValueError("Invalid action bounds")
        torch.manual_seed(network_random_seed)
        self.config = dict(dim_obs=dim_obs, dim_actions=dim_actions, lr=lr,
                           network_random_seed=network_random_seed, ACTION_MAX=ACTION_MAX,
                           ACTION_MIN=ACTION_MIN, step_len=step_len, n_timesteps=n_timesteps,
                           num_categories=num_categories, category_dropout=category_dropout,
                           condition_dropout=condition_dropout, condition_guidance_w=condition_guidance_w,
                           category_guidance_w=category_guidance_w, model_dim=model_dim)
        self.metadata = metadata
        self.step_len = step_len
        self.num_of_states = dim_obs
        self.num_of_actions = dim_actions
        self.ACTION_MIN, self.ACTION_MAX = ACTION_MIN, ACTION_MAX
        model = TemporalUnet(step_len, dim_obs, dim_actions, dim=model_dim,
                             num_categories=num_categories, category_dropout=category_dropout,
                             condition_dropout=condition_dropout)
        self.diffuser = GaussianInvDynDiffusion(
            model, step_len, dim_obs, dim_actions, n_timesteps=n_timesteps,
            clip_denoised=False, returns_condition=True,
            condition_guidance_w=condition_guidance_w, category_guidance_w=category_guidance_w)
        self.optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=lr)
        self.last_metrics = {}

    def trainStep(self, states, actions, returns, masks, category_ids=None, q=None, exploration=None):
        self.train()
        if category_ids is None:
            category_ids = torch.full((len(states),), self.config['num_categories'],
                                      device=states.device, dtype=torch.long)
        states = states * masks[:, :, None]
        x = torch.cat([actions, states], dim=-1)
        self.optimizer.zero_grad()
        loss, info, components = self.diffuser.loss(x, states[:, :1], returns, masks, category_ids, q, exploration)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.)
        self.optimizer.step()
        self.last_metrics = {k: float(v.detach()) for k, v in info.items()}
        return loss.detach(), tuple(v.detach() for v in components)

    @torch.no_grad()
    def forward(self, x, category_id=None, target_return=1., cpa=0., target_compliance=1.):
        if self.metadata is None:
            raise ValueError("Inference requires train-set normalization metadata from a v2 checkpoint")
        if not all(math.isfinite(v) for v in (target_return, cpa, target_compliance)) or cpa < 0:
            raise ValueError("Inference prompts must be finite and CPA nonnegative")
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device=device, dtype=torch.float32).reshape(self.step_len, self.num_of_states + 1)
        current = int(x[0, -1].item()) + 1
        if not 1 <= current <= self.step_len:
            raise ValueError("Invalid decision time step")
        mean = x.new_tensor(self.metadata['state_mean'])
        std = x.new_tensor(self.metadata['state_std'])
        conditions = (x[:current, :-1] - mean) / std
        categories = self.metadata['categories']
        category = categories.index(category_id) if category_id in categories else len(categories)
        category_tensor = torch.tensor([category], device=device)
        prompts = x.new_tensor([[target_return, cpa / self.metadata['cpa_scale'], target_compliance]])
        planned = self.diffuser(conditions, prompts, category_tensor)
        # Terminal successor is a zero sentinel, matching terminal inverse training.
        successor = planned[:, current] if current < self.step_len else torch.zeros_like(conditions[-1:])
        prev1 = conditions[-2:-1] if current > 1 else torch.zeros_like(successor)
        prev2 = conditions[-3:-2] if current > 2 else torch.zeros_like(successor)
        context = torch.cat([prev1, prev2, conditions[-1:], successor], dim=-1)
        action = self.diffuser.inverse(context, prompts, category_tensor)
        if not torch.isfinite(action).all():
            raise RuntimeError("Non-finite generated bid multiplier")
        return action[0].clamp(self.ACTION_MIN, self.ACTION_MAX).cpu()

    def save_net(self, save_path, epi=0):
        os.makedirs(save_path, exist_ok=True)
        torch.save(dict(format_version=2, config=self.config, metadata=self.metadata,
                        state_dict=self.state_dict(), optimizer=self.optimizer.state_dict(), step=epi),
                   os.path.join(save_path, 'diffuser.pt'))

    @classmethod
    def from_checkpoint(cls, load_path, device='cpu'):
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get('format_version') != 2:
            raise ValueError("Legacy DD weights lack category/CPA/normalization metadata; retrain a v2 model")
        model = cls(**checkpoint['config'], metadata=checkpoint['metadata']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()
        return model
