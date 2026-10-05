"""Upgraded world-model architecture components (egg pipeline)."""

import copy
import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as torchd
from torch import nn

from . import networks, tools


class LatentDistBase(nn.Module):
    """Latent-distribution helpers shared with networks.RSSM (same math/API).

    Supports BOTH the categorical latent (``dyn_discrete > 0``: stoch x discrete
    one-hot with unimix) and the continuous diagonal-Gaussian latent
    (``dyn_discrete == 0``: mean/std Normal), selected by the subclass' own
    ``self._discrete`` flag exactly like networks.RSSM."""

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            return torchd.independent.Independent(
                tools.OneHotDist(state["logit"], unimix_ratio=self._unimix_ratio), 1
            )
        mean, std = state["mean"], state["std"]
        return tools.ContDist(
            torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
        )

    def get_dist_dense(self, state, dtype=None):
        if self._discrete:
            return torchd.independent.Independent(
                tools.OneHotDist(state["logit"], unimix_ratio=self._unimix_ratio), 0
            )
        mean, std = state["mean"], state["std"]
        return tools.ContDist(
            torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
        )

    def _suff_stats_layer(self, name, x):
        if name == "ims":
            x = self._imgs_stat_layer(x)
        elif name == "obs":
            x = self._obs_stat_layer(x)
        else:
            raise NotImplementedError(name)
        if self._discrete:
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        mean, std = torch.split(x, [self._stoch] * 2, -1)
        mean = {"none": lambda: mean, "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0)}[
            self._mean_act
        ]()
        std = {
            "softplus": lambda: F.softplus(std),
            "abs": lambda: torch.abs(std + 1),
            "sigmoid": lambda: torch.sigmoid(std),
            "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
        }[self._std_act]()
        std = std + self._min_std
        return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        unwrap = lambda d: d if self._discrete else d._dist

        rep_loss = value = kld(unwrap(dist(post)), unwrap(dist(sg(prior))))
        dyn_loss = kld(unwrap(dist(sg(post))), unwrap(dist(prior)))
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        return loss, value, dyn_loss, rep_loss

    def get_kl_divergence(self, post, prior):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist_dense(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        unwrap = lambda d: d if self._discrete else d._dist
        with torch.no_grad():
            value = kld(unwrap(dist(post)), unwrap(dist(sg(prior))))
        return value


def _rope_freqs(head_dim, max_len, device, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x, cos, sin):
    # x: (B, heads, T, head_dim); cos/sin: (T, head_dim/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim % 2 == 0, "RoPE needs an even head dim"
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x, attn_mask, rope):
        # x: (B, T, D); attn_mask: (B, 1, T, T) bool, True = may attend.
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (B, T, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        cos, sin = rope
        q = _apply_rope(q, cos[:T], sin[:T])
        k = _apply_rope(k, cos[:T], sin[:T])
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


def _modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class ConditionalBlock(nn.Module):
    """Pre-LN transformer block with AdaLN-zero conditioning (le-wm style).

    The per-step condition c (here: embedded action) produces shift/scale/gate
    for both the attention and MLP sublayers; the modulation projection is
    zero-initialized so each block starts as identity."""

    def __init__(self, dim, n_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = CausalSelfAttention(dim, n_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Linear(ff_mult * dim, dim),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c, attn_mask, rope):
        sa, ga, ma, sm, gm, mm = self.adaLN(c).chunk(6, dim=-1)
        x = x + ma * self.attn(_modulate(self.norm1(x), sa, ga), attn_mask, rope)
        x = x + mm * self.mlp(_modulate(self.norm2(x), sm, gm))
        return x


class TransformerRSSM(LatentDistBase):
    """STORM/TransDreamer-style sequence model with the DreamerV3 latent core."""

    def __init__(self, config, embed_size):
        super().__init__()
        tf = dict(config.transformer)
        self._stoch = config.dyn_stoch
        self._deter = config.dyn_deter
        self._hidden = config.dyn_hidden
        self._discrete = config.dyn_discrete
        self._mean_act = getattr(config, "dyn_mean_act", "none")
        self._std_act = getattr(config, "dyn_std_act", "sigmoid2")
        self._min_std = getattr(config, "dyn_min_std", 0.1)
        self._unimix_ratio = config.unimix_ratio
        self._initial = config.initial
        self._num_actions = config.num_actions
        self._embed = embed_size
        self._device = config.device
        self._history = int(tf.get("history", 8))
        assert 1 <= self._history <= 16, "transformer.history must be in [1, 16]"
        self._windowed = bool(tf.get("windowed", True))
        self._depth = int(tf.get("depth", 6))
        self._n_heads = int(tf.get("n_heads", 16))
        self._ff_mult = int(tf.get("ff_mult", 4))
        self._dropout = float(tf.get("dropout", 0.0))
        act = getattr(torch.nn, config.act)

        z_dim = self._stoch * self._discrete if self._discrete else self._stoch
        self._z_dim = z_dim
        # Categorical: logits (stoch*discrete). Continuous: mean+std (2*stoch).
        stat_out = z_dim if self._discrete else 2 * self._stoch
        self._z_in = nn.Sequential(
            nn.Linear(z_dim, self._deter, bias=False),
            nn.LayerNorm(self._deter, eps=1e-03),
            act(),
        )
        self._z_in.apply(tools.weight_init)
        self._a_emb = nn.Sequential(
            nn.Linear(self._num_actions, self._deter),
            nn.SiLU(),
            nn.Linear(self._deter, self._deter),
        )
        self._a_emb.apply(tools.weight_init)
        self.blocks = nn.ModuleList(
            [
                ConditionalBlock(self._deter, self._n_heads, self._ff_mult, self._dropout)
                for _ in range(self._depth)
            ]
        )
        self._final_norm = nn.LayerNorm(self._deter, eps=1e-03)

        img_out = [nn.Linear(self._deter, self._hidden, bias=False)]
        if config.norm:
            img_out.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out.append(act())
        self._img_out_layers = nn.Sequential(*img_out)
        self._img_out_layers.apply(tools.weight_init)

        obs_out = [nn.Linear(self._embed, self._hidden, bias=False)]
        if config.norm:
            obs_out.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out)
        self._obs_out_layers.apply(tools.weight_init)

        self._imgs_stat_layer = nn.Linear(self._hidden, stat_out)
        self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
        self._obs_stat_layer = nn.Linear(self._hidden, stat_out)
        self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )
        max_len = 4096
        head_dim = self._deter // self._n_heads
        cos, sin = _rope_freqs(head_dim, max_len, torch.device(self._device))
        self.register_buffer("_rope_cos", cos, persistent=False)
        self.register_buffer("_rope_sin", sin, persistent=False)

    # ------------------------------------------------------------------ #

    def initial(self, batch_size):
        B, K = batch_size, self._history
        z_dim = self._z_dim
        if self._discrete:
            stats = dict(
                logit=torch.zeros([B, self._stoch, self._discrete], device=self._device),
                stoch=torch.zeros([B, self._stoch, self._discrete], device=self._device),
            )
        else:
            stats = dict(
                mean=torch.zeros([B, self._stoch], device=self._device),
                std=torch.zeros([B, self._stoch], device=self._device),
                stoch=torch.zeros([B, self._stoch], device=self._device),
            )
        state = dict(
            **stats,
            deter=torch.zeros(B, self._deter, device=self._device),
            z_hist=torch.zeros(B, K, z_dim, device=self._device),
            a_hist=torch.zeros(B, K, self._num_actions, device=self._device),
            hist_mask=torch.zeros(B, K, device=self._device),
        )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        return self.get_dist(stats).mode()

    def _transformer(self, tok, cond, attn_mask):
        rope = (self._rope_cos, self._rope_sin)
        x = tok
        for blk in self.blocks:
            x = blk(x, cond, attn_mask, rope)
        return self._final_norm(x)

    def _window_attn_mask(self, hist_mask):
        """(N, K) validity -> (N, 1, K, K) bool: causal within the window with
        invalid (padding / earlier-segment) key slots removed; the diagonal is
        kept so a fully-padded query row stays finite. Same rule in windowed
        observe and img_step."""
        K = self._history
        dev = hist_mask.device
        i = torch.arange(K, device=dev)[:, None]
        j = torch.arange(K, device=dev)[None, :]
        causal = (j <= i)[None, None]
        valid = hist_mask[..., None, None, :] > 0.0
        eye = torch.eye(K, device=dev, dtype=torch.bool)[None, None]
        return (causal & valid) | eye

    def _window_deter(self, z_hist, a_hist, hist_mask):
        """h from exactly the last K = `history` (z, a) tokens run through the
        full depth-L stack (last position). Vectorized over any leading dims, so
        the identical computation serves training (windowed observe, lead=(B,T))
        and imagination (img_step, lead=(B,)) -- receptive field is exactly
        `history` regardless of depth."""
        K = self._history
        lead = z_hist.shape[:-2]
        z = z_hist.reshape(-1, K, z_hist.shape[-1])
        a = a_hist.reshape(-1, K, a_hist.shape[-1])
        m = hist_mask.reshape(-1, K)
        tok = self._z_in(z)
        cond = self._a_emb(a)
        deter = self._transformer(tok, cond, self._window_attn_mask(m))[:, -1]
        return deter.reshape(*lead, self._deter)

    # ------------------------------------------------------------------ #

    def observe(self, embed, action, is_first, state=None, sample=True):
        if state is not None:
            # Rare continuation path: fall back to the sequential scan.
            swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
            e, a, f = swap(embed), swap(action), swap(is_first)
            post, prior = tools.static_scan(
                lambda prev_state, prev_act, emb, first: self.obs_step(
                    prev_state[0], prev_act, emb, first, sample
                ),
                (a, e, f),
                (state, state),
            )
            post = {k: swap(v) for k, v in post.items()}
            prior = {k: swap(v) for k, v in prior.items()}
            return post, prior

        B, T = embed.shape[:2]
        is_first = is_first.float()

        # Posterior for every step in parallel (obs-only posterior).
        post_stats = self._suff_stats_layer("obs", self._obs_out_layers(embed))
        post_dist = self.get_dist(post_stats)
        stoch_post = post_dist.sample() if sample else post_dist.mode()

        # Token t is built from (z_{t-1}, a_t); resets use the learned initial.
        z_dim = self._z_dim
        z_flat = stoch_post.reshape(B, T, z_dim)
        init = self.initial(B)
        init_z = init["stoch"].reshape(B, 1, z_dim)
        z_prev = torch.cat([init_z, z_flat[:, :-1]], dim=1)
        first = is_first[:, :, None]
        z_prev = z_prev * (1.0 - first) + init_z * first
        act_in = action * (1.0 - first)
        seg = torch.cumsum(is_first, dim=1)

        # imagination can start from any step).
        z_hist, a_hist, hist_mask = self._history_buffers(z_prev, act_in, seg)

        if self._windowed:
            # Each transition recomputed from EXACTLY its last `history` tokens
            # (== img_step); depth-independent receptive field, train == imag.
            deter = self._window_deter(z_hist, a_hist, hist_mask)
        else:
            # Legacy banded, segment-causal mask: j <= i, i-j < history, same seg.
            tok = self._z_in(z_prev)
            cond = self._a_emb(act_in)
            i = torch.arange(T, device=embed.device)[None, :, None]
            j = torch.arange(T, device=embed.device)[None, None, :]
            allowed = (j <= i) & ((i - j) < self._history)
            allowed = allowed & (seg[:, :, None] == seg[:, None, :])
            attn_mask = allowed[:, None]
            deter = self._transformer(tok, cond, attn_mask)

        prior_stats = self._suff_stats_layer("ims", self._img_out_layers(deter))
        prior_dist = self.get_dist(prior_stats)
        stoch_prior = prior_dist.sample() if sample else prior_dist.mode()

        post = {
            "stoch": stoch_post,
            "deter": deter,
            **post_stats,
            "z_hist": z_hist,
            "a_hist": a_hist,
            "hist_mask": hist_mask,
        }
        prior = {"stoch": stoch_prior, "deter": deter, **prior_stats}
        return post, prior

    def _history_buffers(self, z_prev, act_in, seg):
        """Sliding windows of the token inputs (z_{s-1}, a_s) for s in
        (t-K+1 .. t], oldest first; invalid slots (before t=0 or in an earlier
        segment) are masked out and zeroed."""
        B, T = z_prev.shape[:2]
        K = self._history
        pad = lambda x: torch.cat(
            [torch.zeros(B, K - 1, *x.shape[2:], device=x.device, dtype=x.dtype), x],
            dim=1,
        )
        # unfold -> (B, T, feat, K) -> (B, T, K, feat); slot k is step t-K+1+k.
        z_hist = pad(z_prev).unfold(1, K, 1).permute(0, 1, 3, 2).contiguous()
        a_hist = pad(act_in).unfold(1, K, 1).permute(0, 1, 3, 2).contiguous()
        seg_pad = torch.cat(
            [-torch.ones(B, K - 1, device=seg.device), seg], dim=1
        )
        mask = (seg_pad.unfold(1, K, 1) == seg[:, :, None]).float()
        z_hist = z_hist * mask[..., None]
        a_hist = a_hist * mask[..., None]
        return z_hist, a_hist, mask

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        if prev_state is None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros(
                (len(is_first), self._num_actions), device=self._device
            )
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action = prev_action * (1.0 - is_first)
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action, sample)
        stats = self._suff_stats_layer("obs", self._obs_out_layers(embed))
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {
            "stoch": stoch,
            "deter": prior["deter"],
            **stats,
            "z_hist": prior["z_hist"],
            "a_hist": prior["a_hist"],
            "hist_mask": prior["hist_mask"],
        }
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        B = prev_action.shape[0]
        z_dim = self._z_dim
        prev_z = prev_state["stoch"].reshape(B, z_dim)

        z_hist = torch.cat([prev_state["z_hist"][:, 1:], prev_z[:, None]], dim=1)
        a_hist = torch.cat([prev_state["a_hist"][:, 1:], prev_action[:, None]], dim=1)
        hist_mask = torch.cat(
            [
                prev_state["hist_mask"][:, 1:],
                torch.ones(B, 1, device=prev_action.device),
            ],
            dim=1,
        )

        deter = self._window_deter(z_hist, a_hist, hist_mask)
        stats = self._suff_stats_layer("ims", self._img_out_layers(deter))
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        return {
            "stoch": stoch,
            "deter": deter,
            **stats,
            "z_hist": z_hist,
            "a_hist": a_hist,
            "hist_mask": hist_mask,
        }

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoBackbone(nn.Module):
    """Pretrained DINO ViT wrapper: (N, 3, H, W) in [0, 1] -> patch tokens
    (N, P, C). CLS/register tokens are dropped; ImageNet normalization is
    applied inside."""

    def __init__(
        self,
        model="dinov3_vits16",
        source="hub",
        weights_path="",
        repo_dir="",
        resize_to=None,
        grad_ckpt=False,
    ):
        super().__init__()
        self.model_name = model
        self.source = source
        self.resize_to = tuple(resize_to) if resize_to else None
        self.grad_ckpt = grad_ckpt
        self._hf = False

        if source == "hf":
            from transformers import AutoModel

            self.net = AutoModel.from_pretrained(model)
            self._hf = True
            self.embed_dim = self.net.config.hidden_size
            self.patch_size = self.net.config.patch_size
            self._skip_tokens = 1 + getattr(self.net.config, "num_register_tokens", 0)
        else:
            if source == "local":
                assert repo_dir, "dino.repo_dir required for source: local"
                kwargs = {"source": "local"}
                if weights_path:
                    kwargs["weights"] = weights_path
                self.net = torch.hub.load(repo_dir, model, **kwargs)
            else:
                repo = (
                    "facebookresearch/dinov3"
                    if "dinov3" in model
                    else "facebookresearch/dinov2"
                )
                # without a weights file the backbone is expected to come from a checkpoint
                kwargs = {"weights": weights_path} if weights_path else {"pretrained": False}
                self.net = torch.hub.load(repo, model, **kwargs)
            self.embed_dim = getattr(self.net, "embed_dim", None) or self.net.num_features
            ps = self.net.patch_embed.patch_size
            self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps

        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def grid_hw(self, image_hw):
        h, w = self.resize_to if self.resize_to else image_hw
        assert h % self.patch_size == 0 and w % self.patch_size == 0, (
            f"image {h}x{w} not divisible by patch size {self.patch_size}; "
            "set dino.resize_to"
        )
        return h // self.patch_size, w // self.patch_size

    def n_tokens(self, image_hw):
        gh, gw = self.grid_hw(image_hw)
        return gh * gw

    def _tokens_cls(self, x):
        """(patch tokens (N, P, C), CLS (N, C)) from ONE backbone pass."""
        if self._hf:
            out = self.net(pixel_values=x).last_hidden_state
            return out[:, self._skip_tokens :], out[:, 0]
        out = self.net.forward_features(x)
        return out["x_norm_patchtokens"], out["x_norm_clstoken"]

    def _preprocess(self, x):
        if self.resize_to is not None and tuple(x.shape[-2:]) != self.resize_to:
            x = F.interpolate(x, size=self.resize_to, mode="bilinear", antialias=True)
        return (x - self._mean) / self._std

    def forward(self, x):
        """(patch tokens, CLS) tuple; the CLS comes for free from the same
        forward_features pass (previously a second full ViT pass)."""
        x = self._preprocess(x)
        needs_grad = torch.is_grad_enabled() and any(
            p.requires_grad for p in self.parameters()
        )
        if self.grad_ckpt and needs_grad:
            return torch.utils.checkpoint.checkpoint(
                self._tokens_cls, x, use_reentrant=False
            )
        return self._tokens_cls(x)


class DinoViewHead(nn.Module):
    """Per-camera trainable projector + learned-query attention pooling.

    n_queries > 1 pools the view into several tokens instead of one, so more
    spatial detail survives into the embedding (and hence the latent)."""

    def __init__(self, token_dim, pool_dim=384, heads=4, n_queries=1):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.proj = nn.Linear(token_dim, pool_dim)
        self.query = nn.Parameter(torch.zeros(1, n_queries, pool_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(pool_dim, heads, batch_first=True)
        self.out_norm = nn.LayerNorm(pool_dim)

    def forward(self, tokens):
        # tokens: (N, P, C) -> (N, n_queries * pool_dim)
        x = self.proj(self.norm(tokens))
        q = self.query.expand(x.shape[0], -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)
        return self.out_norm(pooled).flatten(1)


class DinoMultiEncoder(nn.Module):
    """Drop-in for networks.MultiEncoder with a shared pretrained DINO backbone
    per camera view. forward(obs) -> (B, T, outdim); patch tokens of the last
    forward are stashed in .last_tokens as reconstruction targets."""

    def __init__(self, shapes, config):
        super().__init__()
        enc_cfg = dict(config.encoder)
        dcfg = dict(config.dino)
        recon_cfg = dict(getattr(config, "dino_recon", {"enabled": False}))

        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) == 3 and re.match(enc_cfg["cnn_keys"], k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(enc_cfg["mlp_keys"], k)
        }
        print("DinoMultiEncoder CNN shapes:", self.cnn_shapes)
        print("DinoMultiEncoder MLP shapes:", self.mlp_shapes)
        self._views = list(self.cnn_shapes.keys())
        assert self._views, "DinoMultiEncoder needs at least one camera key"

        self.backbone = DinoBackbone(
            model=dcfg.get("model", "dinov3_vits16"),
            source=dcfg.get("source", "hub"),
            weights_path=dcfg.get("weights_path", ""),
            repo_dir=dcfg.get("repo_dir", ""),
            resize_to=dcfg.get("resize_to", None),
            grad_ckpt=dcfg.get("grad_ckpt", False),
        )
        self._train_encoder = bool(dcfg.get("train_encoder", False))
        self._unfreeze_after = int(dcfg.get("unfreeze_after", 0))
        self.register_buffer("_enc_steps", torch.zeros((), dtype=torch.int64))
        self._chunk = int(dcfg.get("chunk", 256))
        unfreeze_blocks = int(dcfg.get("unfreeze_blocks", 0))
        if not self._train_encoder:
            self.backbone.requires_grad_(False)
        elif unfreeze_blocks > 0:
            self.backbone.requires_grad_(False)
            blocks = self.backbone.net.blocks
            for blk in blocks[-unfreeze_blocks:]:
                blk.requires_grad_(True)
            if hasattr(self.backbone.net, "norm"):
                self.backbone.net.norm.requires_grad_(True)

        image_hw = tuple(self.cnn_shapes[self._views[0]][:2])
        self.grid_hw = self.backbone.grid_hw(image_hw)
        self.n_tokens = self.grid_hw[0] * self.grid_hw[1]
        self.token_dim = self.backbone.embed_dim

        pool_dim = int(dcfg.get("pool_dim", 384))
        pool_heads = int(dcfg.get("pool_heads", 4))
        pool_queries = int(dcfg.get("pool_queries", 1))
        self._fusion = dcfg.get("fusion", "none")
        if self._fusion == "xattn":
            fusion_depth = int(dcfg.get("fusion_depth", 2))
            fusion_heads = int(dcfg.get("fusion_heads", 8))
            self.view_proj = nn.ModuleDict(
                {
                    v: nn.Sequential(
                        nn.LayerNorm(self.token_dim),
                        nn.Linear(self.token_dim, pool_dim),
                    )
                    for v in self._views
                }
            )
            self.view_emb = nn.Parameter(
                torch.zeros(len(self._views), 1, pool_dim)
            )
            nn.init.trunc_normal_(self.view_emb, std=0.02)
            layer = nn.TransformerEncoderLayer(
                pool_dim,
                fusion_heads,
                dim_feedforward=4 * pool_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.fusion_blocks = nn.TransformerEncoder(
                layer, fusion_depth, enable_nested_tensor=False
            )
            self.pool_query = nn.Parameter(
                torch.zeros(1, len(self._views) * pool_queries, pool_dim)
            )
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            self.pool_attn = nn.MultiheadAttention(
                pool_dim, pool_heads, batch_first=True
            )
            self.pool_norm = nn.LayerNorm(pool_dim)
        else:
            assert self._fusion in ("none", "concat"), self._fusion
            self.view_heads = nn.ModuleDict(
                {
                    v: DinoViewHead(self.token_dim, pool_dim, pool_heads, pool_queries)
                    for v in self._views
                }
            )

        self.outdim = len(self._views) * pool_queries * pool_dim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = networks.MLP(
                input_size,
                None,
                enc_cfg["mlp_layers"],
                enc_cfg["mlp_units"],
                enc_cfg["act"],
                enc_cfg["norm"],
                symlog_inputs=enc_cfg["symlog_inputs"],
                name="Encoder",
            )
            self.outdim += enc_cfg["mlp_units"]

        self._store_tokens = bool(recon_cfg.get("enabled", False))
        self._recon_target = recon_cfg.get("target", "frozen")
        assert self._recon_target in ("frozen", "ema", "detach"), self._recon_target
        self._recon_kind = recon_cfg.get("target_kind", "cls_mean")
        assert self._recon_kind in ("cls", "mean", "cls_mean"), self._recon_kind
        self._recon_parts = (
            ["cls", "mean"] if self._recon_kind == "cls_mean" else [self._recon_kind]
        )
        self.recon_out_dim = self.token_dim * len(self._recon_parts)
        self._ema_tau = float(recon_cfg.get("ema_tau", 0.999))
        if (
            self._store_tokens
            and self._train_encoder
            and self._recon_target in ("frozen", "ema")
        ):
            self._target_backbone = copy.deepcopy(self.backbone).requires_grad_(False)
        else:
            self._target_backbone = None
        self.last_tokens = {}

    def _update_ema(self):
        with torch.no_grad():
            for p_ema, p in zip(
                self._target_backbone.parameters(), self.backbone.parameters()
            ):
                p_ema.lerp_(p, 1.0 - self._ema_tau)

    def _backbone_forward(self, x, module, trainable):
        """Backbone forward -> (patch tokens, CLS), ALWAYS chunked over frames
        (self._chunk at a time) so peak activation memory is independent of
        batch_size * batch_length * n_cameras.
        """
        chunks = range(0, len(x), self._chunk)
        if trainable and torch.is_grad_enabled():
            outs = [module(x[i : i + self._chunk]) for i in chunks]
        else:
            with torch.no_grad():
                outs = [module(x[i : i + self._chunk]) for i in chunks]
        return tuple(torch.cat(o, 0) for o in zip(*outs))

    def _global_target(self, patch_tokens, cls):
        """Global recon target (N, token_dim*len(parts)): CLS token and/or
        mean-pooled patch tokens (both from the single target forward)."""
        parts = [
            patch_tokens.mean(dim=1) if kind == "mean" else cls
            for kind in self._recon_parts
        ]
        return torch.cat(parts, -1)

    def forward(self, obs):
        if (
            self._target_backbone is not None
            and self._recon_target == "ema"
            and self.training
        ):
            self._update_ema()
        trainable = (
            self._train_encoder and int(self._enc_steps) >= self._unfreeze_after
        )
        if self.training and torch.is_grad_enabled():
            self._enc_steps += 1
        outputs = []
        self.last_tokens = {}
        view_tokens = {}
        for v in self._views:
            img = obs[v]
            B, T = img.shape[:2]
            x = img.reshape((-1,) + tuple(img.shape[2:])).permute(0, 3, 1, 2)
            tokens, cls = self._backbone_forward(x, self.backbone, trainable)
            if self._store_tokens:
                if self._target_backbone is not None:
                    patch_tgt, cls_tgt = self._backbone_forward(
                        x, self._target_backbone, False
                    )
                else:
                    patch_tgt, cls_tgt = tokens.detach(), cls.detach()
                glob = self._global_target(patch_tgt, cls_tgt)
                self.last_tokens[v] = glob.detach().reshape(B, T, self.recon_out_dim)
            view_tokens[v] = tokens
        if self._fusion == "xattn":
            fused = torch.cat(
                [
                    self.view_proj[v](view_tokens[v]) + self.view_emb[i]
                    for i, v in enumerate(self._views)
                ],
                dim=1,
            )
            fused = self.fusion_blocks(fused)
            q = self.pool_query.expand(fused.shape[0], -1, -1)
            pooled, _ = self.pool_attn(q, fused, fused, need_weights=False)
            outputs.append(self.pool_norm(pooled).flatten(1).reshape(B, T, -1))
        else:
            for v in self._views:
                outputs.append(self.view_heads[v](view_tokens[v]).reshape(B, T, -1))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        return torch.cat(outputs, -1)


class DinoReconHead(nn.Module):
    """Reconstruct a per-view GLOBAL (frozen/EMA) DINO descriptor from the model
    feature -- a light semantic regularizer matched to the encoder's aggressive
    token pool. The target is the CLS token and/or mean-pooled patch tokens
    (see DinoMultiEncoder.recon_out_dim); a small per-view MLP maps
    feat (B, T, F) -> {view: (B, T, out_dim)}. O(F*hidden) params, not
    O(F*P*token_dim) like the old dense patch-token head."""

    def __init__(self, feat_size, views, out_dim, hidden=256, layers=2):
        super().__init__()
        self.heads = nn.ModuleDict()
        for v in views:
            mlp = [nn.Linear(feat_size, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            for _ in range(max(layers - 2, 0)):
                mlp += [nn.Linear(hidden, hidden), nn.SiLU()]
            mlp += [nn.Linear(hidden, out_dim)]
            self.heads[v] = nn.Sequential(*mlp)

    def forward(self, feat):
        return {v: head(feat) for v, head in self.heads.items()}


class DinoPixelDecoder(nn.Module):
    """Pixels from (predicted) DINO patch tokens: reshape the token sequence
    back to its (gh, gw) spatial grid and upsample with transposed convs.
    """

    def __init__(self, views, token_dim, grid_hw, out_hw, depth=32):
        super().__init__()
        self._grid_hw = tuple(grid_hw)
        self._out_hw = tuple(out_hw)
        gh, gw = self._grid_hw
        H, W = self._out_hw
        n = int(np.log2(H // gh))
        assert H == gh * 2**n and W == gw * 2**n, (
            f"out {tuple(out_hw)} not reachable from token grid {tuple(grid_hw)} "
            "with stride-2 stages"
        )

        def build():
            ch = depth * 2 ** (n - 1)
            layers = [nn.Conv2d(token_dim, ch, 1)]
            in_dim = ch
            for i in range(n):
                last = i == n - 1
                out_dim = 3 if last else in_dim // 2
                layers.append(
                    nn.ConvTranspose2d(in_dim, out_dim, 4, 2, padding=1, bias=last)
                )
                if not last:
                    layers.append(networks.ImgChLayerNorm(out_dim))
                    layers.append(nn.SiLU())
                in_dim = out_dim
            net = nn.Sequential(*layers)
            [m.apply(tools.weight_init) for m in net[:-1]]
            net[-1].apply(tools.uniform_weight_init(1.0))
            return net

        self.dec = nn.ModuleDict({v: build() for v in views})

    def forward(self, tokens):
        # tokens: {view: (B, T, P, C)} -> {view: (B, T, H, W, 3)} means.
        out = {}
        gh, gw = self._grid_hw
        for v, tok in tokens.items():
            B, T, P, C = tok.shape
            assert P == gh * gw, (P, self._grid_hw)
            x = tok.reshape(B * T, gh, gw, C).permute(0, 3, 1, 2)
            x = self.dec[v](x)
            mean = x.reshape(B, T, 3, *self._out_hw).permute(0, 1, 3, 4, 2)
            out[v] = mean + 0.5
        return out


class PatchDecoder(nn.Module):
    """MAE-style transformer pixel decoder: a learned (H/p, W/p) grid of patch
    query tokens runs through ConditionalBlocks conditioned on the latent feat
    via AdaLN-zero (same block as TransformerRSSM), and each token predicts its
    own p x p pixel patch with a linear head. Same (features) -> (B, T, H, W, C)
    interface as ConvDecoderRect (selected by decoder.arch: 'patch').
    """

    def __init__(
        self,
        feat_size,
        shape=(3, 192, 256),
        patch_size=16,
        dim=256,
        depth=4,
        n_heads=8,
        ff_mult=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        super().__init__()
        C, H, W = shape
        assert H % patch_size == 0 and W % patch_size == 0, (shape, patch_size)
        assert (dim // n_heads) % 2 == 0, "RoPE needs an even head dim"
        self._shape = shape
        self._patch = patch_size
        self._grid = (H // patch_size, W // patch_size)
        self._cnn_sigmoid = cnn_sigmoid
        n_tok = self._grid[0] * self._grid[1]

        # Learned patch queries double as the (2D) position embedding.
        self.query = nn.Parameter(torch.zeros(1, n_tok, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.cond = nn.Sequential(
            nn.Linear(feat_size, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList(
            [ConditionalBlock(dim, n_heads, ff_mult) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, patch_size * patch_size * C)
        self.head.apply(tools.uniform_weight_init(outscale))
        # Raster-order relative positions for CausalSelfAttention's RoPE; the
        # per-token queries carry the true 2D identity. Buffers move with .to().
        cos, sin = _rope_freqs(dim // n_heads, n_tok, torch.device("cpu"))
        self.register_buffer("_rope_cos", cos, persistent=False)
        self.register_buffer("_rope_sin", sin, persistent=False)

    def forward(self, features, dtype=None):
        lead = features.shape[:-1]
        x = features.reshape(-1, features.shape[-1])
        c = self.cond(x)[:, None]
        tok = self.query.expand(x.shape[0], -1, -1)
        rope = (self._rope_cos, self._rope_sin)
        for blk in self.blocks:
            tok = blk(tok, c, None, rope)
        out = self.head(self.norm(tok))
        gh, gw = self._grid
        p, C = self._patch, self._shape[0]
        out = out.reshape(-1, gh, gw, p, p, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(-1, gh * p, gw * p, C)
        mean = out.reshape(*lead, gh * p, gw * p, C)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean = mean + 0.5
        return mean


class ConvDecoderRect(nn.Module):
    """networks.ConvDecoder generalized to a rectangular minres (h0, w0)."""

    def __init__(
        self,
        feat_size,
        shape=(3, 192, 256),
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=(3, 4),
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        super().__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        h0, w0 = minres
        layer_num = int(np.log2(shape[1] // h0))
        assert shape[1] == h0 * 2**layer_num and shape[2] == w0 * 2**layer_num, (
            f"shape {shape} not reachable from minres {minres} with stride-2 stages"
        )
        self._minres = (h0, w0)
        out_ch = h0 * w0 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (h0 * w0)
        out_dim = in_dim // 2

        layers = []
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False
            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(networks.ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        h0, w0 = self._minres
        x = self._linear_layer(features)
        x = x.reshape([-1, h0, w0, self._embed_size // (h0 * w0)])
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        mean = x.reshape(features.shape[:-1] + self._shape)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean += 0.5
        return mean


class MultiDecoderV2(nn.Module):
    """networks.MultiDecoder that picks the pixel decoder by `arch`:
    'patch' -> PatchDecoder (transformer); 'conv' (default) -> ConvDecoderRect
    when minres is a (h0, w0) pair, else networks.ConvDecoder."""

    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
        arch="conv",
        patch=None,
    ):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            if arch == "patch":
                pcfg = dict(patch or {})
                self._cnn = PatchDecoder(
                    feat_size,
                    shape,
                    patch_size=int(pcfg.get("size", 16)),
                    dim=int(pcfg.get("dim", 256)),
                    depth=int(pcfg.get("depth", 4)),
                    n_heads=int(pcfg.get("heads", 8)),
                    ff_mult=int(pcfg.get("ff_mult", 4)),
                    outscale=outscale,
                    cnn_sigmoid=cnn_sigmoid,
                )
            elif isinstance(minres, (list, tuple)):
                self._cnn = ConvDecoderRect(
                    feat_size,
                    shape,
                    cnn_depth,
                    act,
                    norm,
                    kernel_size,
                    tuple(minres),
                    outscale=outscale,
                    cnn_sigmoid=cnn_sigmoid,
                )
            else:
                self._cnn = networks.ConvDecoder(
                    feat_size,
                    shape,
                    cnn_depth,
                    act,
                    norm,
                    kernel_size,
                    minres,
                    outscale=outscale,
                    cnn_sigmoid=cnn_sigmoid,
                )
        if self.mlp_shapes:
            self._mlp = networks.MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            outputs = self._cnn(features)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)
