"""Adversarial RL for the robust latent safety filter (LUCID).

The safety policy maximizes the safety value V(z). The latent disturbance adds a
residual to the latent dynamics prior, kept inside the uncertainty set by a KL
penalty (radius `kl_radius`) and an in-distribution constraint (OOD score
<= `ood_threshold`).
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions as torchd

from .dreamer import tools
from .nets import ActorProb, Critic, Net, SplitCriticNet


def _cat_dist(logit, unimix=0.01, event_dims=1):
	return torchd.independent.Independent(tools.OneHotDist(logit, unimix_ratio=unimix), event_dims)


def _normal_dist(mean, std, event_dims=1):
	return torchd.independent.Independent(torchd.normal.Normal(mean, std), event_dims)


def _zero_last_linear(net):
	last = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
	nn.init.zeros_(last.weight)
	nn.init.zeros_(last.bias)


class CategoricalDisturbance(nn.Module):
	"""Residual on the prior logits, conditioned on the deterministic state."""

	def __init__(self, cfg, hidden, max_delta):
		super().__init__()
		self.S, self.D, self.deter = cfg.dyn_stoch, cfg.dyn_discrete, cfg.dyn_deter
		self.net = Net(self.deter, self.S * self.D, hidden, norm_layer=nn.LayerNorm)
		_zero_last_linear(self.net)
		self.max_delta = max_delta

	def forward(self, feat, prior):
		raw = self.net(feat[:, -self.deter:].detach())[0].reshape(-1, self.S, self.D)
		return self.max_delta * torch.tanh(raw), raw


class GaussianDisturbance(nn.Module):
	"""Residual on the prior (mean, std), conditioned on [deter, mean, std]."""

	def __init__(self, cfg, hidden, max_delta):
		super().__init__()
		self.S, self.deter = cfg.dyn_stoch, cfg.dyn_deter
		self.net = Net(self.deter + 2 * self.S, 2 * self.S, hidden, norm_layer=nn.LayerNorm)
		_zero_last_linear(self.net)
		self.max_delta = max_delta

	def forward(self, feat, prior):
		x = torch.cat([feat[:, -self.deter:], prior], -1)
		raw = self.net(x.detach())[0].reshape(-1, 2 * self.S)
		d_mean = self.max_delta * torch.tanh(raw[:, :self.S])
		d_std = 0.5 * self.max_delta * torch.tanh(raw[:, self.S:])
		return torch.cat([d_mean, d_std], -1), raw


def aps_mask(probs, mass):
	"""Smallest set of classes whose cumulative probability reaches `mass`."""
	vals, idx = probs.sort(dim=-1, descending=True)
	keep = (vals.cumsum(-1) < mass).sum(-1, keepdim=True) + 1
	ar = torch.arange(probs.shape[-1], device=probs.device).expand_as(vals)
	mask = torch.zeros_like(probs, dtype=torch.bool)
	mask.scatter_(-1, idx, ar < keep)
	return mask


class GammaSchedule:
	"""gamma_k = goal - (goal - init) * decay ** (k // period)."""

	def __init__(self, init=0.85, goal=0.9999, period=1000, decay=0.95):
		self.init, self.goal, self.period, self.decay = init, goal, period, decay
		self.k = 0

	def step(self):
		self.k += 1
		value = self.goal - (self.goal - self.init) * self.decay ** (self.k // self.period)
		return min(value, 1.0)


class SAC(nn.Module):
	def __init__(self, cfg, dim_state, dim_action):
		super().__init__()
		self.cfg = cfg
		self.device = cfg.device
		self.discrete = cfg.dyn_discrete > 0
		self.S, self.D, self.deter = cfg.dyn_stoch, cfg.dyn_discrete, cfg.dyn_deter
		self.dim_state, self.dim_action = dim_state, dim_action

		self.gamma = cfg.gamma_reachability
		self.gamma_schedule = GammaSchedule() if cfg.gamma_anneal else None
		self.tau = cfg.tau
		self.max_action = float(cfg.action_max_scale)
		self.eps = np.finfo(np.float32).eps.item()
		self.updates = 0

		self.actor = ActorProb(Net(dim_state, 0, cfg.control_net, norm_layer=nn.LayerNorm), dim_action).to(self.device)
		self.critic1, self.critic2 = self._critic(), self._critic()
		self.critic1_target, self.critic2_target = self._critic(), self._critic()
		self.critic1_target.load_state_dict(self.critic1.state_dict())
		self.critic2_target.load_state_dict(self.critic2.state_dict())

		max_delta = cfg.disturbance_max_delta or (15.0 if self.discrete else 1.0)
		disturbance_cls = CategoricalDisturbance if self.discrete else GaussianDisturbance
		self.disturbance = disturbance_cls(cfg, cfg.control_net, max_delta).to(self.device)
		self.disturbance_sign = -1.0 if cfg.disturbance_mode == "optimistic" else 1.0

		self.target_entropy = -dim_action
		self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
		self.alpha = 1.0
		self.target_entropy_dist = -self.S
		self.log_alpha_dist_ent = torch.zeros(1, requires_grad=True, device=self.device)
		self.alpha_dist_ent = 1.0
		self.log_alpha_kl = torch.zeros(1, requires_grad=True, device=self.device)

		self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
		self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr)
		self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr)
		self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.actor_lr)
		self.disturbance_opt = optim.Adam(self.disturbance.parameters(), lr=cfg.disturbance_lr)
		self.alpha_dist_ent_opt = optim.Adam([self.log_alpha_dist_ent], lr=cfg.disturbance_lr)
		self.alpha_kl_opt = optim.Adam([self.log_alpha_kl], lr=cfg.disturbance_lr)

		self.density = None
		self.ood_threshold = None
		self.value_reg = None

	def _critic(self):
		cfg = self.cfg
		if cfg.critic_action_embed:
			trunk = SplitCriticNet(
				self.dim_state, self.dim_action,
				cfg.critic_state_embed_dim, cfg.critic_state_embed_layers,
				cfg.critic_action_embed_dim, cfg.critic_action_embed_layers,
				cfg.critic_trunk_dim, cfg.critic_trunk_layers,
			)
		else:
			trunk = Net(self.dim_state, self.dim_action, cfg.critic_net, concat=True)
		return Critic(trunk).to(self.device)

	def set_density(self, density, threshold):
		"""OOD score function for the in-distribution constraint."""
		self.density = density
		self.ood_threshold = threshold

	@property
	def use_density(self):
		return self.cfg.use_ood_constraint and self.density is not None

	# ---------------------------------------------------------------- policy

	def select_action(self, state, eval_mode=False):
		(mu, sigma), _ = self.actor(state)
		dist = torchd.Independent(torchd.Normal(mu, sigma), 1)
		pre = mu if eval_mode else dist.rsample()
		log_prob = dist.log_prob(pre).unsqueeze(-1)
		squashed = torch.tanh(pre)
		log_prob = log_prob - torch.log(self.max_action * (1 - squashed.pow(2)) + self.eps).sum(-1, keepdim=True)
		return self.max_action * squashed, log_prob

	@torch.no_grad()
	def value(self, state):
		"""V(s) = min_i Q_i(s, pi(s)) with the deterministic action."""
		a, _ = self.select_action(state, eval_mode=True)
		return torch.min(self.critic1(state, a), self.critic2(state, a)).squeeze(-1)

	# ------------------------------------------------------------- disturbance

	def disturbance_step(self, feat, logit, eval_mode=False):
		"""Perturbed prior parameters, perturbed latent feature, and log-prob."""
		delta, raw = self.disturbance(feat, logit)
		deter = feat[:, -self.deter:]
		if self.discrete:
			logit_p = logit.reshape(-1, self.S, self.D) + delta
			sample_logit = logit_p
			if self.cfg.aps_bound is not None:
				with torch.no_grad():
					mask = aps_mask(F.softmax(logit.reshape(-1, self.S, self.D), -1), self.cfg.aps_bound)
				sample_logit = logit_p.masked_fill(~mask, -100.0)
			dist = _cat_dist(sample_logit)
			stoch = dist.base_dist.mode() if eval_mode else dist.sample()
			log_prob = dist.log_prob(stoch)
			feat_p = torch.cat([stoch.reshape(-1, self.S * self.D), deter], -1)
			return logit_p.reshape(-1, self.S * self.D), feat_p, log_prob.unsqueeze(-1), raw
		mean = logit[:, :self.S] + delta[:, :self.S]
		std = torch.clamp(logit[:, self.S:] + delta[:, self.S:], self.cfg.disturbance_std_min, self.cfg.disturbance_std_max)
		dist = _normal_dist(mean, std)
		stoch = mean if eval_mode else dist.rsample()
		log_prob = dist.log_prob(stoch)
		return torch.cat([mean, std], -1), torch.cat([stoch, deter], -1), log_prob.unsqueeze(-1), raw

	def project(self, feat_p, feat):
		"""Fall back to the nominal latent wherever the perturbed one is OOD."""
		if not self.use_density:
			return feat_p
		ood = self.density(feat_p) > self.ood_threshold
		feat_p = feat_p.clone()
		feat_p[ood] = feat[ood]
		return feat_p

	@torch.no_grad()
	def perturb(self, feat, logit, eval_mode=False):
		_, feat_p, _, _ = self.disturbance_step(feat, logit, eval_mode)
		return self.project(feat_p, feat)

	def _kl(self, logit, logit_p):
		"""Per-sample KL(perturbed || prior), averaged over latent dimensions."""
		if self.discrete:
			post = _cat_dist(logit_p.reshape(-1, self.S, self.D), event_dims=0)
			prior = _cat_dist(logit.reshape(-1, self.S, self.D).detach(), event_dims=0)
		else:
			post = _normal_dist(logit_p[:, :self.S], logit_p[:, self.S:], event_dims=0)
			prior = _normal_dist(logit[:, :self.S].detach(), logit[:, self.S:].detach(), event_dims=0)
		return torchd.kl.kl_divergence(post, prior).mean(-1)

	# ---------------------------------------------------------------- update

	def _target_value(self, feat, logit):
		if self.cfg.use_disturbance and self.updates >= self.cfg.disturbance_warmup:
			_, feat_p, _, _ = self.disturbance_step(feat, logit)
			feat = self.project(feat_p, feat)
		a, log_prob = self.select_action(feat)
		q = torch.min(self.critic1_target(feat, a), self.critic2_target(feat, a))
		if self.cfg.entropy_in_q_target:
			q = q - self.alpha * log_prob
		return q.squeeze(-1)

	def update(self, batch):
		cfg = self.cfg
		s, logit, a, r, done, s_, logit_ = (batch[k] for k in ("s", "logit", "a", "r", "done", "s_", "logit_"))
		live = ~done
		info = {}

		# V(x) = (1 - g) l(x) + g min(l(x), V(x'))
		with torch.no_grad():
			v_next = self._target_value(s_[live], logit_[live])
			y = r.clone()
			y[live] = self.gamma * torch.min(r[live], v_next) + (1 - self.gamma) * r[live]
			y = y.unsqueeze(-1)

		if self.updates % cfg.critic_update_freq == 0:
			critic_loss = F.mse_loss(self.critic1(s, a), y) + F.mse_loss(self.critic2(s, a), y)
			if self.value_reg is not None:
				reg, reg_info = self.value_reg(self)
				critic_loss = critic_loss + reg
				info.update(reg_info)
			self.critic1_opt.zero_grad()
			self.critic2_opt.zero_grad()
			critic_loss.backward()
			nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
			nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
			self.critic1_opt.step()
			self.critic2_opt.step()
			info["critic_loss"] = critic_loss.item()

		if self.updates % cfg.actor_update_freq == 0:
			a_new, log_prob = self.select_action(s)
			q = torch.min(self.critic1(s, a_new), self.critic2(s, a_new))
			actor_loss = (self.alpha * log_prob - q).mean()
			self.actor_opt.zero_grad()
			actor_loss.backward()
			nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
			self.actor_opt.step()
			alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
			self.alpha_opt.zero_grad()
			alpha_loss.backward()
			self.alpha_opt.step()
			self.alpha = self.log_alpha.exp().item()
			info["actor_loss"] = actor_loss.item()
			info["alpha"] = self.alpha

		if cfg.use_disturbance and self.updates % cfg.disturbance_update_freq == 0:
			info.update(self._update_disturbance(s_[live], logit_[live]))

		for target, source in ((self.critic1_target, self.critic1), (self.critic2_target, self.critic2)):
			for tp, p in zip(target.parameters(), source.parameters()):
				tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
		self.updates += 1
		if self.gamma_schedule is not None:
			self.gamma = self.gamma_schedule.step()
		info["gamma"] = self.gamma
		return info

	def _update_disturbance(self, feat, logit):
		cfg = self.cfg
		logit_p, feat_p, log_prob, raw = self.disturbance_step(feat, logit)
		kl = self._kl(logit, logit_p)
		with torch.no_grad():
			alpha_kl = self.log_alpha_kl.exp()
			coeff = torch.where(kl > cfg.kl_radius, cfg.kl_overbound_mult * alpha_kl, alpha_kl)
			a, _ = self.select_action(feat_p)
		q = torch.min(self.critic1(feat_p, a), self.critic2(feat_p, a))
		kl_term = (coeff * (kl - cfg.kl_radius)).unsqueeze(-1)
		violation = 0.0
		if self.use_density:
			# off-manifold samples get only the density penalty (no value or entropy term)
			excess = torch.relu(self.density(feat_p).unsqueeze(-1) - self.ood_threshold)
			off = excess > 0
			q, log_prob = q.masked_fill(off, 0.0), log_prob.masked_fill(off, 0.0)
			kl_term = kl_term + cfg.ood_penalty_coeff * excess
			violation = off.float().mean().item()
		loss = self.disturbance_sign * q + self.alpha_dist_ent * log_prob + kl_term
		t = cfg.disturbance_reg_threshold
		loss = loss.mean() + 0.1 * (torch.relu(raw.abs() - t) ** 2).mean() + 1e-3 * (raw ** 2).mean()

		self.disturbance_opt.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(self.disturbance.parameters(), 1.0)
		self.disturbance_opt.step()

		kl_dual = -(self.log_alpha_kl.exp() * (kl.detach() - cfg.kl_radius)).mean()
		self.alpha_kl_opt.zero_grad()
		kl_dual.backward()
		self.alpha_kl_opt.step()
		ent_dual = -(self.log_alpha_dist_ent * (log_prob.detach() + self.target_entropy_dist)).mean()
		self.alpha_dist_ent_opt.zero_grad()
		ent_dual.backward()
		self.alpha_dist_ent_opt.step()
		self.alpha_dist_ent = self.log_alpha_dist_ent.exp().item()
		return {"disturbance_loss": loss.item(), "divergence": kl.mean().item(), "kl_multiplier": alpha_kl.item(), "violation": violation}

	# -------------------------------------------------------------- save/load

	def save(self, step, out_dir):
		os.makedirs(out_dir, exist_ok=True)
		for name in ("actor", "critic1", "critic2", "disturbance"):
			torch.save(getattr(self, name).state_dict(), os.path.join(out_dir, f"{name}-{step}.pth"))
		cfg_path = os.path.join(out_dir, "CONFIG.pkl")
		if not os.path.exists(cfg_path):
			with open(cfg_path, "wb") as f:
				pickle.dump(self.cfg, f)

	def load(self, step, model_dir):
		for name in ("actor", "critic1", "critic2", "disturbance"):
			path = os.path.join(model_dir, f"{name}-{step}.pth")
			getattr(self, name).load_state_dict(torch.load(path, map_location=self.device))
		self.critic1_target.load_state_dict(self.critic1.state_dict())
		self.critic2_target.load_state_dict(self.critic2.state_dict())
