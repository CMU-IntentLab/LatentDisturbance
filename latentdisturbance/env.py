"""Batched world-model imagination for safety-filter training.

A reset encodes a (B, T) window of real trajectories into B*T start latents.
Each step returns the failure margin of the current latent and imagines one
step with the latent dynamics.

    l(z)    = 1.5 tanh(margin_scale * min_i(thr_i - P_i(failure | z)))
    r(z, a) = min(l(z), epistemic margin(z, a))   (UNISafe, optional)
"""

import torch


class LatentEnv:
	def __init__(self, cfg, wm, dataset, sa_uq=None):
		self.cfg = cfg
		self.wm = wm
		self.dataset = dataset
		self.sa_uq = sa_uq
		self.discrete = cfg.dyn_discrete > 0
		self.failure_heads = dict(cfg.failure_heads)
		for name in self.failure_heads:
			assert name in wm.heads, f"world model has no '{name}' head"
		self.epistemic = cfg.epistemic_margin
		assert self.epistemic in ("none", "ensemble", "density"), self.epistemic
		if self.epistemic != "none":
			assert sa_uq is not None, f"epistemic_margin '{self.epistemic}' needs a state-action OOD model"
		self.replay_last_action = bool(cfg.replay_last_action)
		self.full_history = bool(cfg.full_history_starts)

	# --------------------------------------------------------------- reward

	@torch.no_grad()
	def failure_margin(self, feat):
		margin = None
		for name, thr in self.failure_heads.items():
			m = thr - self.wm.heads[name](feat).mean
			margin = m if margin is None else torch.minimum(margin, m)
		return (1.5 * torch.tanh(self.cfg.margin_scale * margin)).squeeze(-1)

	@torch.no_grad()
	def epistemic_score(self, feat, action):
		x = torch.cat([feat, action], -1)
		if self.epistemic == "ensemble":
			return self.sa_uq.intrinsic_reward_penn(x)[:, 0]
		return self.sa_uq.score_flat(x)

	@torch.no_grad()
	def epistemic_margin(self, feat, action, threshold=None):
		"""Positive in-distribution, -1 far out of distribution."""
		thr = self.cfg.epistemic_threshold if threshold is None else threshold
		diff = thr - self.epistemic_score(feat, action)
		band = self.cfg.epistemic_band
		if self.cfg.epistemic_shape == "step":
			return torch.where(diff.abs() > band, torch.sign(diff), diff)
		return (diff / band).clamp(-1.0, 1.0)

	# ---------------------------------------------------------------- reset

	def _encode(self, data):
		chunk = self.cfg.encode_chunk
		B, T = data["action"].shape[:2]
		n = max(1, chunk // max(T, 1)) if chunk else B
		if n >= B:
			return self.wm.encoder(data)
		return torch.cat([self.wm.encoder({k: v[i:i + n] for k, v in data.items()}) for i in range(0, B, n)], 0)

	@torch.no_grad()
	def encode_window(self, dataset=None):
		"""Encode one batch of real windows -> flat latent, action executed from each row, raw data."""
		data = self.wm.preprocess(next(self.dataset if dataset is None else dataset))
		post, _ = self.wm.dynamics.observe(self._encode(data), data["action"], data["is_first"])
		a = data["action"]
		nxt = torch.roll(a, -1, dims=1)
		if self.cfg.replay_action == "next":
			# datasets store the action that led into o_t, so a[t+1] is taken from o_t
			use_next = torch.roll(data["is_first"], -1, dims=1) == 0
		else:
			# reuse a[t]; a segment's first entry is a zero placeholder, use a[t+1] there
			use_next = data["is_first"] != 0
		use_next[:, -1] = False
		act_from = torch.where(use_next.bool().unsqueeze(-1), nxt, a).reshape(-1, a.shape[-1])
		latent = {k: v.reshape(-1, *v.shape[2:]) for k, v in post.items()}
		keep = self._full_history_rows(latent)
		if keep is not None:
			latent = {k: v[keep] for k, v in latent.items()}
			act_from = act_from[keep]
		return latent, act_from, post, data, keep

	def _full_history_rows(self, latent):
		"""Transformer RSSM: keep only rows whose history window is complete."""
		if not self.full_history or "hist_mask" not in latent:
			return None
		K = latent["hist_mask"].shape[-1]
		keep = latent["hist_mask"].sum(-1) >= K - 0.5
		return None if bool(keep.all()) else keep

	def _logit(self):
		if self.discrete:
			return self.latent["logit"].reshape(len(self.feat), -1)
		return torch.cat([self.latent["mean"], self.latent["std"]], -1)

	@torch.no_grad()
	def reset(self):
		latent, act_from, _, _, _ = self.encode_window()
		self.latent = latent
		self.recorded_action = act_from
		self.feat = self.wm.dynamics.get_feat(latent)
		self.logit = self._logit()
		self.epistemic_threshold_rows = None
		if self.epistemic == "density" and self.cfg.epistemic_adaptive:
			# each row may drift `epistemic_adaptive_alpha` beyond its own start state
			base = self.epistemic_score(self.feat, act_from) + self.cfg.epistemic_adaptive_alpha
			self.epistemic_threshold_rows = base.clamp(min=self.cfg.epistemic_threshold)
		return self.feat, self.logit

	# ----------------------------------------------------------------- step

	def wm_action(self, action):
		action = action * self.cfg.action_scale
		if self.replay_last_action:
			action = torch.cat([action, self.recorded_action[:, -1:]], -1)
		return action

	@torch.no_grad()
	def step(self, action):
		"""Reward of the current latent, then one imagined step under `action`."""
		action = self.wm_action(action)
		reward = self.failure_margin(self.feat)
		if self.epistemic != "none":
			reward = torch.minimum(reward, self.epistemic_margin(self.feat, action, self.epistemic_threshold_rows))
		reset_rows = None
		if self.cfg.cont_reset_threshold > 0:
			reset_rows = self.wm.heads["cont"](self.feat).mean.reshape(-1) < self.cfg.cont_reset_threshold
		self.latent = self.wm.dynamics.img_step(self.latent, action)
		if reset_rows is not None and reset_rows.any():
			fresh, _, _, _, _ = self.encode_window()
			idx = reset_rows.nonzero(as_tuple=True)[0]
			idx = idx[idx < fresh["deter"].shape[0]]
			self.latent = {k: v.index_copy(0, idx, fresh[k][idx]) for k, v in self.latent.items()}
		self.feat = self.wm.dynamics.get_feat(self.latent).reshape(len(action), -1)
		self.logit = self._logit()
		return self.feat, self.logit, reward

	@torch.no_grad()
	def set_feat(self, feat):
		"""Write a disturbed latent back into the rollout."""
		self.feat = feat
		stoch = feat[:, :-self.cfg.dyn_deter]
		shape = (-1, self.cfg.dyn_stoch, self.cfg.dyn_discrete) if self.discrete else (-1, self.cfg.dyn_stoch)
		self.latent["stoch"] = stoch.reshape(shape)
