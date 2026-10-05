"""One-sided value hinge on latents of known-safe episodes.

    loss = coeff * sum_i mean relu(target - Q_i(s, pi(s)))

The latents are posteriors of real frames plus up to `imagine_steps` prior
steps driven by the recorded actions, drawn with a fixed posterior fraction.
"""

import torch


class ValueRegularizer:
	def __init__(self, cfg, env, dataset):
		self.cfg = cfg
		self.env = env
		self.dataset = dataset
		self.k = int(cfg.value_reg_imagine_steps)
		self.pool = None
		self.steps = None
		self.calls = 0

	def _append(self, feat, step):
		if feat.shape[0] == 0:
			return
		tag = torch.full((feat.shape[0],), step, dtype=torch.uint8, device=feat.device)
		self.pool = feat if self.pool is None else torch.cat([self.pool, feat])
		self.steps = tag if self.steps is None else torch.cat([self.steps, tag])
		n = self.cfg.value_reg_pool
		self.pool, self.steps = self.pool[-n:], self.steps[-n:]

	@torch.no_grad()
	def _refill(self):
		env, wm = self.env, self.env.wm
		latent, _, _, data, keep = env.encode_window(self.dataset)
		self._append(wm.dynamics.get_feat(latent), 0)
		if self.k == 0:
			return
		a = data["action"]
		first = data["is_first"].bool()
		can_step = torch.zeros_like(first)
		can_step[:, :-1] = ~first[:, 1:]
		ahead = torch.zeros_like(first, dtype=torch.long)
		for t in range(first.shape[1] - 2, -1, -1):
			ahead[:, t] = torch.where(can_step[:, t], (ahead[:, t + 1] + 1).clamp(max=self.k), 0)
		ahead = ahead.reshape(-1)
		acts = [torch.roll(a, -(j + 1), dims=1).reshape(-1, a.shape[-1]) for j in range(self.k)]
		if keep is not None:
			ahead, acts = ahead[keep], [x[keep] for x in acts]
		for j in range(self.k):
			latent = wm.dynamics.img_step(latent, acts[j], sample=self.cfg.value_reg_prior_sample)
			self._append(wm.dynamics.get_feat(latent)[ahead > j], j + 1)

	def sample(self, n):
		if self.pool is None:
			for _ in range(64):
				self._refill()
				if self.pool.shape[0] >= self.cfg.value_reg_pool:
					break
		elif self.calls % self.cfg.value_reg_refresh == 0:
			self._refill()
		self.calls += 1
		post = (self.steps == 0).nonzero().reshape(-1)
		img = (self.steps > 0).nonzero().reshape(-1)
		n_post = n if img.numel() == 0 else int(round(n * self.cfg.value_reg_post_frac))
		idx = torch.cat([
			post[torch.randint(post.numel(), (n_post,), device=post.device)],
			img[torch.randint(max(img.numel(), 1), (n - n_post,), device=img.device)],
		])
		return self.pool[idx]

	def __call__(self, agent):
		feat = self.sample(self.cfg.value_reg_batch)
		with torch.no_grad():
			a, _ = agent.select_action(feat, eval_mode=True)
		q1, q2 = agent.critic1(feat, a), agent.critic2(feat, a)
		target = self.cfg.value_reg_target
		loss = self.cfg.value_reg_coeff * (torch.relu(target - q1).mean() + torch.relu(target - q2).mean())
		v = torch.min(q1, q2).detach()
		return loss, {"value_reg_v": v.mean().item(), "value_reg_active": (v < target).float().mean().item()}
