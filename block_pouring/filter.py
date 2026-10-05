"""Least-restrictive safety filter on the world-model latent.

Each step, with base action a and posterior latent z_t:
  V  = min_i Q_i(z', pi(z')),  z' from the latent dynamics under the worst-case latent disturbance
  u  = ensemble disagreement of (z_t, a)
The pose part of a is replaced by the safety policy pi(z_t) when V < value_thr or
u > uq_thr, and then kept overridden for `hold_steps` more steps.
"""

import torch

from block_pouring.task import load_world_model
from latentdisturbance.sac import SAC


class SafetyFilter:
	def __init__(self, cfg, run_dir, step, value_thr=0.0, uq_thr=3.7, hold_steps=3, worst_case=True, world_model=None):
		"""`world_model`: (wm, ensemble) of another filter, to share one world model between filters."""
		if world_model is None:
			wm, models = load_world_model(cfg)
			world_model = (wm.eval(), models["ensemble"].eval())
		self.wm, self.ensemble = world_model
		self.agent = SAC(cfg, cfg.dyn_stoch * cfg.dyn_discrete + cfg.dyn_deter, cfg.policy_action_dim)
		self.agent.load(step, run_dir)
		self.agent.eval()
		self.cfg = cfg
		self.value_thr, self.uq_thr, self.hold_steps = value_thr, uq_thr, hold_steps
		self.worst_case = worst_case
		self.latent = None

	def _observe(self, obs, prev_action=None):
		data = self.wm.preprocess({k: obs[k] for k in ("front_cam", "wrist_cam", "eef_pos", "eef_quat", "is_first", "is_terminal")})
		embed = self.wm.encoder(data)
		self.latent, _ = self.wm.dynamics.obs_step(self.latent, prev_action, embed, data["is_first"], sample=False)

	@torch.no_grad()
	def reset(self, obs):
		self.latent, self.filtered, self.count = None, False, 0
		self._observe(obs)

	@torch.no_grad()
	def update(self, action, obs):
		self._observe(obs, action)

	@torch.no_grad()
	def __call__(self, action):
		"""Returns (filtered action, info dict)."""
		feat = self.wm.dynamics.get_feat(self.latent)
		nxt = self.wm.dynamics.img_step(self.latent, action)
		nxt_feat = self.wm.dynamics.get_feat(nxt)
		if self.worst_case:
			nxt_feat = self.agent.disturbance_step(nxt_feat, nxt["logit"].reshape(1, -1))[1]
		value = self.agent.value(nxt_feat).item()
		uq = self.ensemble.intrinsic_reward_penn(torch.cat([feat, action], -1)).item()
		p_fail = self.wm.heads["failure"](feat).mean.item()

		if self.filtered and self.count < self.hold_steps:
			self.count += 1
		else:
			self.filtered = value < self.value_thr or uq > self.uq_thr
			self.count = 0
		if self.filtered:
			safe, _ = self.agent.select_action(feat, eval_mode=True)
			action = action.clone()
			action[:, :safe.shape[-1]] = safe
		return action, {"value": value, "uncertainty": uq, "p_failure": p_fail, "filtered": self.filtered}
