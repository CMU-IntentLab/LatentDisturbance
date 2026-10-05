"""Train a (robust) reachability value function inside a pretrained world model.

    python -m latentdisturbance.train --config dubins/configs/reach_naughty.yaml
"""

import importlib
import os
import time

import numpy as np
import torch

from . import config as config_lib
from .env import LatentEnv
from .sac import SAC
from .value_reg import ValueRegularizer

DEFAULTS = config_lib.load_yaml(config_lib.ROOT / "latentdisturbance/configs/default.yaml")


class ReplayBuffer:
	def __init__(self, capacity, dim_state, dim_logit, dim_action):
		self.capacity, self.size, self.pos = int(capacity), 0, 0
		shapes = {"s": dim_state, "logit": dim_logit, "a": dim_action, "s_": dim_state, "logit_": dim_logit}
		self.data = {k: np.zeros((self.capacity, d), np.float32) for k, d in shapes.items()}
		self.data["r"] = np.zeros(self.capacity, np.float32)
		self.data["done"] = np.zeros(self.capacity, bool)

	def add(self, **batch):
		n = len(batch["s"])
		idx = (self.pos + np.arange(n)) % self.capacity
		for k, v in batch.items():
			self.data[k][idx] = v
		self.pos = (self.pos + n) % self.capacity
		self.size = min(self.size + n, self.capacity)

	def sample(self, n, device):
		idx = np.random.randint(0, self.size, size=n)
		return {k: torch.as_tensor(v[idx], device=device) for k, v in self.data.items()}


def build(cfg):
	"""World model + OOD models + dataset (from the task module), env, and agent."""
	task = importlib.import_module(f"{cfg.task}.task")
	wm, models = task.load_world_model(cfg)
	dataset = task.make_dataset(cfg)
	sa_uq = {"none": None, "ensemble": models.get("ensemble"), "density": models.get("sa_density")}[cfg.epistemic_margin]
	env = LatentEnv(cfg, wm, dataset, sa_uq)

	stoch = cfg.dyn_stoch * cfg.dyn_discrete if cfg.dyn_discrete else cfg.dyn_stoch
	agent = SAC(cfg, stoch + cfg.dyn_deter, cfg.policy_action_dim)
	if cfg.use_ood_constraint:
		assert models.get("density") is not None, "use_ood_constraint needs the state density model"
		agent.set_density(models["density"], cfg.ood_threshold)
	if cfg.value_reg_coeff > 0:
		agent.value_reg = ValueRegularizer(cfg, env, task.make_value_reg_dataset(cfg))
	return env, agent


def train(cfg, env, agent, logger=None):
	out_dir = config_lib.resolve(cfg.logdir) / (time.strftime("%m%d_%H%M%S_") + cfg.remark)
	model_dir = out_dir / "model"
	os.makedirs(model_dir, exist_ok=True)
	print(f"logging to {out_dir}")

	buffer, stats = None, {}
	warmup = 20 * cfg.reachability_batch_size
	while agent.updates <= cfg.max_updates:
		s, logit = env.reset()
		for t in range(cfg.max_ep_steps):
			with torch.no_grad():
				a, _ = agent.select_action(s)
				s_, logit_, r = env.step(a)
				if cfg.use_disturbance and cfg.rollout_mode == "pessimistic":
					s_ = agent.perturb(s_, logit_)
					env.set_feat(s_)
			if buffer is None:
				buffer = ReplayBuffer(cfg.memory_capacity, s.shape[1], logit.shape[1], a.shape[1])
			buffer.add(
				s=s.cpu().numpy(), logit=logit.cpu().numpy(), a=a.cpu().numpy(), r=r.cpu().numpy(),
				s_=s_.cpu().numpy(), logit_=logit_.cpu().numpy(),
				done=np.full(len(s), t == cfg.max_ep_steps - 1),
			)
			s, logit = s_, logit_

			if buffer.size <= warmup:
				continue
			for _ in range(cfg.updates_per_step):
				info = agent.update(buffer.sample(cfg.reachability_batch_size, agent.device))
				stats.update(info)
				if agent.updates % cfg.check_period == 0:
					agent.save(agent.updates, model_dir)
				if logger is not None:
					logger.log(info, step=agent.updates)
			if agent.updates % 1000 < cfg.updates_per_step:
				print(f"[{agent.updates:7d}] " + " ".join(f"{k} {v:.3f}" for k, v in sorted(stats.items())), flush=True)
	return out_dir


def main():
	cfg = config_lib.parse(defaults=DEFAULTS)
	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	env, agent = build(cfg)
	logger = None
	if cfg.wandb:
		import wandb
		wandb.init(project=cfg.wandb_project, name=cfg.remark, config=vars(cfg))
		logger = wandb
	train(cfg, env, agent, logger)


if __name__ == "__main__":
	main()
