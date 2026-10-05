"""Hooks used by latentdisturbance.train."""

import numpy as np
from gymnasium import spaces

from dubins import dynamics as dyn
from dubins.data import load_episodes, window_iterator
from latentdisturbance.config import resolve
from latentdisturbance.wm import load_world_model as _load

OBS_SPACE = spaces.Dict({"image": spaces.Box(0, 255, (dyn.IMAGE_SIZE, dyn.IMAGE_SIZE, 3), np.uint8)})
ACT_SPACE = spaces.Box(-dyn.TURN_RATE, dyn.TURN_RATE, (1,), np.float32)


def load_world_model(cfg, checkpoint=None):
	cfg.num_actions = 1
	return _load(cfg, OBS_SPACE, ACT_SPACE, checkpoint)


def make_dataset(cfg):
	return window_iterator(resolve(cfg.dataset_path), cfg.batch_size, cfg.batch_length, seed=cfg.seed)


def calibration_trajectories(cfg, paths):
	for path in paths:
		for ep in load_episodes(path).values():
			yield {k: v[None] for k, v in ep.items() if k != "state"}
