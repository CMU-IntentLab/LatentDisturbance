import os

import numpy as np
import torch
from torch import nn

from . import models, tools, uncertainty


class Dreamer(nn.Module):
	"""Offline world model + post-hoc OOD heads.

	Attribute names (_wm, _disag_ensemble, _density, _disag_jrd) define the
	checkpoint keys and must not change.
	"""

	def __init__(self, obs_space, act_space, config, logger, dataset):
		super().__init__()
		self._config = config
		self._logger = logger
		self._dataset = dataset
		self._step = logger.step if logger is not None else 0
		self._metrics = {}

		self._wm = models.WorldModel(obs_space, act_space, self._step, config)
		if config.compile and os.name != "nt":
			self._wm = torch.compile(self._wm)

		self._disag_ensemble = None
		if config.use_ensemble:
			if getattr(config, "disag_type", "ensemble") == "logpzo":
				self._disag_ensemble = uncertainty.logpZO_za(config)
			else:
				self._disag_ensemble = uncertainty.OneStepPredictor(config, self._wm)
		self._density = uncertainty.logpZO(config) if config.use_density else None
		self._disag_jrd = None
		if getattr(config, "use_jrd", False):
			self._disag_jrd = uncertainty.OneStepPredictor(config, self._wm)

	def _log(self, metrics):
		for name, value in metrics.items():
			self._metrics.setdefault(name, []).append(value)
		if self._logger is None or (self._step + 1) % getattr(self._config, "log_every", 100):
			return
		for name, values in self._metrics.items():
			self._logger.scalar(name, float(np.mean(values)))
		self._metrics = {}
		self._logger.write(fps=True)

	def _tick(self, video=False):
		if video and self._config.video_pred_log and (self._step + 1) % 1000 == 0:
			pred = self._wm.video_pred(next(self._dataset), ensemble=self._disag_ensemble)
			if pred is not None and self._logger is not None:
				self._logger.video("train_openl", pred)
		self._step += 1
		if self._logger is not None:
			self._logger.step = self._step

	def train_model(self):
		_, _, metrics = self._wm._train(next(self._dataset), ensemble=self._disag_ensemble)
		self._log(metrics)
		self._tick(video=True)

	def train_uncertainty(self):
		metrics = self._wm.train_uncertainty_only(next(self._dataset), ensemble=self._disag_ensemble)
		self._log(metrics)
		self._tick(video=True)

	def train_density(self):
		self._log(self._wm.train_density_only(next(self._dataset), density=self._density))
		self._tick()

	def train_jrd(self):
		self._log(self._wm.train_jrd_only(next(self._dataset), ensemble=self._disag_jrd))
		self._tick()

	def train_labels(self):
		self._log(self._wm.train_labels_only(next(self._dataset)))
		self._tick()


def make_dataset(episodes, config):
	generator = tools.sample_episodes(episodes, config.batch_length)
	return tools.from_generator(generator, config.batch_size, offline=True)


def checkpoint(agent):
	return {
		"agent_state_dict": agent.state_dict(),
		"optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
	}
