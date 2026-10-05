import collections

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dubins import dynamics as dyn


def load_episodes(path, start=0, stop=None):
	"""HDF5 trajectories -> {name: episode dict} in the dreamer episode format."""
	episodes = collections.OrderedDict()
	with h5py.File(path, "r") as f:
		names = sorted(f["images"].keys())[start:stop]
		for name in names:
			states = f["states"][name][:]
			dones = f["dones"][name][:].astype(bool)
			T = len(states)
			episodes[name] = {
				"image": f["images"][name][:],
				"action": f["actions"][name][:].astype(np.float32),
				"margin": dyn.in_failure(states).astype(np.float32),
				"state": states,
				"is_first": np.arange(T) == 0,
				"is_last": dones,
				"is_terminal": dones,
				"reward": np.zeros(T, np.float32),
				"discount": np.ones(T, np.float32),
			}
	return episodes


class Windows(Dataset):
	"""All length-`T` windows of trajectories that are at least `T` long."""

	def __init__(self, path, T, stop=None):
		self.eps = list(load_episodes(path, stop=stop).values())
		self.index = [(i, s) for i, ep in enumerate(self.eps) for s in range(len(ep["action"]) - T + 1)]
		self.T = T

	def __len__(self):
		return len(self.index)

	def __getitem__(self, idx):
		i, s = self.index[idx]
		out = {k: v[s:s + self.T] for k, v in self.eps[i].items() if k != "state"}
		out["is_first"] = out["is_first"].copy()
		out["is_first"][0] = True
		return out


def window_iterator(path, batch_size, T, stop=None, seed=0):
	loader = DataLoader(Windows(path, T, stop), batch_size=batch_size, shuffle=True, drop_last=True,
						generator=torch.Generator().manual_seed(seed))
	while True:
		for batch in loader:
			yield {k: v.numpy() for k, v in batch.items()}
