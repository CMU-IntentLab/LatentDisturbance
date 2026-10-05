"""Hooks used by latentdisturbance.train, plus the observation/action spaces."""

import pathlib

import numpy as np
from gymnasium import spaces
from torch.utils.data import DataLoader, Dataset

from latentdisturbance.config import resolve
from latentdisturbance.wm import load_world_model as _load

OBS_SPACE = spaces.Dict({
	"eef_pos": spaces.Box(-np.inf, np.inf, (3,), np.float32),
	"eef_quat": spaces.Box(-np.inf, np.inf, (4,), np.float32),
	"front_cam": spaces.Box(0, 255, (128, 128, 3), np.uint8),
	"wrist_cam": spaces.Box(0, 255, (128, 128, 3), np.uint8),
})
ACT_SPACE = spaces.Box(-1, 1, (7,), np.float32)
KEYS = ("front_cam", "wrist_cam", "eef_pos", "eef_quat", "action", "is_first", "is_last", "is_terminal",
		"reward", "discount", "failure", "success")


def load_world_model(cfg, checkpoint=None):
	cfg.num_actions = 7
	return _load(cfg, OBS_SPACE, ACT_SPACE, checkpoint)


class Windows(Dataset):
	"""Length-T windows of the recorded episodes, read from disk on demand."""

	def __init__(self, dirs, T):
		self.T = T
		self.index = []
		for d in dirs:
			for path in sorted(pathlib.Path(resolve(d)).glob("*.npz")):
				n = len(np.load(path)["reward"])
				self.index += [(str(path), s) for s in range(n - T + 1)]
		print(f"{len(self.index)} windows of length {T} from {len(dirs)} directories")

	def __len__(self):
		return len(self.index)

	def __getitem__(self, i):
		path, s = self.index[i]
		with np.load(path) as ep:
			out = {k: ep[k][s:s + self.T].astype(np.float32) for k in KEYS}
		out["is_first"][0] = 1.0
		return out


def make_dataset(cfg, dirs=None):
	loader = DataLoader(Windows(dirs or cfg.dataset_dirs, cfg.batch_length), batch_size=cfg.batch_size, shuffle=True,
						num_workers=4, drop_last=True, persistent_workers=True)

	def iterate():
		while True:
			for batch in loader:
				yield {k: v.numpy() for k, v in batch.items()}

	return iterate()


def calibration_trajectories(cfg, dirs):
	for d in dirs:
		for path in sorted(pathlib.Path(d).glob("*.npz")):
			with np.load(path) as ep:
				yield {k: ep[k][None].astype(np.float32) for k in KEYS}
