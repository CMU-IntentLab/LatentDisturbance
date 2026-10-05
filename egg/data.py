"""Egg episodes read directly from hdf5, with frames kept as JPEG bytes in RAM.

Per trajectory:
    data/actions (T, 7)  data/joint_states (T, 7)  data/gripper_states (T,)
    data/camera_<cam> (T,) JPEG, 192x256       labels/{fallen,flipped} (T,)
    attrs fall_frame / flip_frame (-1 if never)

Batches: <cam>_cam (B, L, H, W, 3) uint8, state (B, L, 8) and action (B, L, 7)
scaled to [-1, 1] (action[t] led into o_t), fallen / flipped labels, and
is_first / is_last / is_terminal flags.
"""

import json
import pathlib

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

LABEL_KEYS = ("fallen", "flipped")


def normalize(x, lo, hi):
	span = np.asarray(hi, np.float64) - np.asarray(lo, np.float64)
	span = np.where(span < 1e-8, 1.0, span)
	return (2.0 * (np.asarray(x, np.float64) - lo) / span - 1.0).astype(np.float32)


def load_norm_stats(path):
	with open(path) as f:
		return json.load(f)


class EpisodeStore:
	"""All episodes in memory; `tail_frames` > 0 also adds the last frames of every
	failure-free episode in `tail_dirs` as short extra episodes."""

	def __init__(self, dirs, cam_keys, stats, tail_dirs=(), tail_frames=0):
		self.cam_keys = list(cam_keys)
		bounds = [np.asarray(stats[k][b], np.float64) for k in ("action", "state") for b in ("low", "high")]
		self.episodes = [self._load(fp, *bounds) for d in dirs for fp in sorted(pathlib.Path(d).expanduser().glob("*.hdf5"))]
		assert self.episodes, f"no .hdf5 episodes under {list(dirs)}"
		self.n_full = len(self.episodes)
		for d in tail_dirs if tail_frames > 0 else ():
			for fp in sorted(pathlib.Path(d).expanduser().glob("*.hdf5")):
				ep = self._load(fp, *bounds, keep_last=int(tail_frames))
				if ep is not None and len(ep["action"]) >= 2:
					self.episodes.append(ep)
		self.lengths = np.array([len(ep["action"]) for ep in self.episodes], np.int64)
		pos = [np.max(np.stack([ep[k] for k in LABEL_KEYS]), 0) > 0.5 for ep in self.episodes]
		self.pos_counts = np.array([int(m.sum()) for m in pos], np.int64)
		self.first_pos = np.array([int(np.argmax(m)) if m.any() else -1 for m in pos], np.int64)
		print(f"{len(self.episodes)} episodes ({len(self.episodes) - self.n_full} tails), {int(self.lengths.sum())} frames")

	def _load(self, fp, a_lo, a_hi, s_lo, s_hi, keep_last=0):
		with h5py.File(fp, "r") as f:
			raw = np.asarray(f["data/actions"][()], np.float64)
			T = len(raw)
			state = np.concatenate([f["data/joint_states"][()], f["data/gripper_states"][()][:, None]], 1)
			action = np.zeros(raw.shape, np.float32)
			action[1:] = normalize(np.clip(raw, a_lo, a_hi)[:-1], a_lo, a_hi)
			success = int(f.attrs.get("fall_frame", -1)) < 0 and int(f.attrs.get("flip_frame", -1)) < 0
			labels = {k: np.asarray(f[f"labels/{k}"][()], np.float32) if f"labels/{k}" in f else np.zeros(T, np.float32)
					  for k in LABEL_KEYS}
			lo = 0
			if keep_last:
				if not success or any(v.max() > 0.5 for v in labels.values()):
					return None
				lo = max(0, T - keep_last)
			n = T - lo
			ep = {"action": action[lo:].copy(), "state": normalize(state[lo:], s_lo, s_hi),
				  "reward": np.zeros(n, np.float32), "is_first": np.zeros(n, bool),
				  "is_last": np.zeros(n, bool), "is_terminal": np.zeros(n, bool)}
			ep["reward"][-1] = float(success)
			ep["is_first"][0] = ep["is_last"][-1] = ep["is_terminal"][-1] = True
			for k in LABEL_KEYS:
				ep[k] = labels[k][lo:].copy()
			for key in self.cam_keys:
				blobs = [bytes(b) for b in f[f"data/camera_{key[:-4]}"][lo:T]]
				offsets = np.concatenate([[0], np.cumsum([len(b) for b in blobs])]).astype(np.int64)
				ep[key] = (np.frombuffer(b"".join(blobs), np.uint8), offsets)
		return ep

	def frames(self, ep, key, start, stop):
		buf, off = ep[key]
		return np.stack([cv2.imdecode(buf[off[t]:off[t + 1]], cv2.IMREAD_COLOR) for t in range(start, stop)])


class Windows(IterableDataset):
	"""Length-weighted episode choice, random start, episodes concatenated to fill
	a window. `pos_frac` of windows straddle a failure onset, `tail_frac` of
	windows are built from tail clips only."""

	def __init__(self, store, length, seed=0, pos_frac=0.0, tail_frac=0.0):
		super().__init__()
		self.store, self.length, self.seed = store, length, seed
		self.pos_frac, self.tail_frac = float(pos_frac), float(tail_frac)
		self.tail_idx = np.arange(store.n_full, len(store.episodes))
		if self.tail_frac > 0:
			w = store.lengths[self.tail_idx].astype(np.float64)
			self.tail_p = w / w.sum()
		self.pos_idx = np.flatnonzero((store.pos_counts > 0) & (store.lengths >= 2))
		if self.pos_frac > 0:
			w = store.pos_counts[self.pos_idx].astype(np.float64)
			self.pos_p = w / w.sum()

	def _episode(self, rng, tail=False):
		if tail:
			idx = int(self.tail_idx[rng.choice(len(self.tail_idx), p=self.tail_p)])
			return idx, int(self.store.lengths[idx])
		p = self.store.lengths / self.store.lengths.sum()
		while True:
			idx = int(rng.choice(len(self.store.episodes), p=p))
			if self.store.lengths[idx] >= 2:
				return idx, int(self.store.lengths[idx])

	def _take(self, ep, start, stop):
		out = {k: v[start:stop].copy() for k, v in ep.items() if not k.endswith("_cam")}
		for k in self.store.cam_keys:
			out[k] = self.store.frames(ep, k, start, stop)
		return out

	def __iter__(self):
		worker = torch.utils.data.get_worker_info()
		rng = np.random.RandomState(self.seed + (worker.id if worker else 0))
		while True:
			tail = self.tail_frac > 0 and rng.rand() < self.tail_frac
			if not tail and self.pos_frac > 0 and rng.rand() < self.pos_frac:
				idx = int(self.pos_idx[rng.choice(len(self.pos_idx), p=self.pos_p)])
				total = int(self.store.lengths[idx])
				start = int(np.clip(self.store.first_pos[idx] - rng.randint(1, self.length + 1), 0, total - 2))
			else:
				idx, total = self._episode(rng, tail)
				start = int(rng.randint(0, total - 1))
			ret = self._take(self.store.episodes[idx], start, min(start + self.length, total))
			ret["is_first"][0] = True
			while len(ret["action"]) < self.length:
				size = len(ret["action"])
				idx, total = self._episode(rng, tail)
				chunk = self._take(self.store.episodes[idx], 0, min(self.length - size, total))
				ret = {k: np.append(ret[k], chunk[k], axis=0) for k in ret}
				ret["is_first"][size] = True
			yield ret


def _collate(samples):
	return {k: np.stack([s[k] for s in samples]) for k in samples[0]}


def make_dataset(cfg, stats, dirs, seed=None, pos_frac=0.0, tail_dirs=(), tail_frames=0, tail_frac=0.0, batch_length=None):
	cam_keys = [f"{c}_cam" for c in cfg.cameras]
	store = EpisodeStore(dirs, cam_keys, stats, tail_dirs, tail_frames)
	ds = Windows(store, batch_length or cfg.batch_length, cfg.seed if seed is None else seed, pos_frac, tail_frac)
	kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=_collate)
	if cfg.num_workers > 0:
		kwargs.update(persistent_workers=True, prefetch_factor=4)
	return iter(DataLoader(ds, **kwargs))
