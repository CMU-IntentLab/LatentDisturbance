import collections
import os
import pathlib
import time
import random

import cv2

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


def symlog(x):
	return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
	def __init__(self, model):
		self._model = model

	def __enter__(self):
		self._model.requires_grad_(requires_grad=True)

	def __exit__(self, *args):
		self._model.requires_grad_(requires_grad=False)


class DummyLogger:
	def __init__(self, logdir=None, step=0):
		self.step = step

	def config(self, *args, **kwargs):
		pass

	def scalar(self, *args, **kwargs):
		pass

	def image(self, *args, **kwargs):
		pass

	def video(self, *args, **kwargs):
		pass

	def write(self, *args, **kwargs):
		pass


class Logger:
	"""Minimal wandb logger. Set WANDB_MODE=disabled to turn it off."""

	def __init__(self, logdir, step, project="latentdisturbance"):
		import wandb
		self._wandb = wandb
		self._last_step = None
		self._last_time = None
		self._scalars = {}
		self._videos = {}
		self.step = step
		name = "_".join(str(logdir).split("/")[-2:])
		wandb.init(project=project, config={"logdir": str(logdir)}, name=name)

	def config(self, config_dict):
		config_dict = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in config_dict.items()}
		self._wandb.config.update(config_dict, allow_val_change=True)

	def scalar(self, name, value):
		self._scalars[name] = float(value)

	def image(self, name, value):
		pass

	def video(self, name, value):
		self._videos[name] = np.array(value)

	def write(self, fps=False, step=False, print_cli=True, **kwargs):
		step = step or self.step
		scalars = dict(self._scalars)
		if fps:
			scalars["fps"] = self._compute_fps(step)
		if print_cli:
			print(f"[{step}]", " / ".join(f"{k} {v:.3g}" for k, v in scalars.items()))
		self._wandb.log({"step": step, **scalars}, step=step)
		for name, value in self._videos.items():
			if np.issubdtype(value.dtype, np.floating):
				value = np.clip(255 * value, 0, 255).astype(np.uint8)
			B, T, H, W, C = value.shape
			value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
			self._wandb.log({name: self._wandb.Video(value, fps=20, format="mp4")}, step=step)
		self._scalars = {}
		self._videos = {}

	def _compute_fps(self, step):
		if self._last_step is None:
			self._last_time = time.time()
			self._last_step = step
			return 0
		duration = time.time() - self._last_time
		steps = step - self._last_step
		self._last_time += duration
		self._last_step = step
		return steps / max(duration, 1e-8)


def compress_frames(frames, quality=90):
	out = []
	for frame in frames:
		ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
		assert ok, "jpeg encode failed"
		out.append(buf.tobytes())
	return np.array(out, dtype=object)


def decompress_frame(blob, flags=cv2.IMREAD_COLOR):
	img = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), flags)
	assert img is not None, "jpeg decode failed"
	return img


def from_generator(generator, batch_size, offline=False):
	while True:
		batch = [next(generator) for _ in range(batch_size)]
		data = {}
		for key in batch[0]:
			if offline and "cam" in key:
				data[key] = np.stack([[decompress_frame(img) for img in b[key]] for b in batch], 0)
			else:
				data[key] = np.stack([b[key] for b in batch], 0)
		yield data


def sample_episodes(episodes, length, seed=0):
	np_random = np.random.RandomState(seed)
	while True:
		size = 0
		ret = None
		p = np.array(
			[len(next(iter(episode.values()))) for episode in episodes.values()]
		)
		p = p / np.sum(p)
		while size < length:
			episode = np_random.choice(list(episodes.values()), p=p)
			total = len(next(iter(episode.values())))
			# make sure at least one transition included
			if total < 2:
				continue
			if not ret:
				index = int(np_random.randint(0, total - 1))
				ret = {
					k: v[index : min(index + length, total)].copy()
					for k, v in episode.items()
					if "log_" not in k
				}
				if "is_first" in ret:
					ret["is_first"][0] = True
			else:
				# 'is_first' comes after 'is_last'
				index = 0
				possible = length - size
				ret = {
					k: np.append(
						ret[k], v[index : min(index + possible, total)].copy(), axis=0
					)
					for k, v in episode.items()
					if "log_" not in k
				}
				if "is_first" in ret:
					ret["is_first"][size] = True
			size = len(next(iter(ret.values())))
		yield ret


def load_episodes(directory, limit=None, episodes=None, offline=False):
	"""Load *.npz episodes (newest name first). `offline` keeps camera frames as jpeg bytes."""
	directory = pathlib.Path(directory).expanduser()
	episodes = collections.OrderedDict() if episodes is None else episodes
	total = 0
	for filename in reversed(sorted(directory.glob("*.npz"))):
		try:
			with filename.open("rb") as f:
				episode = dict(np.load(f))
		except Exception as e:
			print(f"Could not load {filename}: {e}")
			continue
		if offline:
			for k in episode:
				if "cam" in k:
					episode[k] = compress_frames(episode[k])
		episodes["expert_" + filename.stem] = episode
		total += len(episode["reward"]) - 1
		if limit and total >= limit:
			break
	print(f"loaded {len(episodes)} episodes ({total} steps) from {directory}")
	return episodes


class SampleDist:
	def __init__(self, dist, samples=100):
		self._dist = dist
		self._samples = samples

	@property
	def name(self):
		return "SampleDist"

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def mean(self):
		samples = self._dist.sample(self._samples)
		return torch.mean(samples, 0)

	def mode(self):
		sample = self._dist.sample(self._samples)
		logprob = self._dist.log_prob(sample)
		return sample[torch.argmax(logprob)][0]

	def entropy(self):
		sample = self._dist.sample(self._samples)
		logprob = self.log_prob(sample)
		return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
	def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
		if logits is not None and unimix_ratio > 0.0:
			probs = F.softmax(logits, dim=-1)
			probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
			logits = torch.log(probs)
			super().__init__(logits=logits, probs=None)
		else:
			super().__init__(logits=logits, probs=probs)

	def mode(self):
		_mode = F.one_hot(
			torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
		)
		return _mode.detach() + super().logits - super().logits.detach()

	def sample(self, sample_shape=(), seed=None):
		if seed is not None:
			raise ValueError("need to check")
		sample = super().sample(sample_shape).detach()
		probs = super().probs
		while len(probs.shape) < len(sample.shape):
			probs = probs[None]
		sample += probs - probs.detach()
		return sample


class DiscDist:
	def __init__(
		self,
		logits,
		low=-20.0,
		high=20.0,
		transfwd=symlog,
		transbwd=symexp,
		device="cuda",
	):
		self.logits = logits
		self.probs = torch.softmax(logits, -1)
		self.buckets = torch.linspace(low, high, steps=255, device=device)
		self.width = (self.buckets[-1] - self.buckets[0]) / 255
		self.transfwd = transfwd
		self.transbwd = transbwd

	def mean(self):
		_mean = self.probs * self.buckets
		return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

	def mode(self):
		_mode = self.probs * self.buckets
		return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

	# Inside OneHotCategorical, log_prob is calculated using only max element in targets
	def log_prob(self, x):
		x = self.transfwd(x)
		below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
		above = len(self.buckets) - torch.sum(
			(self.buckets > x[..., None]).to(torch.int32), dim=-1
		)
		# this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
		below = torch.clip(below, 0, len(self.buckets) - 1)
		above = torch.clip(above, 0, len(self.buckets) - 1)
		equal = below == above

		dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
		dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
		total = dist_to_below + dist_to_above
		weight_below = dist_to_above / total
		weight_above = dist_to_below / total
		target = (
			F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
			+ F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
		)
		log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
		target = target.squeeze(-2)

		return (target * log_pred).sum(-1)

	def log_prob_target(self, target):
		log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
		return (target * log_pred).sum(-1)


class DiscDistDense:
	def __init__(
		self,
		logits,
		shape,
		low=0,
		high=4,
		transfwd=symlog,
		transbwd=symexp,
		device="cuda",
	):
		logits_shape = logits.shape[:-1]
		self.logits = logits.reshape(*logits_shape, *shape)

		self.probs = torch.softmax(self.logits, -1)
		self.buckets = torch.linspace(low, high, steps=255, device=device)
		self.width = (self.buckets[-1] - self.buckets[0]) / 255
		self.transfwd = transfwd
		self.transbwd = transbwd

	def mean(self):
		_mean = self.probs * self.buckets
		return self.transbwd(torch.sum(_mean, dim=-1))

	def mode(self):
		_mode = self.probs * self.buckets
		return self.transbwd(torch.sum(_mode, dim=-1))

	# Inside OneHotCategorical, log_prob is calculated using only max element in targets
	def log_prob(self, x):
		x = self.transfwd(x)
		below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
		above = len(self.buckets) - torch.sum(
			(self.buckets > x[..., None]).to(torch.int32), dim=-1
		)
		# this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
		below = torch.clip(below, 0, len(self.buckets) - 1)
		above = torch.clip(above, 0, len(self.buckets) - 1)
		equal = below == above

		dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
		dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
		total = dist_to_below + dist_to_above
		weight_below = dist_to_above / total
		weight_above = dist_to_below / total
		target = (
			F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
			+ F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
		)
		log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
		target = target.squeeze(-2)

		return (target * log_pred).sum(-1)
	
	def quantile(self, q: float):
		# cdf: [B, D, K]
		cdf = torch.cumsum(self.probs, dim=-1)
		K = self.probs.shape[-1]

		# index of first bin where CDF >= q
		idx = torch.sum(cdf < q, dim=-1)
		idx = torch.clamp(idx, 0, K - 1)

		idx_prev = torch.clamp(idx - 1, 0, K - 1)

		# gather per-batch per-dim scalars
		cdf_prev = torch.gather(cdf, -1, idx_prev.unsqueeze(-1)).squeeze(-1)
		p_bin = torch.gather(self.probs, -1, idx.unsqueeze(-1)).squeeze(-1)

		b_idx = self.buckets[idx]
		b_prev = self.buckets[idx_prev]

		eps = 1e-8
		t = torch.clamp((q - cdf_prev) / (p_bin + eps), 0.0, 1.0)
		q_trans = (1.0 - t) * b_prev + t * b_idx

		return self.transbwd(q_trans)


class MSEDist:
	def __init__(self, mode, agg="sum"):
		self._mode = mode
		self._agg = agg

	def mode(self):
		return self._mode

	def mean(self):
		return self._mode

	def log_prob(self, value):
		assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
		distance = (self._mode - value) ** 2
		if self._agg == "mean":
			loss = distance.mean(list(range(len(distance.shape)))[2:])
		elif self._agg == "sum":
			loss = distance.sum(list(range(len(distance.shape)))[2:])
		else:
			raise NotImplementedError(self._agg)
		return -loss


class SymlogDist:
	def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
		self._mode = mode
		self._dist = dist
		self._agg = agg
		self._tol = tol

	def mode(self):
		return symexp(self._mode)

	def mean(self):
		return symexp(self._mode)

	def log_prob(self, value):
		assert self._mode.shape == value.shape
		if self._dist == "mse":
			distance = (self._mode - symlog(value)) ** 2.0
			distance = torch.where(distance < self._tol, 0, distance)
		elif self._dist == "abs":
			distance = torch.abs(self._mode - symlog(value))
			distance = torch.where(distance < self._tol, 0, distance)
		else:
			raise NotImplementedError(self._dist)
		if self._agg == "mean":
			loss = distance.mean(list(range(len(distance.shape)))[2:])
		elif self._agg == "sum":
			loss = distance.sum(list(range(len(distance.shape)))[2:])
		else:
			raise NotImplementedError(self._agg)
		return -loss


class ContDist:
	def __init__(self, dist=None, absmax=None):
		super().__init__()
		self._dist = dist
		self.mean = dist.mean
		self.absmax = absmax

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def entropy(self):
		return self._dist.entropy()

	def mode(self):
		out = self._dist.mean
		if self.absmax is not None:
			out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
		return out

	def sample(self, sample_shape=()):
		out = self._dist.rsample(sample_shape)
		if self.absmax is not None:
			out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
		return out

	def log_prob(self, x):
		return self._dist.log_prob(x)


class Bernoulli:
	def __init__(self, dist=None):
		super().__init__()
		self._dist = dist
		self.mean = dist.mean

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def entropy(self):
		return self._dist.entropy()

	def mode(self):
		_mode = torch.round(self._dist.mean)
		return _mode.detach() + self._dist.mean - self._dist.mean.detach()

	def sample(self, sample_shape=()):
		return self._dist.rsample(sample_shape)

	def log_prob(self, x):
		_logits = self._dist.base_dist.logits
		log_probs0 = -F.softplus(_logits)
		log_probs1 = -F.softplus(-_logits)

		return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torchd.normal.Normal):
	def __init__(self, loc, scale, threshold=1, **kwargs):
		super().__init__(loc, scale, **kwargs)
		self._threshold = threshold

	def log_prob(self, event):
		return -(
			torch.sqrt((event - self.mean) ** 2 + self._threshold**2) - self._threshold
		)

	def mode(self):
		return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
	def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
		super().__init__(loc, scale)
		self._low = low
		self._high = high
		self._clip = clip
		self._mult = mult

	def sample(self, sample_shape):
		event = super().sample(sample_shape)
		if self._clip:
			clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
			event = event - event.detach() + clipped.detach()
		if self._mult:
			event *= self._mult
		return event


class TanhBijector(torchd.Transform):
	def __init__(self, validate_args=False, name="tanh"):
		super().__init__()

	def _forward(self, x):
		return torch.tanh(x)

	def _inverse(self, y):
		y = torch.where(
			(torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
		)
		y = torch.atanh(y)
		return y

	def _forward_log_det_jacobian(self, x):
		log2 = torch.math.log(2.0)
		return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


class Optimizer:
	def __init__(
		self,
		name,
		parameters,
		lr,
		eps=1e-4,
		clip=None,
		wd=None,
		wd_pattern=r".*",
		opt="adam",
		use_amp=False,
	):
		assert 0 <= wd < 1
		assert not clip or 1 <= clip
		self._name = name
		self._parameters = parameters
		self._clip = clip
		self._wd = wd
		self._wd_pattern = wd_pattern
		self._opt = {
			"adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
			"nadam": lambda: NotImplemented(f"{opt} is not implemented"),
			"adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
			"sgd": lambda: torch.optim.SGD(parameters, lr=lr),
			"momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
		}[opt]()
		self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

	def __call__(self, loss, params, retain_graph=True):
		assert len(loss.shape) == 0, loss.shape
		metrics = {}
		metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
		self._opt.zero_grad()
		self._scaler.scale(loss).backward(retain_graph=retain_graph)
		self._scaler.unscale_(self._opt)
		norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
		if self._wd:
			self._apply_weight_decay(params)
		self._scaler.step(self._opt)
		self._scaler.update()
		self._opt.zero_grad()
		metrics[f"{self._name}_grad_norm"] = to_np(norm)
		return metrics

	def _apply_weight_decay(self, varibs):
		nontrivial = self._wd_pattern != r".*"
		if nontrivial:
			raise NotImplementedError
		for var in varibs:
			var.data = (1 - self._wd) * var.data


def args_type(default):
	def parse_string(x):
		if default is None:
			return x
		if isinstance(default, bool):
			return bool(["False", "True"].index(x))
		if isinstance(default, int):
			return float(x) if ("e" in x or "." in x) else int(x)
		if isinstance(default, (list, tuple)):
			return tuple(args_type(default[0])(y) for y in x.split(","))
		return type(default)(x)

	def parse_object(x):
		if isinstance(default, (list, tuple)):
			return tuple(x)
		return x

	return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
	last = start
	indices = range(inputs[0].shape[0])
	flag = True
	for index in indices:
		inp = lambda x: (_input[x] for _input in inputs)
		last = fn(last, *inp(index))
		if flag:
			if type(last) == type({}):
				outputs = {
					key: value.clone().unsqueeze(0) for key, value in last.items()
				}
			else:
				outputs = []
				for _last in last:
					if type(_last) == type({}):
						outputs.append(
							{
								key: value.clone().unsqueeze(0)
								for key, value in _last.items()
							}
						)
					else:
						outputs.append(_last.clone().unsqueeze(0))
			flag = False
		else:
			if type(last) == type({}):
				for key in last.keys():
					outputs[key] = torch.cat(
						[outputs[key], last[key].unsqueeze(0)], dim=0
					)
			else:
				for j in range(len(outputs)):
					if type(last[j]) == type({}):
						for key in last[j].keys():
							outputs[j][key] = torch.cat(
								[outputs[j][key], last[j][key].unsqueeze(0)], dim=0
							)
					else:
						outputs[j] = torch.cat(
							[outputs[j], last[j].unsqueeze(0)], dim=0
						)
	if type(last) == type({}):
		outputs = [outputs]
	return outputs


class Every:
	def __init__(self, every):
		self._every = every
		self._last = None

	def __call__(self, step):
		if not self._every:
			return 0
		if self._last is None:
			self._last = step
			return 1
		count = int((step - self._last) / self._every)
		self._last += self._every * count
		return count


class Once:
	def __init__(self):
		self._once = True

	def __call__(self):
		if self._once:
			self._once = False
			return True
		return False


class Until:
	def __init__(self, until):
		self._until = until

	def __call__(self, step):
		if not self._until:
			return True
		return step < self._until


def weight_init(m):
	if isinstance(m, nn.Linear):
		in_num = m.in_features
		out_num = m.out_features
		denoms = (in_num + out_num) / 2.0
		scale = 1.0 / denoms
		std = np.sqrt(scale) / 0.87962566103423978
		nn.init.trunc_normal_(
			m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
		)
		if hasattr(m.bias, "data"):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		space = m.kernel_size[0] * m.kernel_size[1]
		in_num = space * m.in_channels
		out_num = space * m.out_channels
		denoms = (in_num + out_num) / 2.0
		scale = 1.0 / denoms
		std = np.sqrt(scale) / 0.87962566103423978
		nn.init.trunc_normal_(
			m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
		)
		if hasattr(m.bias, "data"):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.LayerNorm):
		m.weight.data.fill_(1.0)
		if hasattr(m.bias, "data"):
			m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
	def f(m):
		if isinstance(m, nn.Linear):
			in_num = m.in_features
			out_num = m.out_features
			denoms = (in_num + out_num) / 2.0
			scale = given_scale / denoms
			limit = np.sqrt(3 * scale)
			nn.init.uniform_(m.weight.data, a=-limit, b=limit)
			if hasattr(m.bias, "data"):
				m.bias.data.fill_(0.0)
		elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
			space = m.kernel_size[0] * m.kernel_size[1]
			in_num = space * m.in_channels
			out_num = space * m.out_channels
			denoms = (in_num + out_num) / 2.0
			scale = given_scale / denoms
			limit = np.sqrt(3 * scale)
			nn.init.uniform_(m.weight.data, a=-limit, b=limit)
			if hasattr(m.bias, "data"):
				m.bias.data.fill_(0.0)
		elif isinstance(m, nn.LayerNorm):
			m.weight.data.fill_(1.0)
			if hasattr(m.bias, "data"):
				m.bias.data.fill_(0.0)

	return f


def tensorstats(tensor, prefix=None):
	metrics = {
		"mean": to_np(torch.mean(tensor)),
		"std": to_np(torch.std(tensor)),
		"min": to_np(torch.min(tensor)),
		"max": to_np(torch.max(tensor)),
	}
	if prefix:
		metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
	return metrics


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def enable_deterministic_run():
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(obj, path="", out=None, visited=None):
	"""State dicts of every torch optimizer reachable through modules / Optimizer wrappers."""
	out = {} if out is None else out
	visited = set() if visited is None else visited
	if id(obj) in visited:
		return out
	visited.add(id(obj))
	attrs = dict(vars(obj))
	if isinstance(obj, nn.Module):
		attrs.update(obj.named_children())
	for name, attr in attrs.items():
		new_path = f"{path}.{name}" if path else name
		if isinstance(attr, torch.optim.Optimizer):
			out[new_path] = attr.state_dict()
		elif isinstance(attr, (nn.Module, Optimizer)):
			recursively_collect_optim_state_dict(attr, new_path, out, visited)
	return out


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
	for path, state_dict in optimizers_state_dicts.items():
		keys = path.split(".")
		obj_now = obj
		for key in keys:
			obj_now = getattr(obj_now, key)
		obj_now.load_state_dict(state_dict)
