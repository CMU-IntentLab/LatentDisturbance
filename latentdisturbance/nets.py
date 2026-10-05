"""Actor / critic / disturbance networks.

Module and attribute names follow the original tianshou-style layout so that
released checkpoints load with `strict=True`.
"""

import numpy as np
import torch
import torch.nn as nn

SIGMA_MIN, SIGMA_MAX = -20, 2


class MLP(nn.Module):
	def __init__(self, input_dim, output_dim=0, hidden_sizes=(), norm_layer=None, activation=nn.ReLU):
		super().__init__()
		sizes = [int(input_dim)] + list(hidden_sizes)
		layers = []
		for i, o in zip(sizes[:-1], sizes[1:]):
			layers.append(nn.Linear(i, o))
			if norm_layer is not None:
				layers.append(norm_layer(o))
			layers.append(activation())
		if output_dim > 0:
			layers.append(nn.Linear(sizes[-1], output_dim))
		self.output_dim = output_dim or sizes[-1]
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		return self.model(x.flatten(1))


class Net(nn.Module):
	"""MLP trunk. With `concat=True` the action is part of the input."""

	def __init__(self, state_dim, action_dim=0, hidden_sizes=(), norm_layer=None, concat=False):
		super().__init__()
		input_dim = int(np.prod(state_dim))
		action_dim = int(np.prod(action_dim))
		if concat:
			input_dim += action_dim
		output_dim = 0 if concat else action_dim
		self.model = MLP(input_dim, output_dim, hidden_sizes, norm_layer)
		self.output_dim = self.model.output_dim

	def forward(self, x):
		return self.model(x), None


def _dreamer_stack(inp, units, layers):
	mods, out = [], int(inp)
	for _ in range(int(layers)):
		mods += [nn.Linear(out, units, bias=False), nn.LayerNorm(units, eps=1e-3), nn.SiLU()]
		out = int(units)
	net = nn.Sequential(*mods)
	for m in net.modules():
		if isinstance(m, nn.Linear):
			std = np.sqrt(2.0 / (m.in_features + m.out_features)) / 0.87962566103423978
			nn.init.trunc_normal_(m.weight.data, std=std, a=-2.0 * std, b=2.0 * std)
	return net, out


class SplitCriticNet(nn.Module):
	"""Separate state and action encoders followed by a shared trunk."""

	def __init__(self, state_dim, action_dim, state_units=512, state_layers=2,
				 action_units=128, action_layers=2, trunk_units=512, trunk_layers=2):
		super().__init__()
		self.state_dim, self.action_dim = int(state_dim), int(action_dim)
		self.state_encoder, s_out = _dreamer_stack(self.state_dim, state_units, state_layers)
		self.action_encoder, a_out = _dreamer_stack(self.action_dim, action_units, action_layers)
		self.trunk, self.output_dim = _dreamer_stack(s_out + a_out, trunk_units, trunk_layers)

	def forward(self, x):
		s = self.state_encoder(x[:, :self.state_dim])
		a = self.action_encoder(x[:, self.state_dim:])
		return self.trunk(torch.cat([s, a], -1)), None


class ActorProb(nn.Module):
	"""Gaussian actor head; squashing is done by the caller."""

	def __init__(self, preprocess_net, action_dim):
		super().__init__()
		self.preprocess = preprocess_net
		self.mu = MLP(preprocess_net.output_dim, action_dim)
		self.sigma = MLP(preprocess_net.output_dim, action_dim)

	def forward(self, obs):
		h, _ = self.preprocess(obs)
		sigma = torch.clamp(self.sigma(h), SIGMA_MIN, SIGMA_MAX).exp()
		return (self.mu(h), sigma), None


class Critic(nn.Module):
	def __init__(self, preprocess_net, max_value=1.5):
		super().__init__()
		self.preprocess = preprocess_net
		self.last = MLP(preprocess_net.output_dim, 1)
		self.max_value = max_value

	def forward(self, obs, act=None):
		x = obs.flatten(1)
		if act is not None:
			x = torch.cat([x, act.flatten(1)], 1)
		h, _ = self.preprocess(x)
		return self.max_value * torch.tanh(self.last(h))
