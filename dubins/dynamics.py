import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# environment constants (shared by data generation, training and evaluation)
SPEED = 1.0
TURN_RATE = 1.25
DT = 0.05
X_MIN, X_MAX, Y_MIN, Y_MAX = -1.5, 1.5, -1.5, 1.5
BUFFER = 0.1
OBS_X, OBS_Y, OBS_R = 0.0, 0.0, 0.5
IMAGE_SIZE = 128


class Renderer:
	"""Top-down 128x128 RGB image of the car and the circular obstacle."""

	def __init__(self, size=IMAGE_SIZE):
		self.fig = plt.figure(figsize=(1, 1), dpi=size)
		self.ax = self.fig.add_axes([0, 0, 1, 1])
		self.ax.set_xlim([X_MIN, X_MAX])
		self.ax.set_ylim([Y_MIN, Y_MAX])
		self.ax.axis("off")
		self.ax.add_patch(Circle([OBS_X, OBS_Y], OBS_R, edgecolor="#b3b3b3", facecolor="#b3b3b3", linewidth=2))
		self.artists = []

	def __call__(self, state):
		x, y, th = (float(v) for v in state[:3])
		for a in self.artists:
			a.remove()
		q = self.ax.quiver(x, y, DT * SPEED * np.cos(th), DT * SPEED * np.sin(th), angles="xy", scale_units="xy",
						   minlength=0, width=0.1, scale=0.15, color="black", zorder=3)
		sc = self.ax.scatter(x, y, s=20, color="black", zorder=3)
		self.artists = [q, sc]
		self.fig.canvas.draw()
		w, h = self.fig.canvas.get_width_height()
		return np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()


def step(state, action, disturbance=None):
	"""Dubins car step. `disturbance` is an additive (dx, dy) velocity, or None."""
	nxt = np.empty_like(state, dtype=np.float32)
	nxt[..., 0] = state[..., 0] + SPEED * DT * np.cos(state[..., 2])
	nxt[..., 1] = state[..., 1] + SPEED * DT * np.sin(state[..., 2])
	nxt[..., 2] = state[..., 2] + DT * action
	if disturbance is not None:
		nxt[..., 0] += disturbance[..., 0] * DT
		nxt[..., 1] += disturbance[..., 1] * DT
	return nxt


def in_failure(state):
	return np.hypot(state[..., 0] - OBS_X, state[..., 1] - OBS_Y) < OBS_R


def out_of_bounds(state):
	return (np.abs(state[..., 0]) > X_MAX - BUFFER) | (np.abs(state[..., 1]) > Y_MAX - BUFFER)


def sample_init_state(rng):
	"""Uniform position outside the obstacle, heading roughly toward the center."""
	while True:
		x = rng.uniform(X_MIN + BUFFER, X_MAX - BUFFER)
		y = rng.uniform(Y_MIN + BUFFER, Y_MAX - BUFFER)
		if np.hypot(x - OBS_X, y - OBS_Y) >= OBS_R:
			break
	th = (np.arctan2(-y, -x) + rng.normal(0, 1)) % (2 * np.pi)
	return np.array([x, y, th], dtype=np.float32)
