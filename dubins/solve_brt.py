"""Ground-truth backward reachable tubes with hj_reachability (optional).

    pip install --upgrade "jax[cuda12]" hj_reachability
    python dubins/solve_brt.py --mode worstcase --out dubins/brt/worstcase_v1_w1.25.npy
    python dubins/solve_brt.py --mode disturbance --dist_bound 0.3 --out dubins/brt/disturbance_v1.0_w1.25_d0.3.npy

worstcase:   the steering sign can be flipped, so the controller cannot steer.
disturbance: additive (dx, dy) in [-d, d]^2 that minimizes the value.
Grid: [-2, 2]^2 x [0, 2pi], 51^3, horizon 2.8 s, margin max(|p| - 0.5, -0.25).
"""

import argparse

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np
from hj_reachability import dynamics, sets


class Dubins(dynamics.ControlAndDisturbanceAffineDynamics):
	def __init__(self, speed=1.0, turn_rate=1.25, dist_bound=0.0, steerable=True):
		self.speed, self.steerable = speed, steerable
		control = sets.Box(jnp.array([-turn_rate]), jnp.array([turn_rate]))
		disturbance = sets.Box(jnp.array([-dist_bound] * 2), jnp.array([dist_bound] * 2))
		super().__init__("max", "min", control, disturbance)

	def open_loop_dynamics(self, state, time):
		return jnp.array([self.speed * jnp.cos(state[2]), self.speed * jnp.sin(state[2]), 0.0])

	def control_jacobian(self, state, time):
		return jnp.array([[0.0], [0.0], [1.0]])

	def disturbance_jacobian(self, state, time):
		return jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

	def optimal_control_and_disturbance(self, state, time, grad_value):
		u, d = super().optimal_control_and_disturbance(state, time, grad_value)
		return (u if self.steerable else 0 * u), d


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--mode", choices=["nominal", "worstcase", "disturbance"], required=True)
	p.add_argument("--dist_bound", type=float, default=0.3)
	p.add_argument("--out", required=True)
	args = p.parse_args()

	extent = 1.1 if args.mode == "nominal" else 2.0
	system = Dubins(steerable=args.mode != "worstcase", dist_bound=args.dist_bound if args.mode == "disturbance" else 0.0)
	grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
		hj.sets.Box(np.array([-extent, -extent, 0.0]), np.array([extent, extent, 2 * np.pi])), (51, 51, 51), periodic_dims=2)
	margin = jnp.clip(jnp.linalg.norm(grid.states[..., :2], axis=-1) - 0.5, -0.25, None)
	settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
	values = hj.step(settings, system, grid, 0.0, margin, -2.8)
	np.save(args.out, np.asarray(values, dtype=np.float32))
	print(f"saved {args.out}")


if __name__ == "__main__":
	main()
