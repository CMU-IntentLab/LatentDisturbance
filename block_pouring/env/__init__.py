"""Block Pouring: pick up and tilt the orange block so the green block slides onto the blue block."""

import gymnasium as gym

from .config import ik_rel_env_cfg

TASK_ID = "Isaac-BlockPouring-Franka-IK-Rel-v0"

gym.register(
	id=TASK_ID,
	entry_point="isaaclab.envs:ManagerBasedRLEnv",
	kwargs={"env_cfg_entry_point": ik_rel_env_cfg.FrankaRobustEnvCfg},
	disable_env_checker=True,
)
