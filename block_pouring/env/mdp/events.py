from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos[env_ids] = torch.tensor(default_pose, device=env.device)


def reset_root_states_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfgs: list[SceneEntityCfg],
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges."""

    # poses
    asset = env.scene[asset_cfgs[0].name]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples_pos = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples_vel = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    for asset_cfg in asset_cfgs:
        asset: RigidObject | Articulation  = env.scene[asset_cfg.name]
        # get default root state
        root_states = asset.data.default_root_state[env_ids].clone()

    
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples_pos[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples_pos[:, 3], rand_samples_pos[:, 4], rand_samples_pos[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    

        velocities = root_states[:, 7:13] + rand_samples_vel

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_partial_and_randomize_physics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # ranges for cube_1 x,y only, keys: "x","y"
    cube1_xy_range: dict[str, tuple[float, float]],
    pick_bottom_pose: dict[str, tuple[float, float]],
    pick_top_pose:    dict[str, tuple[float, float]],
    # optional: {"cube_1": (m_lo, m_hi), ...}
    randomize_pose = True,
    randomize_physics = None,
):
    """
    1) Randomize only the (x,y) of “cube_1” within cube1_xy_range.
    2) Randomize small xyz+roll/pitch/yaw deltas for pick_cube_bottom and pick_cube_1.
    3) (Optional) Randomize mass & friction/restitution via PhysX.
    """
    N = len(env_ids)
    device = env.scene.env_origins.device

    # helper: sample uniform [min, max] per axis for each env using torch.rand
    def sample_ranges(rng: dict[str, tuple[float, float]], axes: list[str]) -> torch.Tensor:
        bounds = torch.tensor([rng.get(a, (0.0, 0.0)) for a in axes],
                              dtype=torch.float32, device=device)
        lo, hi = bounds[:, 0], bounds[:, 1]
        rand = torch.rand((N, len(axes)), device=device)
        return lo.unsqueeze(0) + rand * (hi - lo).unsqueeze(0)

    if randomize_pose:
        # 1) cube_1 x,y only
        cube1 = env.scene["cube_1"]
        default = cube1.data.default_root_state[env_ids].clone()
        new_xy  = sample_ranges(cube1_xy_range, ["x", "y"])
        pos     = default[:, :3].clone()
        pos[:, :2] += new_xy
        quat    = default[:, 3:7]
        cube1.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        cube1.write_root_velocity_to_sim(default[:, 7:13],        env_ids=env_ids)

        # 2) small pose tweaks for bottom & top cubes
        for name, pose_rng in [
            ("pick_cube_bottom", pick_bottom_pose),
            ("pick_cube_1",      pick_top_pose),
        ]:
            asset  = env.scene[name]
            default = asset.data.default_root_state[env_ids].clone()
            dpos   = sample_ranges(pose_rng, ["x", "y", "z"])
            d_eul  = sample_ranges(pose_rng, ["roll", "pitch", "yaw"])# [N,3]
            dq     = math_utils.quat_from_euler_xyz(d_eul[:,0], d_eul[:,1], d_eul[:,2])
            pos    = default[:, :3] + dpos
            quat   = math_utils.quat_mul(default[:, 3:7], dq)
            asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim(default[:, 7:13],       env_ids=env_ids)

    if not randomize_physics:
        return

    # 3a) randomize masses via PhysX
    m_lo, m_hi = 0.02, 0.2
    asset = env.scene['pick_cube_1']
    # Sample exactly N new masses on the same device as the PhysX view
    new_masses = torch.rand((N, 1), device=asset.root_physx_view.get_masses().device)
    new_masses = new_masses * (m_hi - m_lo) + m_lo

    # This will overwrite only those N envs in‐place
    asset.root_physx_view.set_masses(new_masses, env_ids.cpu())

    # 3b) randomize friction & restitution via PhysX
    friction_ranges = { "pick_cube_1": ((0.2, 0.8),(0.1, 0.5),(0.05, 0.3))}
    for name, (sf_range, df_range, re_range) in friction_ranges.items():
        asset = env.scene[name]
        sf_lo, sf_hi = sf_range
        df_lo, df_hi = df_range
        re_lo, re_hi = re_range
        # 1) sample df in [df_lo, min(df_hi, sf_hi)]
        df_hi_eff = min(df_hi, sf_hi)
        df = torch.rand(N, device='cpu') * (df_hi_eff - df_lo) + df_lo

        # 2) sample sf in [max(sf_lo, df), sf_hi]
        sf_lo_eff = torch.maximum(torch.full((N,), sf_lo, device='cpu'), df)
        sf = torch.rand(N, device='cpu') * (sf_hi - sf_lo_eff) + sf_lo_eff
        re = torch.rand(N, device='cpu') * (re_hi - re_lo) + re_lo
        mats    = asset.root_physx_view.get_material_properties()
        new_m   = torch.stack([sf, df, re], dim=1).unsqueeze(1)
        mats[env_ids] = new_m
        asset.root_physx_view.set_material_properties(mats, env_ids.cpu())


    # Sampling from a set


