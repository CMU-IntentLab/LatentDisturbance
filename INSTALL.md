# Installation

Run everything from the repository root with `export PYTHONPATH=$PWD:$PYTHONPATH`.

## latentdisturbance (training, Dubins, egg)

```bash
conda create -n latentdisturbance python=3.10 -y
conda activate latentdisturbance
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

- The egg world model uses the frozen DINOv3 ViT-S/16+ encoder, which is not included in the released checkpoint.
  Request access and download `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` from
  [DINOv3](https://github.com/facebookresearch/dinov3), and place it at
  `checkpoints/egg/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`.
- Regenerating the Dubins ground-truth BRTs (optional): `pip install "jax[cuda12]" hj_reachability`.

## isaaclab (Block Pouring)

Isaac Sim 4.5.0, Isaac Lab v2.1.0 (commit `083ca3cd60`):

```bash
conda create -n isaaclab python=3.10 -y
conda activate isaaclab
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
cd third_party/IsaacLab && git checkout 083ca3cd60
git apply ../../block_pouring/isaaclab.patch
./isaaclab.sh --install && cd ../..
pip install -r requirements-sim.txt
```

The task registers on import as `Isaac-BlockPouring-Franka-IK-Rel-v0` (`block_pouring/env`).

Diffusion Policy (task policy) is trained with the [DPPO](https://github.com/irom-princeton/dppo) codebase:

```bash
git clone https://github.com/irom-princeton/dppo.git third_party/dppo
cd third_party/dppo && git checkout cc7234a
git apply ../../block_pouring/diffusion_policy/dppo.patch && cd ../..
export PYTHONPATH=$PWD/third_party/dppo:$PYTHONPATH
```

## Checkpoints and data

Download from Hugging Face ([README](README.md#-download)) into the repository root:

```
checkpoints/dubins/          wm_naughty.pt  wm_positional.pt  filter_naughty/  filter_positional/
checkpoints/block_pouring/   wm.pt  filter/  filter_nominal/  diffusion_policy/
checkpoints/egg/             wm.pt  norm_stats.json  filter/  dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
data/block_pouring/          demo/  rollouts_0804/  rollouts_0908/  rollouts_0909/  teleop_replay/
data/egg/traj/
```
