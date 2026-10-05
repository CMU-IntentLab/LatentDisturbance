"""Train the block-pouring world model on the offline dataset, then its OOD heads.

    python block_pouring/train_wm.py --config block_pouring/configs/wm.yaml --out checkpoints/block_pouring/wm.pt

Phase 1 trains the world model (and the PENN ensemble alongside it);
phase 2 fine-tunes the ensemble and fits the state density on frozen latents.
"""

import torch
from tqdm import trange

from block_pouring.task import ACT_SPACE, OBS_SPACE
from latentdisturbance import config as config_lib
from latentdisturbance.dreamer import tools
from latentdisturbance.dreamer.agent import Dreamer, checkpoint, make_dataset

DEFAULTS = dict(out="checkpoints/block_pouring/wm.pt", wm_steps=200000, ood_steps=100000, save_every=10000,
				dataset_size=1000000, logdir="logs/block_pouring/wm", wandb_project="block_pouring_wm",
				log_every=1000, init_checkpoint="")


def main():
	cfg = config_lib.parse(defaults=DEFAULTS)
	tools.set_seed_everywhere(cfg.seed)
	out = config_lib.resolve(cfg.out)
	out.parent.mkdir(parents=True, exist_ok=True)

	episodes = None
	for d in cfg.dataset_dirs:
		episodes = tools.load_episodes(config_lib.resolve(d), limit=cfg.dataset_size, episodes=episodes, offline=True)
	logger = tools.Logger(config_lib.resolve(cfg.logdir) / out.stem, 0, project=cfg.wandb_project)
	logger.config(vars(cfg))
	agent = Dreamer(OBS_SPACE, ACT_SPACE, cfg, logger, make_dataset(episodes, cfg)).to(cfg.device)
	agent.requires_grad_(False)
	if cfg.init_checkpoint:
		state = torch.load(config_lib.resolve(cfg.init_checkpoint), map_location="cpu", weights_only=False)
		agent.load_state_dict(state["agent_state_dict"])

	def save(step):
		if (step + 1) % cfg.save_every == 0:
			torch.save(checkpoint(agent), out)

	for step in trange(int(cfg.wm_steps), desc="world model"):
		agent.train_model()
		save(step)
	for step in trange(int(cfg.ood_steps), desc="ood heads"):
		agent.train_uncertainty()
		agent.train_density()
		save(step)
	torch.save(checkpoint(agent), out)
	print(f"saved {out}")


if __name__ == "__main__":
	main()
