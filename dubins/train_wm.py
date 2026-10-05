"""Train the Dubins world model, then the state density model on its latents.

    python dubins/train_wm.py --config dubins/configs/wm.yaml \
        --dataset_path data/dubins/naughty_train.hdf5 --out checkpoints/dubins/wm_naughty.pt
"""

import torch
from tqdm import trange

from dubins.data import load_episodes
from dubins.task import ACT_SPACE, OBS_SPACE
from latentdisturbance import config as config_lib
from latentdisturbance.dreamer import tools
from latentdisturbance.dreamer.agent import Dreamer, checkpoint, make_dataset

DEFAULTS = dict(dataset_path="data/dubins/naughty_train.hdf5", out="checkpoints/dubins/wm.pt",
				num_train_trajs=4800, wm_steps=10000, density_steps=10000, save_every=5000,
				logdir="logs/dubins/wm", wandb_project="dubins_wm", log_every=500)


def main():
	cfg = config_lib.parse(defaults=DEFAULTS)
	tools.set_seed_everywhere(cfg.seed)
	cfg.num_actions = 1
	out = config_lib.resolve(cfg.out)
	out.parent.mkdir(parents=True, exist_ok=True)

	episodes = load_episodes(config_lib.resolve(cfg.dataset_path), stop=cfg.num_train_trajs)
	dataset = make_dataset(episodes, cfg)
	logdir = config_lib.resolve(cfg.logdir) / out.stem
	logger = tools.Logger(logdir, 0, project=cfg.wandb_project)
	logger.config(vars(cfg))
	agent = Dreamer(OBS_SPACE, ACT_SPACE, cfg, logger, dataset).to(cfg.device)
	agent.requires_grad_(False)

	for phase, steps, fn in (("world model", cfg.wm_steps, agent.train_model),
							 ("density", cfg.density_steps, agent.train_density)):
		for step in trange(int(steps), desc=phase):
			fn()
			if (step + 1) % cfg.save_every == 0:
				torch.save(checkpoint(agent), out)
	torch.save(checkpoint(agent), out)
	print(f"saved {out}")


if __name__ == "__main__":
	main()
