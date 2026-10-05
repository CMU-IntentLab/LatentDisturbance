"""Egg world model in three phases, each starting from the previous checkpoint.

    python egg/train_wm.py --config egg/configs/wm.yaml --phase wm     --out checkpoints/egg/wm_phase1.pt
    python egg/train_wm.py --config egg/configs/wm.yaml --phase ood    --init checkpoints/egg/wm_phase1.pt --out checkpoints/egg/wm_phase2.pt
    python egg/train_wm.py --config egg/configs/wm.yaml --phase labels --init checkpoints/egg/wm_phase2.pt --out checkpoints/egg/wm.pt

wm:     DINOv3 encoder (frozen), transformer RSSM, decoders and label heads.
ood:    frozen world model; fits logpZO (state), logpZO_za (state-action) and the PENN ensemble.
labels: frozen world model and OOD heads; refits fallen / flipped on `label_traindir`.
"""

import torch
from tqdm import trange

from egg.data import load_norm_stats, make_dataset
from egg.task import build_spaces
from latentdisturbance import config as config_lib
from latentdisturbance.dreamer import tools
from latentdisturbance.dreamer.agent import Dreamer, checkpoint

OOD = ("_disag_ensemble", "_density", "_disag_jrd")
DEFAULTS = dict(phase="wm", init="", out="checkpoints/egg/wm.pt", wm_steps=200000, ood_steps=100000,
				label_steps=5000, save_every=1000, label_balance=0.0, logdir="logs/egg/wm", wandb_project="egg_wm",
				log_every=1000, label_traindir=[
					"data/egg/260709_egg", "data/egg/260709_egg_blacktape", "data/egg/260709_egg_expert",
					"data/egg/260710_egg_expert", "data/egg/260711_dp_rollout", "data/egg/260711_egg_failures",
					"data/egg/260712_egg", "data/egg/260729_success", "data/egg/260711_openpi_rollout"])


def main():
	cfg = config_lib.parse(defaults=DEFAULTS)
	tools.set_seed_everywhere(cfg.seed)
	if cfg.phase == "wm":
		cfg.use_density = cfg.use_ensemble = cfg.use_jrd = False
	cfg.dino = dict(cfg.dino, weights_path=str(config_lib.resolve(cfg.dino["weights_path"])))
	stats = load_norm_stats(config_lib.resolve(cfg.norm_stats))
	obs_space, act_space = build_spaces(cfg, stats)
	cfg.num_actions = act_space.shape[0]

	dirs = cfg.label_traindir if cfg.phase == "labels" else cfg.offline_traindir
	pos_frac = cfg.label_balance if cfg.phase == "labels" else 0.0
	dataset = make_dataset(cfg, stats, [config_lib.resolve(d) for d in dirs], pos_frac=pos_frac)
	out = config_lib.resolve(cfg.out)
	out.parent.mkdir(parents=True, exist_ok=True)
	logger = tools.Logger(config_lib.resolve(cfg.logdir) / out.stem, 0, project=cfg.wandb_project)
	logger.config(vars(cfg))
	agent = Dreamer(obs_space, act_space, cfg, logger, dataset).to(cfg.device)
	agent.requires_grad_(False)

	if cfg.init:
		state = torch.load(config_lib.resolve(cfg.init), map_location="cpu", weights_only=False)["agent_state_dict"]
		missing, unexpected = agent.load_state_dict(state, strict=False)
		assert not unexpected, unexpected[:5]
		assert all(k.startswith(OOD) for k in missing), [k for k in missing if not k.startswith(OOD)][:5]

	steps = {"wm": cfg.wm_steps, "ood": cfg.ood_steps, "labels": cfg.label_steps}[cfg.phase]
	for step in trange(int(steps), desc=cfg.phase):
		if cfg.phase == "wm":
			agent.train_model()
		elif cfg.phase == "ood":
			agent.train_uncertainty()
			agent.train_jrd()
			agent.train_density()
		else:
			agent.train_labels()
		if (step + 1) % cfg.save_every == 0:
			torch.save({**checkpoint(agent), "norm_stats": stats}, out)
	torch.save({**checkpoint(agent), "norm_stats": stats}, out)
	print(f"saved {out}")


if __name__ == "__main__":
	main()
