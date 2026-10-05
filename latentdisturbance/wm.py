import torch

from .config import resolve
from .dreamer.agent import Dreamer

# frozen pretrained image backbone; released checkpoints omit it and it is loaded from `dino.weights_path`
_PRETRAINED = "_wm.encoder.backbone.net."
# head names used by older checkpoints
_RENAMES = {"heads.margin.layers.Margin_": "heads.margin.layers.Label_margin_"}


def _clean(key):
	key = key.replace("_orig_mod.", "")
	for old, new in _RENAMES.items():
		key = key.replace(old, new)
	return key


def load_world_model(cfg, obs_space, act_space, checkpoint=None):
	"""Build the offline Dreamer agent and load (world model, OOD heads) strictly.

	Modules are left in train mode, as during reach training; call `.eval()` for
	deployment (the DINO encoder jitters its patch positions in train mode).
	Returns the world model and a dict with the optional
	`density` (state), `ensemble` (state-action PENN) and `sa_density`
	(state-action flow) models.
	"""
	compile_flag = cfg.compile
	cfg.compile = False
	agent = Dreamer(obs_space, act_space, cfg, logger=None, dataset=None).to(cfg.device)
	cfg.compile = compile_flag
	path = resolve(checkpoint or cfg.wm_checkpoint)
	state = torch.load(path, map_location="cpu", weights_only=False)["agent_state_dict"]
	state = {_clean(k): v for k, v in state.items()}
	own = agent.state_dict()
	missing = [k for k in own if k not in state and not k.startswith(_PRETRAINED)]
	assert not missing, f"checkpoint {path} is missing {missing[:5]} ({len(missing)} keys)"
	agent.load_state_dict({k: state[k] for k in own if k in state}, strict=False)
	agent.requires_grad_(False)

	models = {"density": agent._density}
	if agent._disag_ensemble is not None:
		key = "sa_density" if getattr(cfg, "disag_type", "ensemble") == "logpzo" else "ensemble"
		models[key] = agent._disag_ensemble
	if agent._disag_jrd is not None:
		models["ensemble"] = agent._disag_jrd
	return agent._wm, models
