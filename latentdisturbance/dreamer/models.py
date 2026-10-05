import re
import torch
from torch import nn


from . import networks, networks_v2, tools
from .uncertainty import OneStepPredictor, logpZO

to_np = lambda x: x.detach().cpu().numpy()


class WorldModel(nn.Module):
	def __init__(self, obs_space, act_space, step, config):
		super(WorldModel, self).__init__()
		self._step = step
		self._use_amp = True if config.precision == 16 else False
		self._config = config
		shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
		if getattr(config, "enc_arch", "cnn") == "dino":
			self.encoder = networks_v2.DinoMultiEncoder(shapes, config)
		else:
			self.encoder = networks.MultiEncoder(shapes, **config.encoder)
		self.embed_size = self.encoder.outdim
		if getattr(config, "dyn_arch", "gru") == "transformer":
			self.dynamics = networks_v2.TransformerRSSM(config, self.embed_size)
		else:
			self.dynamics = networks.RSSM(
				config.dyn_stoch,
				config.dyn_deter,
				config.dyn_hidden,
				config.dyn_rec_depth,
				config.dyn_discrete,
				config.act,
				config.norm,
				config.dyn_mean_act,
				config.dyn_std_act,
				config.dyn_min_std,
				config.unimix_ratio,
				config.initial,
				config.num_actions,
				self.embed_size,
				config.device,
			)
			self.dynamics.shift_action = bool(getattr(config, "observe_shift_action", False))
		self.heads = nn.ModuleDict()
		if config.dyn_discrete:
			feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
		else:
			feat_size = config.dyn_stoch + config.dyn_deter
		decoder_cfg = dict(config.decoder)
		if not getattr(config, "pixel_recon", True):
			# Disable the pixel branch entirely; MLP heads (state) remain.
			decoder_cfg["cnn_keys"] = "$^"
		if isinstance(decoder_cfg.get("minres"), (list, tuple)):
			self.heads["decoder"] = networks_v2.MultiDecoderV2(
				feat_size, shapes, **decoder_cfg
			)
		else:
			self.heads["decoder"] = networks.MultiDecoder(
				feat_size, shapes, **decoder_cfg
			)
		if dict(config.reward_head).get("enabled", True):
			self.heads["reward"] = networks.MLP(
				feat_size,
				(255,) if config.reward_head["dist"] == "symlog_disc" else (),
				config.reward_head["layers"],
				config.units,
				config.act,
				config.norm,
				dist=config.reward_head["dist"],
				outscale=config.reward_head["outscale"],
				device=config.device,
				name="Reward",
			)
		self.heads["cont"] = networks.MLP(
			feat_size,
			(),
			config.cont_head["layers"],
			config.units,
			config.act,
			config.norm,
			dist="binary",
			outscale=config.cont_head["outscale"],
			device=config.device,
			name="Cont",
		)

		if 'failure_head' in self._config:
			self.heads["failure"] = networks.MLP(
				feat_size,
				(),
				config.failure_head["layers"],
				config.units,
				config.act,
				config.norm,
				dist="binary",
				outscale=config.failure_head["outscale"],
				device=config.device,
				name="Cont",
			)

		if 'success_head' in self._config:
			self.heads["success"] = networks.MLP(
				feat_size,
				(),
				config.success_head["layers"],
				config.units,
				config.act,
				config.norm,
				dist="binary",
				outscale=config.success_head["outscale"],
				device=config.device,
				name="Cont",
			)

		label_head = dict(getattr(config, "label_head", {}))
		for key in getattr(config, "label_heads", []):
			self.heads[key] = networks.MLP(
				feat_size,
				(),
				label_head.get("layers", 2),
				label_head.get("units", config.units),
				config.act,
				config.norm,
				dist="binary",
				outscale=label_head.get("outscale", 1.0),
				device=config.device,
				name=f"Label_{key}",
			)

		dino_recon = dict(getattr(config, "dino_recon", {"enabled": False}))
		if getattr(config, "enc_arch", "cnn") == "dino" and dino_recon.get("enabled", False):
			self.heads["dino"] = networks_v2.DinoReconHead(
				feat_size,
				self.encoder._views,
				self.encoder.recon_out_dim,
				hidden=dino_recon.get("hidden", 256),
				layers=dino_recon.get("layers", 2),
			)

		for name in config.grad_heads:
			assert name in self.heads, name
		opt_params = [p for p in self.parameters() if p.requires_grad]
		if getattr(config, "enc_arch", "cnn") == "dino" and dict(config.dino).get(
			"train_encoder", False
		):
			# Finetuned backbone gets its own (much smaller) learning rate.
			backbone_ids = {id(p) for p in self.encoder.backbone.parameters()}
			opt_params = [
				{
					"params": [p for p in opt_params if id(p) in backbone_ids],
					"lr": float(dict(config.dino).get("encoder_lr", 1e-5)),
				},
				{"params": [p for p in opt_params if id(p) not in backbone_ids]},
			]
		self._model_opt = tools.Optimizer(
			"model",
			opt_params,
			config.model_lr,
			config.opt_eps,
			config.grad_clip,
			config.weight_decay,
			opt=config.opt,
			use_amp=self._use_amp,
		)
		print(
			f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
		)
		# other losses are scaled by 1.0.
		self._scales = dict(
			cont=config.cont_head["loss_scale"],
		)
		if "reward" in self.heads:
			self._scales["reward"] = config.reward_head["loss_scale"]

		if 'failure_head' in self._config:
			self._scales['failure'] =config.failure_head["loss_scale"]
		if 'success_head' in self._config:
			self._scales['success'] =config.success_head["loss_scale"]
		if "dino" in self.heads:
			self._scales["dino"] = dino_recon.get("loss_scale", 1.0)
		for key in getattr(config, "label_heads", []):
			self._scales[key] = label_head.get("loss_scale", 1.0)

		self._label_keys = [
			k for k in getattr(config, "label_heads", []) if k in self.heads
		]
		self._label_params = [
			p for k in self._label_keys for p in self.heads[k].parameters()
		]
		if self._label_keys:
			self._label_opt = tools.Optimizer(
				"label",
				self._label_params,
				getattr(config, "label_lr", config.model_lr),
				config.opt_eps,
				config.grad_clip,
				config.weight_decay,
				opt=config.opt,
				use_amp=self._use_amp,
			)

	def _train(self, data, ensemble: OneStepPredictor| None = None):
		data = self.preprocess(data)

		with tools.RequiresGrad(self):
			with torch.cuda.amp.autocast(self._use_amp):
				embed = self.encoder(data)
				post, prior = self.dynamics.observe(
					embed, data["action"], data["is_first"]
				)
				kl_free = self._config.kl_free
				dyn_scale = self._config.dyn_scale
				rep_scale = self._config.rep_scale
				kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
					post, prior, kl_free, dyn_scale, rep_scale
				)
				assert kl_loss.shape == embed.shape[:2], kl_loss.shape
				preds = {}
				for name, head in self.heads.items():
					if name == "dino":
						continue
					grad_head = name in self._config.grad_heads
					feat = self.dynamics.get_feat(post)
					feat = feat if grad_head else feat.detach()
					pred = head(feat)
					if type(pred) is dict:
						preds.update(pred)
					else:
						preds[name] = pred
				losses = {}
				for name, pred in preds.items():

					if name == "kl":
						kl_target = self.dynamics.get_kl_divergence(post, prior)
						loss = -pred.log_prob(kl_target.detach().unsqueeze(-1)).sum()
					else:
						loss = -pred.log_prob(data[name])

					assert loss.shape == embed.shape[:2], (name, loss.shape)
					losses[name] = loss

				if "dino" in self.heads:
					grad_head = "dino" in self._config.grad_heads
					feat = self.dynamics.get_feat(post)
					feat = feat if grad_head else feat.detach()
					dino_preds = self.heads["dino"](feat)
					# Sum over the channel dim (global descriptor) so loss_scales
					# are comparable across heads.
					dino_loss = 0.0
					for view, pred in dino_preds.items():
						target = self.encoder.last_tokens[view]
						dino_loss = dino_loss + (pred - target).pow(2).sum(-1)
					assert dino_loss.shape == embed.shape[:2], dino_loss.shape
					losses["dino"] = dino_loss

				scaled = {
					key: value * self._scales.get(key, 1.0)
					for key, value in losses.items()
				}
				model_loss = sum(scaled.values()) + kl_loss
			metrics = self._model_opt(torch.mean(model_loss), self.parameters())

		metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
		metrics["kl_free"] = kl_free
		metrics["dyn_scale"] = dyn_scale
		metrics["rep_scale"] = rep_scale
		metrics["dyn_loss"] = to_np(dyn_loss)
		metrics["rep_loss"] = to_np(rep_loss)
		metrics["kl"] = to_np(torch.mean(kl_value))
		with torch.cuda.amp.autocast(self._use_amp):
			metrics["prior_ent"] = to_np(
				torch.mean(self.dynamics.get_dist(prior).entropy())
			)
			metrics["post_ent"] = to_np(
				torch.mean(self.dynamics.get_dist(post).entropy())
			)
			context = dict(
				embed=embed,
				feat=self.dynamics.get_feat(post),
				kl=kl_value,
				postent=self.dynamics.get_dist(post).entropy(),
			)

		# PENN ensemble trained alongside the world model (logpzo heads are fit afterwards)

		if ensemble is not None and getattr(self._config, "disag_type", "ensemble") == "ensemble":
			with tools.RequiresGrad(ensemble):
				stoch = post["stoch"]
				if self._config.dyn_discrete:
					stoch = torch.reshape(
						stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),))
					)
				target = {
					"embed": embed,
					"stoch": stoch,
					"deter": post["deter"],
					"feat": feat,
				}[self._config.disag_target]
				with torch.no_grad():
					inputs = self.dynamics.get_feat(post)
				ensemble_mets = ensemble.train_ensemble_penn_fixed(inputs, data["action"], target, data["is_first"])
				metrics.update({k: v for k, v in ensemble_mets.items()})

		post = {k: v.detach() for k, v in post.items()}

		return post, context, metrics

	def train_uncertainty_only(self, data, ensemble: OneStepPredictor| None = None):
		data = self.preprocess(data)

		with torch.no_grad() :
			embed = self.encoder(data)
			post, prior = self.dynamics.observe(
				embed, data["action"], data["is_first"]
			)

		if ensemble is not None:
			if getattr(self._config, "disag_type", "ensemble") == "logpzo":
				# Flow-matching density over (feat_t, a_{t+1}) pairs.
				with torch.no_grad():
					inputs = self.dynamics.get_feat(post)
				with tools.RequiresGrad(ensemble):
					metrics = ensemble.train_step(inputs, data["action"], data["is_first"])
				post = {k: v.detach() for k, v in post.items()}
				return metrics
			with tools.RequiresGrad(ensemble):
				stoch = post["stoch"]
				if self._config.dyn_discrete:
					stoch = torch.reshape(
						stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),))
					)
				target = {
					"embed": embed,
					"stoch": stoch,
					"deter": post["deter"],
				}[self._config.disag_target]
				with torch.no_grad():
					inputs = self.dynamics.get_feat(post)

				for _ in range(1):
					ensemble_mets = ensemble.train_ensemble_penn_fixed(inputs, data["action"], target, data["is_first"])
				metrics = ensemble_mets

		post = {k: v.detach() for k, v in post.items()}

		return metrics

	def train_jrd_only(self, data, ensemble: OneStepPredictor| None = None):
		data = self.preprocess(data)

		with torch.no_grad():
			embed = self.encoder(data)
			post, prior = self.dynamics.observe(
				embed, data["action"], data["is_first"]
			)

		metrics = {}
		if ensemble is not None:
			with tools.RequiresGrad(ensemble):
				stoch = post["stoch"]
				if self._config.dyn_discrete:
					stoch = torch.reshape(
						stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),))
					)
				target = {
					"embed": embed,
					"stoch": stoch,
					"deter": post["deter"],
				}[self._config.disag_target]
				with torch.no_grad():
					inputs = self.dynamics.get_feat(post)
				metrics = ensemble.train_ensemble_penn_fixed(
					inputs, data["action"], target, data["is_first"]
				)

		post = {k: v.detach() for k, v in post.items()}
		return metrics

	def train_density_only(self, data, density: logpZO| None = None):
		data = self.preprocess(data)

		with torch.no_grad():
			embed = self.encoder(data)
			post, prior = self.dynamics.observe(
				embed, data["action"], data["is_first"]
			)

			inputs = self.dynamics.get_feat(post)

		with tools.RequiresGrad(density):
			metrics = density.train_step(inputs)
			uq = density(inputs)

		return metrics

	def train_labels_only(self, data):
		"""Refit ONLY the per-frame label heads (fallen/flipped) on frozen features."""
		assert self._label_keys, (
			"no label heads to train -- config.label_heads is empty"
		)
		data = self.preprocess(data)

		with torch.no_grad():
			embed = self.encoder(data)
			post, _ = self.dynamics.observe(
				embed, data["action"], data["is_first"]
			)
			feat = self.dynamics.get_feat(post).detach()

		for p in self._label_params:
			p.requires_grad_(True)
		try:
			with torch.cuda.amp.autocast(self._use_amp):
				losses = {}
				for key in self._label_keys:
					loss = -self.heads[key](feat).log_prob(data[key])
					assert loss.shape == feat.shape[:2], (key, loss.shape)
					losses[key] = loss
				label_loss = sum(
					value * self._scales.get(key, 1.0)
					for key, value in losses.items()
				)
			metrics = self._label_opt(
				torch.mean(label_loss), self._label_params
			)
		finally:
			for p in self._label_params:
				p.requires_grad_(False)

		metrics.update(
			{f"{key}_loss": to_np(torch.mean(v)) for key, v in losses.items()}
		)
		return metrics

	
	
	

	# this function is called during both rollout and training
	def preprocess(self, obs):
		obs = {
			k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
			for k, v in obs.items()
		}

		for k in obs.keys():        
			if re.match(self._config.encoder["cnn_keys"], k):
				obs[k] = obs[k] / 255.0

		if "discount" in obs:
			obs["discount"] *= self._config.discount
			# (batch_size, batch_length) -> (batch_size, batch_length, 1)
			obs["discount"] = obs["discount"].unsqueeze(-1)
		# 'is_first' is necesarry to initialize hidden state at training
		assert "is_first" in obs
		# 'is_terminal' is necesarry to train cont_head
		assert "is_terminal" in obs
		obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)

		if "failure" in obs:
			obs["failure"] = obs["failure"].unsqueeze(-1)
		if "success" in obs:
			obs["success"] = obs["success"].unsqueeze(-1)
		for k in getattr(self._config, "label_heads", []):
			if k in obs:
				obs[k] = obs[k].unsqueeze(-1)

		return obs

	def video_pred(self, data, ensemble: OneStepPredictor| None = None):
		image_key = getattr(self._config, "video_pred_key", "front_cam")
		has_cnn_decoder = image_key in getattr(self.heads["decoder"], "cnn_shapes", {})
		if not has_cnn_decoder:
			# No pixel path for this key (pixel_recon off -> feature-only WM).
			return None

		def decode(feat):
			return self.heads["decoder"](feat)[image_key].mode()

		with torch.no_grad():
			data = self.preprocess(data)
			embed = self.encoder(data)

			obs_step = 5

			states, _ = self.dynamics.observe(
				embed[:6, :obs_step], data["action"][:6, :obs_step], data["is_first"][:6, :obs_step]
			)
			recon = decode(self.dynamics.get_feat(states))[:6]
			init = {k: v[:, -1] for k, v in states.items()}
			prior = self.dynamics.imagine_with_action(data["action"][:6, obs_step:], init)
			openl = decode(self.dynamics.get_feat(prior))
			truth = data[image_key][:6]

			# Clip when finished
			row, col = torch.where(data['is_first'][:6, obs_step:] == 1.)
			for i in range(row.size(0)):
				data['is_first'][row[i], obs_step+col[i]:] = 1.
				openl[row[i], col[i]:] = openl[row[i], col[i]-1]
				truth[row[i], obs_step+col[i]:] = truth[row[i], obs_step+col[i]-1]

			# observed image is given until 5 steps
			model = torch.cat([recon[:, :obs_step], openl], 1)
			error = (model - truth + 1.0) / 2.0

			video_pred = torch.cat([truth, model, error], 2)

			return video_pred.detach().cpu().numpy()
