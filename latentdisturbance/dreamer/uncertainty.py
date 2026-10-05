import torch
from torch import nn

from . import tools
from .ensemble.penn import EnsembleStochasticLinear


def clamp_preserve_gradients(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Clamps the values of the tensor into ``[lower, upper]`` but keeps the gradients."""
    return x + (x.clamp(min=lower, max=upper) - x).detach()

class OneStepPredictor(nn.Module):
    def __init__(self, config, world_model):
        super(OneStepPredictor, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            stoch = config.dyn_stoch * config.dyn_discrete
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.dyn_deter,
            "feat": config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]

        input_dim = feat_size + (config.num_actions if config.disag_action_cond else 0)

        self._networks = EnsembleStochasticLinear(in_features=input_dim, 
                                                 out_features=size,
                                                 hidden_features=input_dim,
                                                 ensemble_size=config.disag_models,
                                                 explore_var='jrd', 
                                                 residual=True)
        
        
        self.criterion = self.gaussian_nll_loss 
        
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "ensemble",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )
        self.config = config

    def gaussian_nll_loss(self, mu, target, var):
        # Custom Gaussian Negative Log Likelihood Loss
        loss = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
        return torch.mean(loss)
    
    def intrinsic_reward_penn(self, inputs):

        self._networks.eval()

        if len(inputs.shape) == 3:
            N, T, D = inputs.shape
            inputs = inputs.reshape(N * T, D)

            with torch.no_grad():
                ensemble_outputs = self._networks(inputs)
                div = ensemble_outputs[-1]
            
            div = div.view(N, T, -1)
        else:
            with torch.no_grad():
                ensemble_outputs = self._networks(inputs)
                div = ensemble_outputs[-1]

        if self._config.disag_log:
            div = torch.log(div)

        return div
    
    def train_ensemble_penn(self, inputs, targets):
        self._networks.train()
        with torch.cuda.amp.autocast(self._use_amp):
            if self._config.disag_offset:
                targets = targets[:, self._config.disag_offset :]
                inputs = inputs[:, : -self._config.disag_offset]

            targets = targets.detach()
            inputs = inputs.detach()
            
            train_loss = torch.FloatTensor([0]).cuda()
            N, T, D = inputs.shape
            inputs = inputs.reshape(N * T, D)

            for i in range(self.config.disag_models):                
                (mu, log_std) = self._networks.single_forward(
                    inputs, index=i)
                
                mu = mu.view(N, T, -1)
                log_std = log_std.reshape(N, T, -1)

                yhat_mu = mu
                var = torch.square(torch.exp(log_std))
                loss = self.gaussian_nll_loss(yhat_mu, targets, var)
                loss = loss.mean()
                self._expl_opt(loss, self._networks.parameters())
                
                train_loss += loss

        metrics = {"ensemble_loss": train_loss.item() / self.config.disag_models}

        with torch.no_grad():
            div = self.intrinsic_reward_penn(inputs)
        metrics["log_disagreement"] = div.cpu().numpy()

        return metrics
    
    def train_ensemble_penn_fixed(self, feats, actions, targets, is_first):
        self._networks.train()
        with torch.cuda.amp.autocast(self._use_amp):


            feats = feats[:, :-1]
            actions = actions[:, 1:]
            inputs = torch.concat([feats, actions], -1)
            targets = targets[:, 1:]

            valid_idx = torch.roll(is_first, shifts=-1, dims=1)[:, :-1] == 0.

            valid_inputs = inputs[valid_idx]
            valid_targets = targets[valid_idx]

            valid_inputs = valid_inputs.detach()
            valid_targets = valid_targets.detach()
            
            train_loss = torch.FloatTensor([0]).cuda()
            
            for i in range(self.config.disag_models):                
                (mu, log_std) = self._networks.single_forward(
                    valid_inputs, index=i)

                yhat_mu = mu.unsqueeze(0)
                var = torch.square(torch.exp(log_std.unsqueeze(0)))
                loss = self.gaussian_nll_loss(yhat_mu, valid_targets, var)
                loss = loss.mean()
                self._expl_opt(loss, self._networks.parameters())
                
                train_loss += loss

        metrics = {"ensemble_loss": train_loss.item() / self.config.disag_models}

        with torch.no_grad():
            div = self.intrinsic_reward_penn(valid_inputs).mean()
        metrics["log_disagreement"] = div.cpu().numpy()

        return metrics
    
    def get_disagreement_fixed(self, feats, actions, is_first):
        with torch.cuda.amp.autocast(self._use_amp):

            feats = feats[:, :-1]
            actions = actions[:, 1:]
            inputs = torch.concat([feats, actions], -1)

            valid_idx = torch.roll(is_first, shifts=-1, dims=1)[:, :-1] == 0.

            valid_inputs = inputs[valid_idx]
            valid_inputs = valid_inputs.detach()

        with torch.no_grad():
            div = self.intrinsic_reward_penn(valid_inputs)

        return div
    

from .unet1d import ConditionalUnet1D

def get_unet(input_dim):
    return ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=False
    )


class logpZO(torch.nn.Module):
    def __init__(self, config, device='cuda'):
        super(logpZO, self).__init__()
        if config.dyn_discrete:
            self.input_dim = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            self.input_dim = config.dyn_stoch + config.dyn_deter
        self.net = get_unet(self.input_dim).to(device)
        self.device= device

        self._use_amp = True if config.precision == 16 else False
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._density_opt = tools.Optimizer(
            "density",
            self.net.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )
        self.config = config

    def train_step(self, input: torch.Tensor):
        self.net.train()
        input = input.reshape(-1, 1, self.input_dim)
        x0, x1 = input, torch.randn_like(input).to(self.device)
        vtrue = x1 - x0
        cont_t = torch.rand(len(x1),).to(self.device)
        cont_t = cont_t.view(-1, *[1 for _ in range(len(input.shape)-1)])
        xnow = x0 + cont_t * vtrue
        time_scale = 100
        vhat = self.net(xnow, (cont_t.view(-1)*time_scale).long())
        
        loss = (vhat - vtrue).pow(2).mean()

        self._density_opt(loss, self.net.parameters())

        metrics = {"density_loss": loss.item()}

        return metrics
    
    def forward(self, input: torch.Tensor):

        self.net.eval()

        input = input.reshape(-1, 1, self.input_dim)
        timesteps = torch.zeros(input.shape[0], device=input.device)
        pred_v = self.net(input, timesteps)
        input = input + pred_v
        logpZO = input.reshape(len(input), -1).pow(2).sum(dim=-1)

        return logpZO


class logpZO_za(torch.nn.Module):
    """Flow-matching (rectified-flow) OOD score over (feat_t, a_{t+1}) pairs."""

    def __init__(self, config, device='cuda'):
        super(logpZO_za, self).__init__()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.input_dim = feat_size + config.num_actions
        self.net = get_unet(self.input_dim).to(device)
        self.device = device

        self._use_amp = True if config.precision == 16 else False
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._opt = tools.Optimizer(
            "uq_flow",
            self.net.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )
        self.config = config

    def _pair(self, feats, actions, is_first):
        """(feat_t, a_{t+1}) for valid transitions -> (N_valid, input_dim).

        Same pairing and is_first masking as OneStepPredictor's
        train_ensemble_penn_fixed: action[t] is the prev-action convention, so
        the action executed FROM step t is actions[:, t+1]."""
        inputs = torch.concat([feats[:, :-1], actions[:, 1:]], -1)
        valid_idx = torch.roll(is_first, shifts=-1, dims=1)[:, :-1] == 0.
        return inputs[valid_idx].detach()

    def train_step(self, feats, actions, is_first):
        self.net.train()
        x0 = self._pair(feats, actions, is_first).reshape(-1, 1, self.input_dim)
        x1 = torch.randn_like(x0).to(self.device)
        vtrue = x1 - x0
        cont_t = torch.rand(len(x1),).to(self.device)
        cont_t = cont_t.view(-1, *[1 for _ in range(len(x0.shape) - 1)])
        xnow = x0 + cont_t * vtrue
        time_scale = 100
        vhat = self.net(xnow, (cont_t.view(-1) * time_scale).long())

        loss = (vhat - vtrue).pow(2).mean()

        self._opt(loss, self.net.parameters())

        return {"uq_flow_loss": loss.item()}

    def score(self, feats, actions, is_first):
        """Per-valid-transition OOD score (N_valid,); higher = more OOD."""
        self.net.eval()
        with torch.no_grad():
            x = self._pair(feats, actions, is_first).reshape(-1, 1, self.input_dim)
            timesteps = torch.zeros(x.shape[0], device=x.device)
            pred_v = self.net(x, timesteps)
            x = x + pred_v
            return x.reshape(len(x), -1).pow(2).sum(dim=-1)

    def score_flat(self, x):
        """OOD score for already-paired flat inputs (N, feat+action) -> (N,).

        For env-time stepping where (feat_t, action executed from t) is known
        directly, so `score`'s sequence pairing / is_first masking is not
        needed. Same mechanism as `score` / logpZO.forward: one Euler step of
        the learned velocity field, squared norm; higher = more OOD."""
        self.net.eval()
        with torch.no_grad():
            x = x.reshape(-1, 1, self.input_dim)
            timesteps = torch.zeros(x.shape[0], device=x.device)
            pred_v = self.net(x, timesteps)
            x = x + pred_v
            return x.reshape(len(x), -1).pow(2).sum(dim=-1)