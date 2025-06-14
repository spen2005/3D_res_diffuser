import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from functools import partial
from gymnasium import spaces

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

from stable_baselines3.common.torch_layers import (
    MlpExtractor,
)

class ActorCriticPolicy(nn.Module):

    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_sde = False,
        log_std_init = 0.0,
        full_std = True,
        use_expln = False,
        squash_output = False,
        optimizer_class = torch.optim.Adam,
        optimizer_kwargs = None,
        net_arch = dict(pi=[64, 64], vf=[64, 64]),
        activation_fn = nn.Tanh,
        ortho_init = True,
        features_extractor_class = None,
        features_extractor_kwargs = None,
        features_extractor = None,
        normalize_images = True,
        device = "cuda",
    ):
        super().__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        if features_extractor is None:
            features_extractor_kwargs = {}

        self.device = torch.device(device)

        self.observation_space = observation_space        
        self.action_space = action_space
        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.log_std_init = log_std_init

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.features_extractor = features_extractor_class(
            self.observation_space, **features_extractor_kwargs
        )
        dist_kwargs = None

        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        self.features_dim = self.features_extractor.features_dim
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def predict(
        self,
        observation,
        state,
        episode_start,
        deterministic=False,
    ):
        self.train(False)
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            actions = self._predict(obs_tensor, deterministic)

        actions.reshape((-1, *self.action_space.shape))
        actions = torch.clip(actions, self.action_space.low, self.action_space.high)
        return actions, state
    
    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """(float) Useful for pickling policy."""
        del progress_remaining
        return 0.0

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0) 

    def _predict(self, observation, deterministic = False):
        return self.get_distribution(observation).get_actions(deterministic=deterministic)
    
        
    def evaluate_actions(self, obs, actions):
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy
    
    def get_distribution(self, obs):
        features = self.feature_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)
    
    def predict_values(self, obs):
        features = self.features_extractor(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)
    
    def _build(self, lr_schedule):
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch = self.net_arch,
            activation_fn = self.activation_fn,
            device = self.device,
        )

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
            self.log_std = log_std.detach().clone().to(self.device)  # detach to avoid gradients on log_std
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            # if not self.share_features_extractor:
            #     # Note(antonin): this is to keep SB3 results
            #     # consistent, see GH#1148
            #     del module_gains[self.features_extractor]
            #     module_gains[self.pi_features_extractor] = np.sqrt(2)
            #     module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def forward(self, obs, deterministic=False):
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        # put the distribution on device
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        torch.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    def load(cls, path: str, device):
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = torch.device(device)
        # Note(antonin): we cannot use `weights_only=True` here because we need to allow
        # gymnasium imports for the policy to be loaded successfully
        saved_variables = torch.load(path, map_location=device, weights_only=False)

        # Create policy object
        model = cls(**saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def _get_constructor_parameters(self):
        data = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data