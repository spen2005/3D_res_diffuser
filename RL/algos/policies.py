import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from functools import partial
from gymnasium import spaces

from stable_baselines3.common.distributions import (
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
        log_std_init = 0.0,
        optimizer_class = torch.optim.Adam,
        optimizer_kwargs = None,
        net_arch = dict(pi=[64, 64], vf=[64, 64]),
        activation_fn = nn.Tanh,
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
        self.features_extractor = features_extractor_class(
            self.observation_space, **features_extractor_kwargs
        )

        self.features_dim = self.features_extractor.features_dim
        self.action_dist = make_proba_distribution(action_space)

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
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim_pi, log_std_init=self.log_std_init
        )

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
        }

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
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

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
        return dict(
            observation_space = self.observation_space,
            action_space = self.action_space,
            normalize_images = self.normalize_images,
            net_arch = self.net_arch,
            log_std_init=self.log_std_init,
            lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs,
        )  