import sys
import warnings
import time
import random
from collections import deque
from typing import Any, ClassVar, Optional, TypeVar, Union, Dict, List, Tuple

import numpy as np
import torch
import gymnasium as gym
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from stable_baselines3.common.utils import check_for_correct_spaces, explained_variance, get_schedule_fn, configure_logger, obs_as_tensor, safe_mean, update_learning_rate, configure_logger, set_random_seed, get_system_info
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from src.algos.policies import ActorCriticPolicy
SelfPPO = TypeVar("SelfPPO", bound="MultiTaskPPO")


class MultiTaskPPO:
    
    def __init__(
            self,
            policy,
            env,
            learning_rate: float = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: float = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            rollout_buffer_class = None,
            rollout_buffer_kwargs = None,
            target_kl: float = None,
            stats_window_size: int = 100,
            tensorboard_log: str = None,
            policy_kwargs = None,
            verbose: int = 0,
            seed: int = None,
            device = "cuda",
            _init_setup_model: bool = True,
            monitor_wrapper: bool = True,
    ):
        
        self.device = torch.device(device)
        self.policy_class = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.verbose = verbose
        self.action_noise = None
        self.start_time = 0.0
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self._last_obs = None  
        self._last_episode_starts = None
        # When using VecNormalize:
        self._last_original_obs = None
        self._episode_num = 0
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1.0
        # Buffers for logging
        self._stats_window_size = stats_window_size
        self.ep_info_buffers = None  
        self.ep_success_buffer = None 
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        # Whether the user passed a custom logger or not
        self._custom_logger = False
        self.env = None
        self._vec_normalize_env = None

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = env.n_envs
        self.env = env

        if isinstance(self.action_space, spaces.Box):
            assert np.all(
                np.isfinite(np.array([self.action_space.low, self.action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

        self.n_steps = n_steps
        self.n_tasks = env.n_tasks
        self.envs_idx_list = env.env_ids_list
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if normalize_advantage:
            assert (batch_size > 1), "`batch_size` must be greater than 1."

        if self.env is not None:
            for i in range(self.n_tasks):
                # Check that `n_steps * n_tasks > 1` to avoid NaN
                # when doing advantage normalization
                buffer_size = len(self.envs_idx_list[i]) * self.n_steps
                assert buffer_size > 1 or (
                    not normalize_advantage
                ), f"`n_steps * n_tasks` must be greater than 1. Currently n_steps={self.n_steps} and n_tasks={self.env.n_tasks}"
                # Check that the rollout buffer size is a multiple of the mini-batch size
                untruncated_batches = buffer_size // batch_size
                if buffer_size % batch_size > 0:
                    warnings.warn(
                        f"You have specified a mini-batch size of {batch_size},"
                        f" but because the {i}th `RolloutBuffer` is of size `n_steps * n_tasks = {buffer_size}`,"
                        f" after every {untruncated_batches} untruncated mini-batches,"
                        f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                        f"We recommend using a `batch_size` that is a factor of `n_steps * n_tasks`.\n"
                        f"Info: (n_steps={self.n_steps} and n_tasks={self.env.n_tasks})"
                    )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """ 
        # Switch to train mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(True)
        self.policy.train(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # collected_rollout_buffer = [rollout_data for rollout_buffer in self.rollout_buffers for rollout_data in rollout_buffer.get(self.batch_size)]
        # random.shuffle(collected_rollout_buffer)
        # normalize per-task advantage
        for rollout_buffer in self.rollout_buffers:
            rollout_buffer.advantages = (rollout_buffer.advantages - rollout_buffer.advantages.mean()) / (rollout_buffer.advantages.std() + 1e-8)


        # print(self.rollout_buffers[0].observations.keys())
        keys = self.rollout_buffers[0].observations.keys()
        self.total_rollout_buffer.observations = {key: np.concatenate([rollout_buffer.observations[key] for rollout_buffer in self.rollout_buffers], axis=1) for key in keys}
        self.total_rollout_buffer.actions = np.concatenate([rollout_buffer.actions for rollout_buffer in self.rollout_buffers], axis=1)
        self.total_rollout_buffer.values = np.concatenate([rollout_buffer.values for rollout_buffer in self.rollout_buffers], axis=1)
        self.total_rollout_buffer.log_probs = np.concatenate([rollout_buffer.log_probs for rollout_buffer in self.rollout_buffers], axis=1)
        self.total_rollout_buffer.advantages = np.concatenate([rollout_buffer.advantages for rollout_buffer in self.rollout_buffers], axis=1)
        self.total_rollout_buffer.returns = np.concatenate([rollout_buffer.returns for rollout_buffer in self.rollout_buffers], axis=1)
        self.total_rollout_buffer.full = True

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.total_rollout_buffer.get(self.batch_size):
                # Convert to pytorch tensor or to TensorDict
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Compute explained variance
        for i, rollout_buffer in enumerate(self.rollout_buffers):
            explained_var = explained_variance(rollout_buffer.values.flatten(), rollout_buffer.returns.flatten())
            self.logger.record(f"train/explained_variance-{i}", explained_var)

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
    
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())
        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffers, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffers is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()
        return self
    
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer


        self.total_rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.rollout_buffers = []

        for i in range(self.n_tasks):
            rollout_buffer = self.rollout_buffer_class(
                self.n_steps,
                self.observation_space,  # type: ignore[arg-type]
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=len(self.envs_idx_list[i]),
                **self.rollout_buffer_kwargs,
            )
            self.rollout_buffers.append(rollout_buffer)

        self.policy = ActorCriticPolicy(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
   
    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        rollout_buffers: List[RolloutBuffer],
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(False)
        self.policy.train(False)

        n_steps = 0

        self.total_rollout_buffer.reset()
        for rollout_buffer in rollout_buffers:
            # Reset the buffer
            rollout_buffer.reset()

        callback.on_rollout_start()


        while n_steps < n_rollout_steps:
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            # actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            clipped_actions = torch.clip(actions, torch.tensor(self.action_space.low, device=self.device), torch.tensor(self.action_space.high, device=self.device))

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.n_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            for i, envs_idx in enumerate(self.envs_idx_list):
                selected_infos = [infos[idx] for idx in envs_idx]
                selected_dones = [dones[idx] for idx in envs_idx]
                # Update the info buffer for each env
                self._update_info_buffer(selected_infos, selected_dones, i)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
            
            for i, rollout_buffer in enumerate(rollout_buffers):
                last_obs_cpu = {key: value[self.envs_idx_list[i]].cpu().numpy() for key, value in self._last_obs.items()}

                rollout_buffer.add(
                    # self._last_obs,  # type: ignore[arg-type]
                    last_obs_cpu,
                    actions[self.envs_idx_list[i]].cpu(),
                    rewards[self.envs_idx_list[i]].cpu(),
                    self._last_episode_starts[self.envs_idx_list[i]],  # type: ignore[arg-type]
                    values[self.envs_idx_list[i]].cpu(),
                    log_probs[self.envs_idx_list[i]].cpu(),   
                )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        for i, rollout_buffer in enumerate(rollout_buffers):
            # Compute the last value for the rollout buffer
            # and compute the returns and advantages
            # using the last value of the rollout buffer
            rollout_buffer.compute_returns_and_advantage(last_values=values[self.envs_idx_list[i]], dones=dones[self.envs_idx_list[i]])

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True
    
    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a List of callbacks into a callback
        if isinstance(callback, List):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ):
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffers is None or reset_num_timesteps:
            self.ep_info_buffers = []
            for i in range(self.n_tasks):
                # Initialize buffers if they don't exist, or reinitialize if resetting counters
                ep_info_buffer = deque(maxlen=self._stats_window_size)
                self.ep_info_buffers.append(ep_info_buffer)
                self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            assert self.env is not None
            self._last_obs = self.env.reset()  # type: ignore[assignment]
            self._last_episode_starts = np.ones((self.env.n_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            # self._logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)
            self.logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback
    
    def _update_learning_rate(self, optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a List of optimizers.
        """
        # Log the current learning rate
        self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, List):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)
    
    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == torch.device("cuda").type)
        self.action_space.seed(seed)
        # self.env is always a VecEnv
        if self.env is not None:
            self.env.seed(seed)

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None, envs_id: int = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        assert self.ep_info_buffers is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffers[envs_id].extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffers is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")

        total_ep_info_buffer = [ep_info for ep_info_buffer in self.ep_info_buffers for ep_info in ep_info_buffer]
        if len(total_ep_info_buffer) > 0 and len(total_ep_info_buffer[0]) > 0:
            # Compute and record the metrics
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in total_ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in total_ep_info_buffer]))
            self.logger.record("rollout/ep_goal_rew", safe_mean([ep_info["g"] for ep_info in total_ep_info_buffer]))
            self.logger.record("rollout/success_rate", safe_mean([ep_info["s"] for ep_info in total_ep_info_buffer]))

        for i, ep_info_buffer in enumerate(self.ep_info_buffers):
            if len(ep_info_buffer) > 0 and len(ep_info_buffer[0]) > 0:
                # Compute and record the metrics
                self.logger.record(f"rollout/ep_rew_mean-{i}", safe_mean([ep_info["r"] for ep_info in ep_info_buffer]))
                self.logger.record(f"rollout/ep_len_mean-{i}", safe_mean([ep_info["l"] for ep_info in ep_info_buffer]))
                self.logger.record(f"rollout/ep_goal_rew-{i}", safe_mean([ep_info["g"] for ep_info in ep_info_buffer]))
                self.logger.record(f"rollout/success_rate-{i}", safe_mean([ep_info["s"] for ep_info in ep_info_buffer]))

        # raise ValueError("Error: the environment should not return an empty info buffer")
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.

        Names can point to specific variables under classes, e.g.
        "policy.optimizer" would point to ``optimizer`` object of ``self.policy``
        if this object.

        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and List of other Torch variables to store with ``th.save``.
        """
        state_dicts = ["policy"]

        return state_dicts, []
    
    def set_parameters(
        self,
        load_path_or_dict,
        exact_match: bool = True,
        device = "cuda",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = {}
        if isinstance(load_path_or_dict, Dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device, load_data=False)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, torch.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-Dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-Dict is missing keys.
                #
                # Solution: Just load the state-Dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])  # type: ignore[arg-type]
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )
    
    def load(  # noqa: C901
        cls,
        path,
        env = None,
        device = "cuda",
        custom_objects = None,
        print_system_info = False,
        force_reset = True,
        **kwargs,
    ):
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            saved_net_arch = data["policy_kwargs"].get("net_arch")
            if saved_net_arch and isinstance(saved_net_arch, List) and isinstance(saved_net_arch[0], Dict):
                data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load policies saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a A2C/PPO model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        except ValueError as e:
            # Patch to load DQN policies saved using SB3 < 2.4.0
            # The target network params are no longer in the optimizer
            # See https://github.com/DLR-RM/stable-baselines3/pull/1963
            saved_optim_params = params["policy.optimizer"]["param_groups"][0]["params"]  # type: ignore[index]
            n_params_saved = len(saved_optim_params)
            n_params = len(model.policy.optimizer.param_groups[0]["params"])
            if n_params_saved == 2 * n_params:
                # Truncate to include only online network params
                params["policy.optimizer"]["param_groups"][0]["params"] = saved_optim_params[:n_params]  # type: ignore[index]

                model.set_parameters(params, exact_match=True, device=device)
                warnings.warn(
                    "You are probably loading a DQN model saved with SB3 < 2.4.0, "
                    "we truncated the optimizer state so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/pull/1963 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        return model
    
    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state Dict
            params[name] = attr.state_dict()
        return params


    def save(
        self,
        path,
        exclude = ['_last_obs','_last_episode_starts','_last_original_obs','env','_vec_normalize_env','rollout_buffer','ep_info_buffer','logger','_custom_logger'],
        include = ['device'],
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter List so we don't mutate the original Dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build Dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build Dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)