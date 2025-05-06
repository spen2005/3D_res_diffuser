import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from datetime import datetime

@hydra.main(version_base="1.1", config_name="custom_config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    import logging
    import os
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    # Naming the run
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.run_name}_{time_str}"

    # Ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # Creating a new function to return a pushT environment
    from custom_envs.pusht_single_env import PushTEnv
    from custom_envs.customenv_utils import CustomRayVecEnv, PushTAlgoObserver

    def create_pusht_env(**kwargs):
        env = PushTEnv()
        return env

    # Register the custom environment
    env_configurations.register('pushT', {
        'vecenv_type': 'CUSTOMRAY',
        'env_creator': lambda **kwargs: create_pusht_env(**kwargs),
    })

    # Register the vecenv
    vecenv.register('CUSTOMRAY', lambda config_name, num_actors, **kwargs: CustomRayVecEnv(env_configurations.configurations, config_name, num_actors, **kwargs))

    # Convert the Hydra config to a dictionary
    rlg_config_dict = OmegaConf.to_container(cfg.train, resolve=True)  # Using OmegaConf directly for conversion

    # Build an rl_games runner
    def build_runner():
        runner = Runner(algo_observer=PushTAlgoObserver())
        return runner

    # Create runner and set the settings
    runner = build_runner()
    runner.load(rlg_config_dict)
    runner.reset()

    # Run either training or playing
    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
    })

if __name__ == "__main__":
    launch_rlg_hydra()
