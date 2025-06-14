import argparse
import os
import yaml
import time
import random
import importlib
from typing import Optional
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import torch
import wandb
import genesis as gs
from gymnasium import spaces
from wandb.integration.sb3 import WandbCallback

# Assume these are moved to src/ and imports reflect that
# Example: from src.envs.multitask_env import MultiTaskVecEnv
from src.envs.multitask_env import MultiTaskVecEnv
from src.algos.multitask_ppo import MultiTaskPPO # Adjust import path
from src.algos.ppo import PPO # Adjust import path
# Import task classes (or use dynamic loading below)
# from src.DEXMAN_100.BottleLifting.BottleLiftingTask1 import BottleLiftingTask1 # Adjust import path
# from src.DEXMAN_100.BottleLifting.BottleLiftingTask2 import BottleLiftingTask2 # Adjust import path
# from src.DEXMAN_100.BallPicking.BallPickingTask1 import BallPickingTask1 # Adjust import path
# from src.DEXMAN_100.BallPicking.BallPickingTask2 import BallPickingTask2 # Adjust import path
# from src.DEXMAN_100.oakink_0b3d1.BowlLiftingTask1 import BowlLiftingTask1 # Adjust import path
# from src.DEXMAN_100.oakink_0b3d1.CupPickingTask1 import CupPickingTask1 # Adjust import path
# from src.DEXMAN_100.oakink_4c966.JarTwistingTask1 import JarTwistingTask1 # Adjust import path
# from src.DEXMAN_100.oakink_5ad66.ScoopingTask1 import ScoopingTask1
from src.DEXMAN_100.ManipTask import ManipTask

# Import feature extractor
from src.models.cnn_feature_extractor import CNN_Extractor # Adjust import path

# Use config loading utilities from src/utils/config.py
from src.utils.config import load_config, merge_configs # Make sure this file exists

# --- Task Registry ---
TASK_REGISTRY = {
    # "BottleLiftingTask1": BottleLiftingTask1,
    # "BottleLiftingTask2": BottleLiftingTask2,
    # "BallPickingTask1": BallPickingTask1,
    # "BallPickingTask2": BallPickingTask2,
    # "BowlLiftingTask1": BowlLiftingTask1,
    # "CupPickingTask1": CupPickingTask1,
    # "JarTwistingTask1": JarTwistingTask1,
    # "ScoopingTask1": ScoopingTask1,
    "ManipTask": ManipTask,
}

# --- Feature Extractor Registry ---
FEATURE_EXTRACTOR_REGISTRY = {
    "CNN_Extractor": CNN_Extractor,
}

# --- Algorithm Registry ---
ALGORITHM_REGISTRY = {
    "MultiTaskPPO": MultiTaskPPO,
    "PPO": PPO,
}

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL agent for dexterous manipulation tasks.")
    parser.add_argument('--config', type=str, default='configs/multitask_ppo_default.yaml',
                        help='Path to the main YAML configuration file.')
    # Add arguments to override specific config values if needed
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Override list of tasks defined in config file (e.g., --tasks ball_picking bottle_lifting)')
    parser.add_argument('--task-configs-path', nargs='+', default=None,
                        help='Override list of task config paths (e.g., --task-configs-path configs/task1.yaml configs/task2.yaml)')
    parser.add_argument('--n-envs', type=int, default=None, help='Override total number of parallel environments.')
    parser.add_argument('--total-timesteps', type=int, default=None, help='Override total training timesteps.')
    parser.add_argument('--seed', type=int, default=42, help='Override random seed.')
    parser.add_argument('--logdir', type=str, default=None, help='Override base directory for logs.')
    parser.add_argument('--run-name', type=str, default=None, help='Specify a custom run name (overrides auto-generation).')
    parser.add_argument('--show-viewer', action='store_true', help='Show simulation viewer (overrides config).')
    parser.add_argument('--disable-wandb', action='store_true',
                        help='Disable Weights & Biases logging, overriding config settings.')


    args = parser.parse_args()
    return args

def load_and_prepare_config(args: argparse.Namespace) -> dict:
    """Loads base config, task configs, and merges args."""
    print(f"[*] Loading base configuration from: {args.config}")
    cfg = load_config(args.config)

    # --- Override config with command-line arguments ---
    if args.tasks is not None:
        cfg['tasks'] = args.tasks
        print(f"[*] Overriding tasks with command line: {cfg['tasks']}")
    if args.task_configs_path is not None:
        cfg['task_configs_path'] = args.task_configs_path
        print(f"[*] Overriding tasks config paths with command line: {cfg['task_configs_path']}")
    if args.n_envs is not None:
        cfg['environment']['n_envs_total'] = args.n_envs
        print(f"[*] Overriding n_envs for evaluation: {args.n_envs}")
        # Warning if envs_per_task was set, as it might become inconsistent
        if cfg['environment'].get('envs_per_task') is not None:
             original_total = sum(cfg['environment']['envs_per_task'])
             if original_total != args.n_envs:
                  warnings.warn(f"Overriding n_envs_total ({args.n_envs}) while envs_per_task was specified in config (sum={original_total}). Distribution might change.")
             # Clear envs_per_task if n_envs is overridden, let MultiTaskVecEnv distribute
             cfg['environment']['envs_per_task'] = None
    if args.total_timesteps is not None:
        cfg['total_timesteps'] = args.total_timesteps
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.logdir is not None:
        cfg['logging']['logdir'] = args.logdir
    if args.show_viewer: # Note: store_true means presence sets it true
         cfg['environment']['show_viewer'] = True
    if args.disable_wandb:
        print("[*] Disabling WandB logging via command-line argument (--disable-wandb).")
        if 'logging' not in cfg:
            cfg['logging'] = {} # Create logging section if it doesn't exist
        # Set wandb config to an empty dictionary, which setup_wandb treats as disabled
        cfg['logging']['wandb'] = {}

    # --- Load individual task configurations ---
    task_details = {}
    if 'tasks' not in cfg or not cfg['tasks']:
         raise ValueError("No tasks specified in configuration or command line arguments.")

    print(f"[*] Loading configurations for tasks: {cfg['tasks']}")
    for i, task_name in enumerate(cfg['tasks']):
        print("[*] Loading task config for:", task_name)
        task_config_path = cfg['task_configs_path'][i]
        task_cfg = load_config(task_config_path)
        if not task_cfg:
            raise FileNotFoundError(f"Could not load task configuration: {task_config_path}")
        # Resolve relative paths in task params if necessary (e.g., relative to project root)
        # Example: Assuming paths are relative to project root
        for key, value in task_cfg.get('params', {}).items():
             if 'path' in key and isinstance(value, str) and value.startswith('./'):
                  # This assumes your script is run from the project root
                  task_cfg['params'][key] = os.path.abspath(value)
        task_details[task_name] = task_cfg

    cfg['resolved_task_configs'] = task_details # Store loaded/resolved task configs

    # Add the user-provided run name if specified
    if args.run_name is not None:
         cfg['logging']['user_specified_run_name'] = args.run_name

    return cfg


@contextmanager
def setup_wandb(cfg: dict, run_dir: str):
    """Initializes WandB and ensures finish is called."""
    wandb_cfg = cfg.get('logging', {}).get('wandb', {})
    if not wandb_cfg:
        print("[*] WandB logging is disabled in config.")
        yield None # Yield None if wandb is not configured
        return

    try:
        print("[*] Initializing WandB...")
        run = wandb.init(
            project=wandb_cfg.get('project', 'default_project'),
            entity=wandb_cfg.get('entity'), # Uses WANDB_ENTITY env var or default if None
            sync_tensorboard=wandb_cfg.get('sync_tensorboard', False),
            config=cfg, # Log the entire config
            name=os.path.basename(run_dir), # Use the generated run directory name for WandB run name
            dir=cfg['logging']['logdir'], # Set base log directory for wandb files
            save_code=True, # Save main script to W&B
            # reinit=True # Add if running multiple inits in one script (not typical here)
        )
        print(f"[*] WandB initialized. Run name: {run.name}, URL: {run.url}")
        yield run # Provide the run object to the main logic if needed
    finally:
        if wandb.run is not None:
            print("[*] Finishing WandB run...")
            wandb.finish()
            print("[*] WandB run finished.")


def setup_logging_and_saving(cfg: dict):
    """Sets up directories for logging and model saving."""
    log_cfg = cfg.get('logging', {})
    base_logdir = log_cfg.get('logdir', 'training_runs') # Default base dir

    # Generate a run name
    if 'user_specified_run_name' in log_cfg:
         run_name = log_cfg['user_specified_run_name']
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_names_str = "_".join(cfg.get('tasks', ['unknown']))
        algo_name = cfg.get('algorithm', {}).get('name', 'algo')
        prefix = log_cfg.get('run_name_prefix', 'run')
        run_name = f"{prefix}_{algo_name}_{task_names_str}_{timestamp}"

    run_dir = os.path.join(base_logdir, run_name)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    tensorboard_log_path = os.path.join(run_dir, 'tensorboard_logs')

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_log_path, exist_ok=True)


    print(f"[*] Run Name: {run_name}")
    print(f"[*] Base Log Directory: {base_logdir}")
    print(f"[*] Run Directory: {run_dir}")

    # Save the final effective config to the run directory
    config_save_path = os.path.join(run_dir, 'effective_config.yaml')
    try:
        with open(config_save_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"[*] Effective configuration saved to {config_save_path}")
    except Exception as e:
        print(f"ERROR saving effective config: {e}")


    # Return paths and the WandB context manager
    return run_dir, checkpoints_dir, tensorboard_log_path, setup_wandb(cfg, run_dir)


def set_random_seeds(seed: Optional[int]):
    """Sets random seeds for reproducibility."""
    if seed is None:
        print("INFO: No seed provided, using random seeds.")
        seed = random.randint(0, 2**32 - 1) # Generate a seed if none provided

    print(f"[*] Setting random seeds to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        # Setting deterministic options can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def initialize_simulator(cfg: dict):
    """Initializes the Genesis simulator."""
    sim_cfg = cfg.get('simulator', {})
    print("[*] Initializing Genesis Simulator for Training...")
    try:
        gs.init(
            seed=cfg.get('seed', 42),
            backend=getattr(gs, sim_cfg.get('backend', 'gpu'), gs.gpu),
            # logging_level=sim_cfg.get('logging_level', None),
            logging_level='warning',
            precision='32',
        )
        # Optional Context setup
        context_opts = sim_cfg.get('context_options')
        if context_opts:
            print("[*] Creating Genesis Context...")
            gs.Context(**context_opts)
        print("[*] Genesis initialized.")
    except Exception as e:
        print(f"ERROR initializing Genesis: {e}")
        raise # Re-raise critical error

def setup_environment(cfg: dict) -> MultiTaskVecEnv:
    """Creates the multi-task vectorized environment."""
    env_cfg = cfg.get('environment', {})
    task_configs_dict = cfg['resolved_task_configs']
    print(f"[*] Task configs loaded: {task_configs_dict}")
    task_names_to_run = cfg['tasks']

    print(f"[*] Setting up environment for tasks: {task_names_to_run}")

    # Prepare lists for MultiTaskVecEnv constructor
    task_classes = []
    task_configs_list = []

    for i, task_name in enumerate(task_names_to_run):
        task_cfg = task_configs_dict[task_name]
        class_name = task_cfg.get('class_name')
        if not class_name or class_name not in TASK_REGISTRY:
            raise ValueError(f"Task class '{class_name}' not found in TASK_REGISTRY for task '{task_name}'.")

        task_classes.append(TASK_REGISTRY[class_name])
        # Pass only the 'params' sub-dict and maybe task_id to the actual task class constructor
        config_for_env = task_cfg.get('params', {})
        config_for_env['task_id'] = task_cfg.get('task_id', -1) # Pass task_id if present
        task_configs_list.append(config_for_env)


    # --- Define Observation and Action Spaces ---
    # Using parameters from config where possible
    obs_cfg = env_cfg.get('observation_space', {})
    act_cfg = env_cfg.get('action_space', {})

    observation_space = spaces.Dict({
        'image': spaces.Box(
            low=0, high=255,
            shape=tuple(obs_cfg.get('image_shape', [64, 64, 3])), # Default shape
            dtype=np.uint8
        ),
        'proprioception': spaces.Box(
            low=-np.inf, high=np.inf,
            shape=tuple(obs_cfg.get('proprioception_shape', [104])), # Default shape
            dtype=np.float32
        )
    })
    print(f"[*] Observation Space: {observation_space}")

    # Use the *maximum* limit found across tasks for defining the space bounds? Or average?
    # Let's use max for safety, assuming the policy learns to scale down.
    scale_coeffs = env_cfg.get('action_scale_coeffs', {'xyz': 800.0, 'joint': 10, 'xyz_limit': 0.005, 'joint_limit': 0.4}) # Default coeffs
    xyz_limit = scale_coeffs['xyz_limit']
    joint_limit = scale_coeffs['joint_limit']
    action_scale = np.concatenate([
        np.array([scale_coeffs['xyz']] * 6),
        np.array([scale_coeffs['joint']] * 44)
    ])

    low = np.concatenate([np.array([-xyz_limit]*6), np.array([-joint_limit]*44)]) * action_scale
    high = np.concatenate([np.array([xyz_limit]*6), np.array([joint_limit]*44)]) * action_scale
    action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    print(f"[*] Action Space: {action_space}")
    print(f"[*] Derived Action Scale: xyz_coeff={scale_coeffs['xyz']}, joint_coeff={scale_coeffs['joint']}")


    # --- Create Environment Instance ---
    print(f"[*] Instantiating MultiTaskVecEnv with {env_cfg.get('n_envs_total')} envs...")
    env = MultiTaskVecEnv(
        task_configs=task_configs_list, # List of param dicts
        task_classes=task_classes,     # List of class objects
        n_envs=env_cfg.get('n_envs_total'),
        observation_space=observation_space, # Defined above
        action_space=action_space,         # Defined above
        action_scale=action_scale,           # Derived scale factor
        xyz_limit=xyz_limit,               # Derived limit
        joint_limit=joint_limit,           # Derived limit
        envs_per_task=env_cfg.get('envs_per_task'), # e.g., None or [512, 512]
        show_viewer=env_cfg.get('show_viewer', False),
        cam_pos=tuple(env_cfg.get('cam_pos', [0, 0.3, 1.6])), # Default camera position
        cam_lookat=tuple(env_cfg.get('cam_lookat', [0, 0.47, 0.7])), # Default camera lookat
        robot_config=load_config(env_cfg.get('robot_config', 'configs/humanoids/g1_shadow.yaml')), # Load robot config
    )
    print("[*] Environment created.")
    return env


def setup_algorithm(cfg: dict, env: MultiTaskVecEnv, tb_log_path: str):
    """Initializes the RL algorithm."""
    algo_cfg = cfg.get('algorithm', {})
    policy_cfg = cfg.get('policy', {})
    algo_name = algo_cfg.get('name', 'MultiTaskPPO')

    if algo_name not in ALGORITHM_REGISTRY:
         raise ValueError(f"Algorithm '{algo_name}' not found in ALGORITHM_REGISTRY.")
    AlgoClass = ALGORITHM_REGISTRY[algo_name]
    print(f"[*] Initializing Algorithm: {algo_name}")

    # --- Policy Kwargs ---
    policy_kwargs = {}
    feat_ext_name = policy_cfg.get('features_extractor_class')
    if feat_ext_name:
        if feat_ext_name not in FEATURE_EXTRACTOR_REGISTRY:
             raise ValueError(f"Feature extractor '{feat_ext_name}' not found in FEATURE_EXTRACTOR_REGISTRY.")
        policy_kwargs["features_extractor_class"] = FEATURE_EXTRACTOR_REGISTRY[feat_ext_name]
        policy_kwargs["features_extractor_kwargs"] = policy_cfg.get("features_extractor_kwargs", {})

    net_arch = policy_cfg.get('net_arch')
    if net_arch:
        # SB3 expects list of dicts or just list depending on structure. Adapt as needed.
        # Assuming structure like dict(pi=[64, 64], vf=[64, 64]) as in original script
        policy_kwargs["net_arch"] = net_arch # Pass directly if format matches expected

    # --- Algorithm Instantiation ---
    # Filter algo_cfg to only include valid parameters for the constructor
    valid_algo_params = {k: v for k, v in algo_cfg.items() if k != 'name'} # Exclude 'name'

    model = AlgoClass(
        policy=policy_cfg.get('name', "MlpPolicy"), # Default policy name if needed by SB3 style
        env=env,
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        tensorboard_log=tb_log_path,
        seed=cfg.get('seed'),
        **valid_algo_params # Pass other params like learning_rate, n_steps, batch_size etc.
    )
    print(f"[*] Algorithm {algo_name} initialized.")
    return model

def train_agent(agent, cfg: dict, run_dir: str, checkpoints_dir: str, wandb_context):
    """Runs the training loop and saves the final model."""
    log_cfg = cfg.get('logging', {})
    total_timesteps = cfg.get('total_timesteps', 1_000_000) # Default timesteps
    save_freq = log_cfg.get('wandb', {}).get('save_freq', 100_000) # Default save freq
    final_model_name = f"final_model_{total_timesteps}steps"

    print(f"[*] Starting training for {total_timesteps} timesteps...")
    start_time = time.time()

    # Setup WandbCallback if WandB is enabled
    callback = None
    if wandb.run is not None: # Check if wandb was initialized successfully
        callback = WandbCallback(
            model_save_path=checkpoints_dir, # Save checkpoints within the run's checkpoint dir
            model_save_freq=save_freq,
            gradient_save_freq=0, # Disable gradient saving unless needed
            verbose=1
        )
        print(f"[*] Using WandbCallback (Save Freq: {save_freq})")


    # --- Training ---
    try:
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=1 # Default SB3 log interval, adjust if needed
            # Add reset_num_timesteps=False if resuming training later
        )
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        # Optionally save model even if training errored out
        # emergency_save_path = os.path.join(run_dir, "model_on_error.zip")
        # print(f"Attempting to save model to {emergency_save_path}...")
        # agent.save(emergency_save_path)

    end_time = time.time()
    print(f"[*] Training finished in {(end_time - start_time)/3600:.2f} hours.")

    # --- Final Model Save ---
    final_save_path = os.path.join(run_dir, final_model_name)
    print(f"[*] Saving final model to: {final_save_path}")
    agent.save(final_save_path)
    print("[*] Final model saved.")



def cleanup(env):
    pass
    # """Closes the environment."""
    # try:
    #     print("[*] Closing environment...")
    #     if env is not None:
    #         env.close()
    #     print("[*] Environment closed.")
    # except Exception as e:
    #     print(f"ERROR closing environment: {e}")

def main():
    """Main execution function."""
    args = parse_args()
    cfg = load_and_prepare_config(args)

    run_dir, checkpoints_dir, tb_log_path, wandb_context = setup_logging_and_saving(cfg)

    # Use the WandB context manager
    with wandb_context: # This ensures wandb.finish() is called even on error
        set_random_seeds(cfg.get('seed'))
        initialize_simulator(cfg)
        env = None # Initialize env to None for cleanup
        try:
            env = setup_environment(cfg)
            agent = setup_algorithm(cfg, env, tb_log_path)
            train_agent(agent, cfg, run_dir, checkpoints_dir, wandb_context)
        except Exception as e:
             print(f"\n--- An error occurred during setup or training ---")
             import traceback
             traceback.print_exc()
             print(f"Error: {e}")
             print(f"Run artifacts/logs might be incomplete in: {run_dir}")
        finally:
            # Ensure cleanup happens
            cleanup(env)

    print(f"\n[*] Script finished. Results saved in: {run_dir}")

if __name__ == "__main__":
    main()