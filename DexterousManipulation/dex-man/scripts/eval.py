# eval.py
import argparse
import os
import yaml
import time
import random
import importlib
from typing import Optional, List, Dict, Any
from datetime import datetime
import warnings

import numpy as np
import torch
import genesis as gs
from gymnasium import spaces
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
# from src.DEXMAN_100.oakink_4c966.JarTwistingTask1 import JarTwistingTask1
# from src.DEXMAN_100.oakink_5ad66.ScoopingTask1 import ScoopingTask1
from src.DEXMAN_100.ManipTask import ManipTask


# Import feature extractor
from src.models.cnn_feature_extractor import CNN_Extractor # Adjust import path

# Use config loading utilities from src/utils/config.py
from src.utils.config import load_config, merge_configs # Make sure this file exists

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
def load_and_prepare_config(args: argparse.Namespace) -> dict:
    """Loads base config, task configs, and merges args for evaluation."""
    print(f"[*] Loading base configuration from: {args.config}")
    cfg = load_config(args.config) # Load the *original training config*

    # --- Override specific config values for evaluation ---
    # Primarily seed and viewer, potentially num_envs if desired for eval
    if args.seed is not None:
        cfg['seed'] = args.seed # Use the eval seed
    if args.show_viewer is not None: # Check if passed, could be True or False
         cfg['environment']['show_viewer'] = args.show_viewer
         print(f"[*] Setting viewer based on command line: {args.show_viewer}")
    if args.num_envs is not None:
        cfg['environment']['n_envs_total'] = args.num_envs
        print(f"[*] Overriding num_envs for evaluation: {args.num_envs}")
        # Warning if envs_per_task was set, as it might become inconsistent
        if cfg['environment'].get('envs_per_task') is not None:
             original_total = sum(cfg['environment']['envs_per_task'])
             if original_total != args.num_envs:
                  warnings.warn(f"Overriding n_envs_total ({args.num_envs}) while envs_per_task was specified in config (sum={original_total}). Distribution might change.")
             # Clear envs_per_task if num_envs is overridden, let MultiTaskVecEnv distribute
             cfg['environment']['envs_per_task'] = None


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

    return cfg


# --- Utility Functions (Copied/Adapted from train.py) ---

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
    print("[*] Initializing Genesis Simulator for Evaluation...")
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


# --- Evaluation Specific Functions ---

def parse_args():
    """Parses command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent for dexterous manipulation tasks.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the original YAML configuration file used for training (for env setup).')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved agent model file (.zip).')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                        help='Number of episodes to run for evaluation.')
    parser.add_argument('--seed', type=int, default=None, # Default to random seed for eval unless specified
                        help='Override random seed for evaluation.')
    parser.add_argument('--num-envs', type=int, default=None,
                        help='Override number of parallel environments for evaluation (optional).')
    parser.add_argument('--show-viewer', action=argparse.BooleanOptionalAction, default=None,
                        help='Show simulation viewer during evaluation (e.g. --show-viewer or --no-show-viewer). Overrides config.')
    parser.add_argument('--device', type=str, default='cuda', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for evaluation (cpu, cuda, auto).')

    parser.add_argument('--recording', action=argparse.BooleanOptionalAction, default=None, help='Record or not')

    args = parser.parse_args()
    # Basic validation
    if not os.path.exists(args.config):
         parser.error(f"Config file not found: {args.config}")
    if not os.path.exists(args.model_path) or not args.model_path.endswith('.zip'):
         parser.error(f"Model file not found or not a .zip file: {args.model_path}")

    return args

def run_evaluation(cfg: dict, model_path: str, n_eval_episodes: int, device: str, recording: bool):
    """Loads model and runs evaluation."""

    # --- Environment Setup ---
    # Uses the config loaded via load_and_prepare_config_eval
    env = setup_environment(cfg)

    # --- Load Model ---
    algo_name = cfg.get('algorithm', {}).get('name', 'PPO') # Default to PPO if not specified
    if algo_name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{algo_name}' specified in config not found in ALGORITHM_REGISTRY.")
    AlgoClass = ALGORITHM_REGISTRY[algo_name]

    print(f"[*] Loading model ({algo_name}) from: {model_path}")
    try:
        # Pass the environment for observation/action space checks
        model = AlgoClass.load(AlgoClass, path=model_path, env=env, device=device)
        print(f"[*] Model loaded successfully on device '{model.device}'.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Ensure the model path is correct and the environment configuration matches the training setup.")
        raise

    # --- Evaluation Loop ---
    episode_rewards = []
    episode_lengths = []
    # Success rates would require the env to return 'is_success' in the info dict
    # episode_successes = []

    num_envs = env.n_envs
    current_rewards = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs, dtype="int")

    print(f"[*] Starting evaluation for {n_eval_episodes} episodes across {num_envs} environments...")
    start_time = time.time()
    episodes_completed = 0

    model.policy.train(False)
    if recording:
        env.cam.start_recording()
    obs = env.reset() # Get initial observations

    while episodes_completed < n_eval_episodes:
        # Use deterministic actions for evaluation
        actions, values, log_probs = model.policy(obs, deterministic=True)
        # actions[:] *= 0
        actions = torch.clip(actions, torch.tensor(model.action_space.low, device=model.device), torch.tensor(model.action_space.high, device=model.device))

        # Clip actions if the environment doesn't do it internally (check env implementation)
        # action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, rewards, dones, infos = env.step(actions)

        current_rewards += rewards.cpu().numpy()
        current_lengths += 1

        for i in range(num_envs):
            if dones[i]:
                if episodes_completed < n_eval_episodes: # Only record if we still need episodes
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    # if 'is_success' in infos[i]:
                    #      episode_successes.append(infos[i]['is_success'])
                    episodes_completed += 1
                    print(f"  Episode {episodes_completed}/{n_eval_episodes} finished. Reward: {current_rewards[i]:.2f}, Length: {current_lengths[i]}")

                # Reset the specific environment that finished
                current_rewards[i] = 0
                current_lengths[i] = 0
                # Note: VecEnv automatically resets the env and returns the new obs[i]
                # If using non-SB3 VecEnv, might need manual reset handling here.

        # Render if viewer is enabled
        # if cfg.get('environment', {}).get('show_viewer', False):
             # Rendering might be handled internally by MultiTaskVecEnv based on show_viewer flag
             # Or you might need env.render() if supported by the VecEnv wrapper
             # time.sleep(0.01) # Add small delay for visualization
             # pass # Assuming render happens in step/reset if viewer is on

    if recording:
        env.cam.stop_recording(save_to_filename="video.mp4", fps=60)
    eval_duration = time.time() - start_time
    print(f"[*] Evaluation finished in {eval_duration:.2f} seconds.")

    # --- Calculate and Report Metrics ---
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if episode_rewards else 0
    mean_length = np.mean(episode_lengths) if episode_lengths else 0
    std_length = np.std(episode_lengths) if episode_lengths else 0
    # mean_success = np.mean(episode_successes) if episode_successes else float('nan')

    print("\n--- Evaluation Results ---")
    print(f"Episodes Evaluated: {episodes_completed} (Target: {n_eval_episodes})")
    print(f"Mean Reward:        {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length:{mean_length:.2f} +/- {std_length:.2f}")
    # print(f"Mean Success Rate:  {mean_success:.2%}")
    print("------------------------\n")
    return env # Return env for potential cleanup outside

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
    # torch.set_printoptions(sci_mode=False)
    """Main execution function for evaluation."""
    args = parse_args()
    cfg = load_and_prepare_config(args)

    set_random_seeds(cfg.get('seed'))
    initialize_simulator(cfg)

    env = None # Initialize env to None for cleanup
    try:
        env = run_evaluation(
            cfg=cfg,
            model_path=args.model_path,
            n_eval_episodes=args.n_eval_episodes,
            device=args.device,
            recording=args.recording    
        )
    except Exception as e:
         print(f"\n--- An error occurred during evaluation ---")
         import traceback
         traceback.print_exc()
         print(f"Error: {e}")
    finally:
        # Ensure cleanup happens
        cleanup(env)

    print(f"\n[*] Evaluation script finished.")

if __name__ == "__main__":
    main()
