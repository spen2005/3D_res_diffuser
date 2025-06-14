# src/utils/config.py

import yaml
import os
import copy
from typing import Dict, Any, Optional

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path: Path to the YAML configuration file. If None or the
                     file doesn't exist, returns an empty dictionary.

    Returns:
        A dictionary containing the configuration settings.
    """
    if config_path is None:
        print("INFO: No config path provided, using default empty config.")
        return {}

    if not os.path.exists(config_path):
        print(f"WARNING: Config file not found at {config_path}. Using default empty config.")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None: # Handle empty YAML file case
                return {}
            print(f"[*] Configuration loaded successfully from: {config_path}")
            return config
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse YAML file {config_path}: {e}")
        # Depending on requirements, you might want to raise the error
        # raise e
        return {} # Or return empty dict on error
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading config {config_path}: {e}")
        # raise e
        return {}


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges two configuration dictionaries.

    Values in override_config take precedence. Dictionaries are merged
    recursively, while other types are simply overwritten.

    Args:
        base_config: The base configuration dictionary.
        override_config: The dictionary with overrides.

    Returns:
        A new dictionary representing the merged configuration.
        The input dictionaries are not modified.
    """
    # Start with a deep copy of the base to avoid modifying the original
    merged_config = copy.deepcopy(base_config)

    for key, value in override_config.items():
        # If the key exists in the base and both values are dictionaries, recurse
        if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
            merged_config[key] = merge_configs(merged_config[key], value)
        # Otherwise, just override (or add) the value from the override config
        else:
            merged_config[key] = copy.deepcopy(value) # Deepcopy overrides as well for safety

    return merged_config

# --- Example Usage (can be removed or put under if __name__ == '__main__') ---
if __name__ == '__main__':
    # Create dummy config files for testing
    os.makedirs('temp_configs', exist_ok=True)
    base_yaml_content = """
    learning_rate: 0.001
    optimizer: Adam
    environment:
      name: Walker2d-v3
      frame_skip: 4
    algorithm:
      gamma: 0.99
      batch_size: 256
    """
    override_yaml_content = """
    learning_rate: 0.0005 # Override
    environment:
      frame_skip: 8      # Override nested value
      time_limit: 1000   # Add nested value
    new_param: true        # Add new top-level value
    """
    with open('temp_configs/base.yaml', 'w') as f:
        f.write(base_yaml_content)
    with open('temp_configs/override.yaml', 'w') as f:
        f.write(override_yaml_content)

    print("--- Loading Base Config ---")
    base_cfg = load_config('temp_configs/base.yaml')
    print(base_cfg)

    print("\n--- Loading Override Config ---")
    override_cfg = load_config('temp_configs/override.yaml')
    print(override_cfg)

    print("\n--- Merging Configs (Override takes precedence) ---")
    merged_cfg = merge_configs(base_cfg, override_cfg)
    print(yaml.dump(merged_cfg, default_flow_style=False)) # Print nicely using yaml.dump

    print("\n--- Loading Non-existent Config ---")
    non_existent_cfg = load_config('temp_configs/does_not_exist.yaml')
    print(non_existent_cfg)

    print("\n--- Merging with Empty Base ---")
    merged_empty_base = merge_configs({}, override_cfg)
    print(yaml.dump(merged_empty_base, default_flow_style=False))

    print("\n--- Merging with Empty Override ---")
    merged_empty_override = merge_configs(base_cfg, {})
    print(yaml.dump(merged_empty_override, default_flow_style=False))


    # Cleanup dummy files
    # import shutil
    # shutil.rmtree('temp_configs')