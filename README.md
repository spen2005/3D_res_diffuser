# Residual Learning for Cross Embodiment Robotic Tasks
This is the repository for Residual Learning for Cross Embodiment Robotic Tasks, our Embodied Vision 2025 final project.
## Setup
```
conda create -n resrl python=3.10
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e .
cd ..
pip install torch
pip install stable-baselines3[extra]
```

## Example Usage
```
python scripts/train.py --config configs/muck_ppo.yaml --run-name muck --n-envs 1024 --seed 42
```