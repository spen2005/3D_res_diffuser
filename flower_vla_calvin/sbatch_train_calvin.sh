#!/bin/bash

#SBATCH -p accelerated
#SBATCH -A hk-project-sustainebot
#SBATCH -J FLOW_CALVIN

# Cluster Settings
#SBATCH -n 4       # Number of tasks
#SBATCH -c 16  # Number of cores per task
#SBATCH -t 12:00:00 ## 1-00:30:00 # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4


# Define the paths for storing output and error files
#SBATCH --output=/home/hk-project-sustainebot/ft4740/code/flower_vla_calvin/logs/outputs/%x_%j.out
#SBATCH --error=/home/hk-project-sustainebot/ft4740/code/flower_vla_calvin/logs/outputs/%x_%j.err


# -------------------------------
# Activate the virtualenv / conda environment
conda activate flower_cal

export TORCH_USE_CUDA_DSA=1
# NNODES=1
# NODE_RANK=0
# PORT=29500
# MASTER_ADDR=127.0.0.1
#CUDA_VISIBLE_DEVICES=0,1,2,3  

srun python /home/hk-project-sustainebot/ft4740/code/flower_vla_calvin/flower/training_calvin.py seed=42
# batch_size=2 model=smolflow_agent # batch_size=4 model.use_lora=False # model=vlm_berg_agent # batch_size=2 model.vla_mode='reduced_head' #model.use_perceiver=False model.use_incontext=True #seed=242 #model=mode_agent