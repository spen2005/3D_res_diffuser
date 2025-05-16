```bash
# git clone https://github.com/Genesis-Embodied-AI/Genesis.git
# cd Genesis
# pip install -e .
# cd ..
# pip install stable-baselines3[extra]
# cd RL
# pip install -e .
conda create -n resrl python=3.10
pip install -r requirements.txt
conda install -c conda-forge libstdcxx-ng
cd RL
pip install -e .
python envs/Genesis/train.py
cd ..
cd flower_vla_calvin
python testing.py
cd ..
```