```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e .
cd ..
pip install stable-baselines3[extra]
cd RL
pip install -e .
python envs/Genesis/train.py 