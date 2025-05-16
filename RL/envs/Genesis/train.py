import genesis as gs
import torch
import torch.nn as nn
import time
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

from RL.algos.ppo import PPO

from vecenv import VecEnv

gs.init(backend=gs.gpu)

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # Get input dimensions
        img_shape = observation_space['image'].shape
        if len(img_shape) == 3:  # [H, W, C]
            img_height, img_width = img_shape[0], img_shape[1]
            n_input_channels = img_shape[2]
        else:  # [C, H, W] format
            n_input_channels = img_shape[0]
            img_height, img_width = img_shape[1], img_shape[2]
            
        n_proprio = observation_space['proprioception'].shape[0]

        super().__init__(observation_space, features_dim=256)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the output size of CNN for the actual input dimensions
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.zeros(1, n_input_channels, img_height, img_width)
            ).shape[1]
            
        # Add image feature reduction layer
        self.image_encoder = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.proprio_net = nn.Sequential(
            nn.Linear(n_proprio, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Now combined_net takes reduced image features (128) + proprio features (64)
        self.combined_net = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

    def forward(self, observations):
        image = observations['image'].float()
        
        # Permute dimensions from [B, H, W, C] to [B, C, H, W]
        if len(image.shape) == 4 and image.shape[-1] == 3:  # If input is [B, H, W, C]
            image = image.permute(0, 3, 1, 2)

        # Process image through CNN
        image_features = self.cnn(image)
        image_features = self.image_encoder(image_features)

        proprio_features = observations['proprioception'].float()
        proprio_features = self.proprio_net(proprio_features)

        # Combine features
        combined = torch.cat([image_features, proprio_features], dim=1)

        return self.combined_net(combined)

class MyEnv(VecEnv):
    
    def __init__(
        self, 
        observation_space,
        action_space,
        bottle_pos = (-0.24, 0.2786, 0.63),
        n_envs = 1024, 
        show_viewer = False, 
        cam_pos = (0, 0, 2), 
        cam_lookat = (0.0, 1.0, 0.0), 
        cam_fov = 40, 
        res = (64, 64),
        dt = 0.01,
        spacing = (0, 0),
        GUI = False
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.bottle_pos = bottle_pos

        super().__init__(n_envs, show_viewer, cam_pos, cam_lookat, cam_fov, res, dt, spacing, GUI)

        self.robot.set_dofs_kp(np.array([5000]*9)) 
        self.robot.set_dofs_kv(np.array([500]*9))

        self.traj_idxs = [0 for _ in range(n_envs)]
        self.accum_xyz_r = torch.zeros((n_envs, 3), device=gs.device)
        self.accum_xyz_l = torch.zeros((n_envs, 3), device=gs.device)
        self.sum_rewards = torch.zeros((n_envs,), device=gs.device)
        self.goal_rewards = torch.zeros((n_envs,), device=gs.device)
        self.prev_time = time.time()
    
    def reset(self):
        envs_idx = list(range(self.n_envs))
        self._reset_env(envs_idx)
        return self._get_obs()

    def step(self, actions):

        for i in range(self.n_envs):
            self.traj_idxs[i] += 1

        rewards, goal_rewards = self._get_reward(actions)
        self.sum_rewards += rewards
        self.goal_rewards += goal_rewards
        dones = self._get_done()
        infos = self._get_info(dones)

        # Reset finished environments
        done_envs_idx = []
        for i, done in enumerate(dones):
            if done:
                done_envs_idx.append(i)
        self._reset_env(done_envs_idx)

        # Extra step to get the transitions
        qpos = self.robot.inverse_kinematics(
            pos = np.random.uniform(0.5, 1, (self.n_envs, 3)),
            quat = np.tile([1, 0, 0, 0], (self.n_envs, 1)),
            max_samples = 20,
            init_qpos = self.robot.get_dofs_position().contiguous(),
            link = self.robot.get_link("panda_link7"),
        )
        self.robot.control_dofs_position(qpos)
        for _ in range(100):
            self.scene.step()

        states = self._get_obs() 
        print("Time taken for one step: ", (time.time() - self.prev_time) / self.n_envs)
        self.prev_time = time.time()

        return states, rewards, dones, infos

    def _reset_env(self, envs_idx):
        if len(envs_idx) == 0:
            return
        for idx in envs_idx:
            self.traj_idxs[idx] = 0
            
        return

    def _add_entity(self):
        # Load right robot URDF
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="./assets/panda_bullet/panda.urdf",
                pos = (0, -0.4, 0),
                fixed=True,
                merge_fixed_links=True,
                convexify=True,
                decompose_robot_error_threshold=float("inf"),
            )
        )
        self.table = self.scene.add_entity(
            gs.morphs.Box(
                pos = (0, 0.9, 0.3),
                size = (1.5, 1.5, 0.6),
                fixed = True,
            ),
            surface=gs.surfaces.Plastic(color=(0, 0, 0))
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                pos = (0, 0.22, 0.62),
                size = (0.04, 0.04, 0.04),
                fixed = False,
            ),
            surface=gs.surfaces.Plastic(color=(1.0, 0, 0))
        )
        self.bowl = self.scene.add_entity(
            gs.morphs.URDF(
                pos = (-0.2, 0.22, 0.7),
                file="./assets/C12001/C12001.urdf",
                fixed=False, 
                convexify=True,
                coacd_options=gs.options.CoacdOptions(
                        threshold=0.05,
                ),
            )
        )
    
    def _get_obs(self):
        import time
        start = time.time()
        images = self.cam.render()[0]

        images = torch.tensor(images, device=gs.device)

        proprioception = []
        for i in range(self.n_envs):
            proprioception.append(torch.zeros(9))
        proprioception = torch.stack(proprioception)

        state = {
            'image': images,
            'proprioception': proprioception
        }
        return state

    def _get_reward(self, actions):
        reward = torch.zeros(self.n_envs, device=gs.device)
        goal_reward = torch.zeros(self.n_envs, device=gs.device)

        return reward, goal_reward

    def _get_done(self):
        dones = np.zeros(self.n_envs)
        return dones

    def _get_info(self, dones):
        info = []
        for i in range(self.n_envs):
            if dones[i]:
                info.append({"terminal_observation": None, "TimeLimit.truncated": True, "episode": {"r": self.sum_rewards[i].item(), "l": self.traj_idxs[i], "g": self.goal_rewards[i].item()}})
            else:
                info.append({"terminal_observation": None, "TimeLimit.truncated": False}) 
        return info
    

if __name__ == "__main__":

    from flower_vla_calvin.flower.models.flower import FLOWERVLA
    import json
    from safetensors.torch import load_file
    
    from flower_vla_calvin.flower.models.flower import FLOWERVLA

    with open("config.json", "r") as f:
        config = json.load(f)

    base_model = FLOWERVLA(**config["model_config"], load_pretrained = True, pretrained_model_path="model.safetensors")
    # state_dict = load_file("model.safetensors")
    # print(state_dict.keys())
    # model.load_state_dict(state_dict)
    base_model = base_model.cuda()
    base_model.eval()
    print("done loading")

    observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(64,64,3),
                dtype=np.uint8
            ),
            'proprioception': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(9,),
                dtype=np.float32
            )
    })
    
    action_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)

    env = MyEnv(observation_space, action_space, show_viewer=True, n_envs=4)

    policy_kwargs = {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {},
        "net_arch": dict(pi=[64, 64], vf=[64, 64])  # Policy and value networks after feature extraction
    }

    model = PPO(
        "ActorCriticPolicy",
        env,
        learning_rate = 3e-4,
        n_steps = 128,
        batch_size = 64,
        gamma = 0.99,
        gae_lambda = 0.95,
        policy_kwargs=policy_kwargs,
        verbose = 1,    
        tensorboard_log = "./ppo_env_tensorboard/",
    )

    model.learn(total_timesteps=20000000)
    model.save("64_64_1024_32_64_20M_xyz_random_bottle-9")