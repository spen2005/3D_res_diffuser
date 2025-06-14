import genesis as gs # Assuming gs is initialized elsewhere
import torch
import torch.nn as nn
import time
import numpy as np
import random
# import pickle
import joblib

from src.DEXMAN_100.task import Task
from typing import Tuple, Dict, List, Any
from gymnasium import spaces


# Assume Task base class is defined above

class ManipTask(Task):
    """Task: manipulate objects using dual arms based on reference trajectories."""

    def __init__(self, task_id: int, config: Dict[str, Any], envs_per_task: int = 1, xyz_limit: float = None, joint_limit: float = None, n_envs: int = 1) -> None:
        super().__init__(task_id, config)
        print("config:", config)
        # Load trajectories specific to this task
        self.xyz_limit = xyz_limit
        # self.obj_traj = pickle.load(open(config['obj_traj_path'], 'rb'))
        self.obj_traj = joblib.load(config['obj_traj_path'])
        print(self.obj_traj.keys())
        # config['obj_keys'] = []
        self.obj_traj = {key: self.obj_traj[key] for key in config['obj_keys']}
        self.obj_pos_diff_tolerance = config['obj_pos_diff_tolerance']
        self.obj_rot_diff_tolerance = config['obj_rot_diff_tolerance']
        self.friction = config['friction']
        self.table_pos = config['table_pos']

        translation = config['translation']
        translation = torch.tensor([translation[0], translation[1], translation[2]], device=gs.device)
        
        start_idx = config['start_idx']
        end_idx = config['end_idx']

        self.traj = torch.from_numpy(np.float32(np.load(config['robot_traj_path'])[start_idx:end_idx])).to(gs.device)
        self.traj[:, 75:78] += translation
        self.traj[:, 82:85] += translation
        self.keypoints_r = self.traj[:, [89,  90,  91,  101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]].reshape(-1, 6, 3)
        self.keypoints_l = self.traj[:, [116, 117, 118, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142]].reshape(-1, 6, 3)

        for i, obj in enumerate(self.obj_traj.keys()):
            self.obj_traj[obj]["pos"] = torch.from_numpy(np.float32(self.obj_traj[obj]["pos"][start_idx:end_idx])).to(gs.device)
            self.obj_traj[obj]["quat"] = torch.from_numpy(np.float32(self.obj_traj[obj]["quat"][start_idx:end_idx])).to(gs.device)

        self.scale_factor = 1
        self.total_steps = 0

    def add_entity(self, env: 'MultiTaskVecEnv', task_id: int) -> None:
        """Add per task entities to the environment."""
        # If objects differ per task, this needs to be handled within task.reset()
        # or by creating separate object managers.

        self.pos_diff = torch.zeros((env.n_envs, len(self.obj_traj.keys())), device=gs.device) # pos distance to ground truth
        self.quat_diff = torch.zeros((env.n_envs, len(self.obj_traj.keys())), device=gs.device) # quat distance to ground truth
        self.obj_list = []
        obj_dict = {}

        for i, obj in enumerate(self.obj_traj.keys()):
            object_path = f"./assets/oakink_coacd/{obj}/{obj}.urdf"
            new_object = env.scene.add_entity(
                morph=gs.morphs.URDF(
                    file=object_path,
                    pos=(2 + 100 * task_id + i, 0, 0), # Initial pos set during reset
                    fixed=False, 
                    convexify=True,
                    coacd_options=gs.options.CoacdOptions(
                        threshold=0.05,
                    ),
                    # coacd_options=gs.options.CoacdOptions(
                    #     threshold=0.05,
                    # ),
                    # decimate=True,
                    # decompose_object_error_threshold=0.15,
                ),    
                # material = gs.materials.Rigid(
                #     friction = 4
                # ),
                # vis_mode="collision",
                # visualize_contact=True,
            )
            new_object.set_friction(3.5)
            self.obj_list.append(new_object)
            obj_dict[obj] = new_object

        return obj_dict
    
    def reset(self, env: 'MultiTaskVecEnv', envs_idx: List, task_id: int = None, n_dofs: int = None) -> None:
        if len(envs_idx) == 0:
            return

        # init_idx = [random.randint(0, 300) for _ in range(len(envs_idx))]
        init_idx = [0 for _ in range(len(envs_idx))]
        env.traj_idxs[envs_idx] = torch.tensor(init_idx, dtype=torch.int32, device=gs.device)

        all_idx = [i for i in range(env.n_envs)]

        # pos = self.traj[all_idx][:, 75:75+3]
        # quat = self.traj[all_idx][:, 78:78+4]
        pos = self.traj[all_idx][:, 82:82+3]
        quat = self.traj[all_idx][:, 85:85+4]

        q_pos = env.robot.inverse_kinematics(
            pos = pos,
            quat = quat,
            max_samples = 20,
            init_qpos = env.robot.get_dofs_position().contiguous(),
            link = env.robot.get_link("panda_hand"),
        )

        q_pos[:, -2] = 0.04

        q_pos = q_pos[envs_idx]

        env.robot.set_dofs_position(
            position=q_pos,
            envs_idx=torch.tensor(envs_idx, device=gs.device),
        )

        table_pos = torch.zeros((len(envs_idx), 3), device=gs.device, dtype=torch.float32)
        table_pos[:, 0] = self.table_pos[0]
        table_pos[:, 1] = self.table_pos[1]
        table_pos[:, 2] = self.table_pos[2]

        env.table.set_pos(
            pos=table_pos,
            envs_idx=torch.tensor(envs_idx, device=gs.device),
        )

        # Set the object positions based on the trajectory
        for i, obj in enumerate(self.obj_traj.keys()):
            pos = self.obj_traj[obj]["pos"][init_idx]
            pos = torch.tensor(pos, device=gs.device)
            quat = self.obj_traj[obj]["quat"][init_idx]
            quat = torch.tensor(quat, device=gs.device)
            self.obj_list[i].set_pos(
                pos=pos,
                envs_idx=torch.tensor(envs_idx, device=gs.device),
            )
            self.obj_list[i].set_quat(
                quat=quat,
                envs_idx=torch.tensor(envs_idx, device=gs.device),
            )

        self.pos_diff[envs_idx] = 0.0
        self.quat_diff[envs_idx] = 0.0

    def compute_actions(
        self, 
        env: 'MultiTaskVecEnv', 
        policy_action: torch.Tensor, 
        envs_idx: List, 
        wrist_pos_r_idx: int, 
        wrist_quat_r_idx: int, 
        wrist_pos_l_idx: int, 
        wrist_quat_l_idx: int, 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # policy_action to float
        # policy_action = policy_action.float()
        # print("policy_action", policy_action[0])
        # policy_action[:] = 0

        """Return joint angles and target xyz and quaternion for the arms."""
        # policy_action is shape (len(envs_idx), action_dim)
        # Important: Access task state using envs_idx
        traj_idx = env.traj_idxs[envs_idx]

        # # Set the object positions based on the trajectory
        # for i, obj in enumerate(self.obj_traj.keys()):
        #     pos = self.obj_traj[obj]["pos"][traj_idx]
        #     pos = torch.tensor(pos, device=gs.device)
        #     quat = self.obj_traj[obj]["quat"][traj_idx]
        #     quat = torch.tensor(quat, device=gs.device)
        #     self.obj_list[i].set_pos(
        #         pos=pos,
        #         envs_idx=torch.tensor(envs_idx, device=gs.device),
        #     )
        #     self.obj_list[i].set_quat(
        #         quat=quat,
        #         envs_idx=torch.tensor(envs_idx, device=gs.device),
        #     )


        # TODO
        # env.accum_xyz[envs_idx] = 0

        # Accumulate XYZ deltas (clipping applied here)
        # delta_xyz_r = torch.clip(policy_action[:, 0:3], -self.xyz_limit, self.xyz_limit)
        # env.accum_xyz[envs_idx] = torch.add(env.accum_xyz[envs_idx], delta_xyz_r)
        delta_xyz_l = torch.clip(policy_action[:, 0:3], -self.xyz_limit, self.xyz_limit)
        env.accum_xyz[envs_idx] = torch.add(env.accum_xyz[envs_idx], delta_xyz_l)

        # Target EE poses based on trajectory + accumulated deltas
        # target_xyz = torch.add(env.accum_xyz[envs_idx], self.traj[traj_idx, wrist_pos_r_idx : wrist_pos_r_idx + 3])
        # quat = self.traj[traj_idx, wrist_quat_r_idx : wrist_quat_r_idx + 4]
        target_xyz = torch.add(env.accum_xyz[envs_idx], self.traj[traj_idx, wrist_pos_l_idx : wrist_pos_l_idx + 3])
        # env.accum_xyz[envs_idx] = 0
        quat = self.traj[traj_idx, wrist_quat_l_idx : wrist_quat_l_idx + 4]

        target_xyz = target_xyz.float()

        # gripper_action = env.robot.get_dofs_position()[envs_idx, -1]
        # gripper_action += policy_action[:, -1]
        gripper_action = torch.zeros(len(envs_idx), device=gs.device, dtype=torch.float32) + 0.04
        # for i, idx in enumerate(traj_idx):
        #     if idx > 10:
        #         gripper_action[i] = 0.003
        #     else:
        #         gripper_action[i] = max(0.005, 0.04 - idx * 0.0035)

        return target_xyz, quat, gripper_action
    

    def get_reward(
        self, env: 'MultiTaskVecEnv', envs_idx: List, task_id: int, policy_action: torch.Tensor
    ) -> Tuple[float, float]:
        self.total_steps += 1
        if self.total_steps % 32 == 0:
            self.scale_factor = max(0.7, self.scale_factor - 0.001)
            print("scale factor", self.scale_factor)

        # Penalty for exceeding action limits (based on policy_action *before* clipping in compute_actions)
        # xyz_actions = policy_action[:, 0:6]
        # joint_actions = policy_action[:, 6:50]
        # xyz_penalty = -torch.sum(torch.relu(torch.abs(xyz_actions) - self.xyz_limit) / self.xyz_limit, dim=1) * 0.
        # joint_penalty = -torch.sum(torch.relu(torch.abs(joint_actions) - self.joint_limit) / self.joint_limit, dim=1) * 0.
        # reward = torch.add(xyz_penalty, joint_penalty)


        reward = torch.zeros(len(envs_idx), device=gs.device, dtype=torch.float32)

        traj_idx = env.traj_idxs[envs_idx]
        # Goal reward based on difference of distance between t-1 and t
        goal_reward = torch.zeros(len(envs_idx), device=gs.device, dtype=torch.float32)
        for i, obj in enumerate(self.obj_traj.keys()):
            obj_pos = self.obj_list[i].get_pos()[envs_idx]

            pos_diff = torch.exp(-80 * torch.norm(obj_pos - self.obj_traj[obj]["pos"][traj_idx], dim=1))
            # pos_diff = (-torch.norm(obj_pos - self.obj_traj[obj]["pos"][traj_idx], dim=1) + 6) / 6
            goal_reward += 5 * pos_diff
            self.pos_diff[envs_idx, i] = pos_diff

            obj_quat = self.obj_list[i].get_quat()[envs_idx]
            obj_quat_diff = torch.exp(-3 * self.quat_dist(obj_quat, self.obj_traj[obj]["quat"][traj_idx]))
            # obj_quat_diff = (-self.quat_dist(obj_quat, self.obj_traj[obj]["quat"][traj_idx]) + 1.57) / 1.57
            goal_reward += obj_quat_diff
            self.quat_diff[envs_idx, i] = obj_quat_diff

           
        # If the goal_reward is NaN, set goal reward to 0
        goal_reward[torch.isnan(goal_reward)] = 0
        reward = goal_reward 
        success = torch.zeros(len(envs_idx), device=gs.device, dtype=torch.float32)

        # print("max reward", torch.max(reward))
        # print("min reward", torch.min(reward))

        # debug
        # reward = torch.ones(len(envs_idx), device=gs.device, dtype=torch.float32)
        goal_reward = torch.ones(len(envs_idx), device=gs.device, dtype=torch.float32)


        return reward.float(), goal_reward.float(), success.float() # Return total reward and the goal component

    def done_list(self, env: 'MultiTaskVecEnv', envs_idx: List, task_id: int) -> List:
        dones = []
        traj_idx = env.traj_idxs[envs_idx]
        max_traj_len = len(self.traj)  # Assuming all trajectories have same length

        traj_done = (traj_idx >= max_traj_len - 2)
        dones.extend([envs_idx[i] for i, done in enumerate(traj_done) if done])

        for j, obj in enumerate(self.obj_traj.keys()):
            obj_pos_batch = self.obj_list[j].get_pos()[envs_idx]  # shape: (batch_size, 3)
            obj_quat_batch = self.obj_list[j].get_quat()[envs_idx]  # shape: (batch_size, 4)
            target_pos_batch = torch.stack([self.obj_traj[obj]["pos"][idx] for idx in traj_idx])  # shape: (batch_size, 3)
            target_quat_batch = torch.stack([self.obj_traj[obj]["quat"][idx] for idx in traj_idx])  # shape: (batch_size, 4)

            pos_diff = torch.norm(obj_pos_batch - target_pos_batch, dim=-1)  # shape: (batch_size,)
            quat_diff = self.quat_dist(obj_quat_batch, target_quat_batch)    # shape: (batch_size,)

            pos_fail_mask = pos_diff > self.obj_pos_diff_tolerance * self.scale_factor
            quat_fail_mask = quat_diff > self.obj_rot_diff_tolerance * self.scale_factor

            for i in range(len(envs_idx)):
                if pos_fail_mask[i]:
                    print(f"Env {envs_idx[i]}: Object {obj} pos diff {pos_diff[i]:.3f} exceeds tolerance {self.obj_pos_diff_tolerance * self.scale_factor} at timestep {traj_idx[i]}")
                    dones.append(envs_idx[i])
                elif quat_fail_mask[i]:
                    print(f"Env {envs_idx[i]}: Object {obj} quat diff {quat_diff[i]:.3f} exceeds tolerance {self.obj_rot_diff_tolerance * self.scale_factor} at timestep {traj_idx[i]}")
                    dones.append(envs_idx[i])
         
        dones = list(set(dones))
        # update tolerance
        # self.obj_pos_diff_tolerance = max(0.03, self.obj_pos_diff_tolerance - 0.0000003)
        # self.obj_rot_diff_tolerance = max(0.75, self.obj_rot_diff_tolerance - 0.0000075)

        return dones


    def get_proprioception(
        self, 
        envs_idx: List, 
        env: 'MultiTaskVecEnv',
        wrist_pos_r_idx: int, 
        wrist_quat_r_idx: int, 
        wrist_pos_l_idx: int, 
        wrist_quat_l_idx: int, 
    ) -> torch.Tensor:
        # Image rendering is typically done for all envs at once in the main env loop
        # We assume the image for envs_idx is passed or accessible

        xyz = env.robot.get_link("panda_hand").get_pos()[envs_idx]
        quat = env.robot.get_link("panda_hand").get_quat()[envs_idx]

        traj_idxs = env.traj_idxs[envs_idx]
        next_traj_idxs = torch.clamp(traj_idxs + 1, max=len(self.traj) - 1)

        next_actions = self.traj[next_traj_idxs]
        next_xyz = next_actions[:, wrist_pos_r_idx : wrist_pos_r_idx + 3] + env.accum_xyz[envs_idx]
        next_quat = next_actions[:, wrist_quat_r_idx : wrist_quat_r_idx + 4] + env.accum_quat[envs_idx]

        # next_xyz = next_actions[:, wrist_pos_l_idx : wrist_pos_l_idx + 3] + env.accum_xyz[envs_idx]
        # next_quat = next_actions[:, wrist_quat_l_idx : wrist_quat_l_idx + 4] + env.accum_quat[envs_idx]
        gripper_state = env.robot.get_dofs_position()[envs_idx, -1]

        # Concatenate all at once
        proprioception = torch.cat([
            xyz,
            quat,
            next_xyz,
            next_quat,
            gripper_state.unsqueeze(1),
        ], dim=1)

        # Here we return only proprioception, image handled in main loop
        return proprioception
    
    def quat_dist(self, quat1, quat2):
        """
        quat1, quat2: (N, 4) torch tensors, format [w, x, y, z]
        Returns: (N,) torch tensor, rotation angles in radians (absolute value)
        """
        
        # Compute relative quaternion: q_rel = q1_conj * q2
        w1, x1, y1, z1 = quat1.unbind(dim=-1)
        w2, x2, y2, z2 = quat2.unbind(dim=-1)
        
        # q1 conjugate
        w1c = w1
        x1c = -x1
        y1c = -y1
        z1c = -z1
        
        # Quaternion multiplication: (w1c, x1c, y1c, z1c) * (w2, x2, y2, z2)
        w = w1c*w2 - x1c*x2 - y1c*y2 - z1c*z2
        x = w1c*x2 + x1c*w2 + y1c*z2 - z1c*y2
        y = w1c*y2 - x1c*z2 + y1c*w2 + z1c*x2
        z = w1c*z2 + x1c*y2 - y1c*x2 + z1c*w2
        
        # Relative quaternion is (w, x, y, z)

        w = torch.abs(w)
        
        # Compute rotation angle
        w = torch.clamp(w, -1.0, 1.0)
        theta = 2 * torch.acos(w)  # radians
        
        return torch.abs(theta)