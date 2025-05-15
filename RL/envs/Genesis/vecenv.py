import genesis as gs
import numpy as np

class VecEnv:
    def __init__(self, n_envs = 20, show_viewer = True, cam_pos = (1, 0, 1.1), cam_lookat = (0.0, 0.0, 0.6), cam_fov = 40, res = (512, 512), dt = 0.01, spacing = (1.5, 1.5), GUI = False, save_video = False): 
        self.n_envs = n_envs
        self.dt = dt
        self.save_video = save_video
        self._build_scene(show_viewer, cam_pos, cam_lookat, cam_fov, res, dt, spacing, GUI)

    def reset():
        pass

    def step(actions):
        pass

    def reset_env(self, envs_idx):
        pass

    def _build_scene(self, show_viewer, cam_pos, cam_lookat, cam_fov, res, dt, spacing, GUI):
        print("num envs: ", self.n_envs)
        self.scene = gs.Scene(
            show_viewer = show_viewer,
            sim_options=gs.options.SimOptions(
                dt       = dt,
                substeps = 2,
            ),
            viewer_options = gs.options.ViewerOptions(
                camera_pos = cam_pos,
                camera_lookat = cam_lookat,
                camera_fov = cam_fov,
            ),
            rigid_options = gs.options.RigidOptions(
                dt = dt,
                enable_self_collision = False,
                use_contact_island=False,  # Note: for multiple-contact
                constraint_solver=gs.constraint_solver.Newton,
                tolerance=1e-8,
                iterations=100,
                ls_iterations=50,
                max_collision_pairs=1000,
            ),
            vis_options = gs.options.VisOptions(
                env_separate_rigid = True,
            ),
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )


        self.cam = self.scene.add_camera(
            res    = res,
            pos    = (cam_pos[0] , cam_pos[1] , cam_pos[2]),
            lookat = (cam_lookat[0], cam_lookat[1], cam_lookat[2]),
            fov    = cam_fov,
            GUI    = GUI,
        )
        if self.save_video:
            self.cam2 = self.scene.add_camera(
                res    = (1920, 1080),
                pos    = (cam_pos[0] , cam_pos[1] , cam_pos[2]),
                lookat = (cam_lookat[0], cam_lookat[1], cam_lookat[2]),
                fov    = cam_fov,
                GUI    = False,
            )
            

        self._add_entity()
        self.scene.build(n_envs = self.n_envs, env_spacing=(0,0), center_envs_at_origin = False)
    
    def _add_entity(self):
        pass

    def _get_obs(self):
        pass

    def _get_reward(self):
        pass

    def _get_done(self):
        pass

    def _get_info(self):
        pass