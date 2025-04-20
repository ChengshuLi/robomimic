"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy

import omnigibson as og
import omnigibson.lazy as lazy
import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB
from omnigibson.envs import create_wrapper

import omnigibson.utils.transform_utils as T
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

from mimicgen.train_scripts.train_prep_data import compute_point_cloud_from_rgbd, pcd_vis, color_pcd_vis

import torch as th
import numpy as np
import gym
import time
import copy


from omnigibson.macros import gm

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False

DEBUG = True


class EnvOmniGibson(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self,
        env_name,
        **kwargs,
    ):
        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)

        if og.sim is not None:
            og.sim.stop()
            og.clear()

        self.env = og.Environment(configs=kwargs)
        # TODO: uncomment the following lines for data generation.
        controller_config = {
            "base": {"name": "HolonomicBaseJointController", "motor_type": "position", "command_input_limits": None, "use_impedances": False, "control_freq": 10},
            "trunk": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
            "arm_left": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
            "arm_right": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
            "gripper_left": {"name": "MultiFingerGripperController", "mode": "binary"},
            "gripper_right": {"name": "MultiFingerGripperController", "mode": "binary"},
            "camera": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
        }

        self.env.robots[0].reload_controllers(controller_config=controller_config)
        # self.env.robots[0]._grasping_mode = "sticky"
        self.env.scene.update_initial_state()

        # Debug visualization
        self.eef_current_marker = PrimitiveObject(
            relative_prim_path="/eef_current_marker",
            primitive_type="Cube",
            name="eef_current",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([1, 0, 0, 1]),
        ) if DEBUG else None
        self.eef_goal_marker = PrimitiveObject(
            relative_prim_path="/eef_goal_marker",
            primitive_type="Cube",
            name="eef_goal_marker",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([0, 1, 0, 1]),
        ) if DEBUG else None

        # Debug visualization for bimanual setup
        self.eef_current_marker_left = PrimitiveObject(
            relative_prim_path="/eef_current_marker_left",
            primitive_type="Cube",
            name="eef_current_left",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([1, 0, 0, 1]),
        ) if DEBUG else None
        self.eef_goal_marker_left = PrimitiveObject(
            relative_prim_path="/eef_goal_marker_left",
            primitive_type="Cube",
            name="eef_goal_marker_left",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([0, 1, 0, 1]),
        ) if DEBUG else None
        self.eef_current_marker_right = PrimitiveObject(
            relative_prim_path="/eef_current_marker_right",
            primitive_type="Cube",
            name="eef_current_right",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([1, 0, 0, 1]),
        ) if DEBUG else None
        self.eef_goal_marker_right = PrimitiveObject(
            relative_prim_path="/eef_goal_marker_right",
            primitive_type="Cube",
            name="eef_goal_marker_right",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([0, 0, 1, 1]),
        ) if DEBUG else None

        if DEBUG:
            # og.sim.batch_add_objects([self.eef_current_marker, self.eef_goal_marker], [self.env.scene] * 2)
            og.sim.batch_add_objects([self.eef_current_marker_left, self.eef_goal_marker_left, 
                                      self.eef_current_marker_right, self.eef_goal_marker_right], [self.env.scene] * 4)
            og.sim.step()

        # Create CuRobo instance
        if self._init_kwargs['init_curobo']:
            self.primitive = StarterSemanticActionPrimitives(self.env, self.env.robots[0], enable_head_tracking=False, curobo_batch_size=1)
            self.cmg = self.primitive._motion_generator

        self.policy_rollout = False
        self.with_color = False

    def step(self, action, video_writer=None):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        # og_step_start_time = time.time()
        obs, r, done, truncated, info = self.env.step(action)
        # print('og step time', time.time()-og_step_start_time)
        
        # TODO: still need to verify whether the video writer works for the omnigibson env
        # if video_writer:
        #     robot_name = self.env.robots[0].name
        #     viewer_img = obs["external::viewer::rgb"]
        #     ego_img = obs[f"{robot_name}::{robot_name}:eyes:Camera:0::rgb"]
        #     eef_left_img = obs[f"{robot_name}::{robot_name}:left_eef_link:Camera:0::rgb"]
        #     eef_right_img = obs[f"{robot_name}::{robot_name}:right_eef_link:Camera:0::rgb"]
        #     concatenated_img = hori_concatenate_image([viewer_img, ego_img, eef_left_img, eef_right_img])
        #     video_writer.append_data(concatenated_img)

        # replace the observation with newly added IL obs function 
        # get_obs_time = time.time()
        obs = self.get_obs_IL()
        # print('get obs time', time.time()-get_obs_time)

        subtask_success_signal = self.get_subtask_success_signal()
        info['subtask_success_signal'] = subtask_success_signal

        # return obs, r, done, info
        # changed to output with truncated
        return obs, r, done, truncated, info

    # TODO: make it more generalizable
    # Get task relevant objects based on the env name (BDDL activity name)
    def _get_task_relevant_objs(self):
        if self.name.startswith("test_pen_book"):
            obj_names = ["rubber_eraser.n.01_1", "hardback.n.01_1"]
        elif self.name.startswith("test_cabinet"):
            obj_names = ["cabinet.n.01_1"]
        elif self.name.startswith("test_tiago_giftbox"):
            obj_names = ["gift_box.n.01_1"]
        elif self.name.startswith("test_tiago_notebook"):
            obj_names = ["notebook.n.01_1", "breakfast_table.n.01_1"]
        elif self.name.startswith("test_tiago_cup"):
            obj_names = ["coffee_cup.n.01_1", "dixie_cup.n.01_1", "breakfast_table.n.01_1"]
        elif self.name.startswith("test_r1_cup"):
            return [self.env.scene.object_registry("name", name) for name in ["coffee_cup", "teacup", "breakfast_table"]]
        else:
            raise ValueError(f"Unknown environment name: {self.name}")

        return [self.env.task.object_scope[obj] for obj in obj_names]

    def early_termination(self, env_step, ob_dict=None):
        """
        Check if the episode should be terminated early.
        """
        if env_step < 20:
            self.initial_positions = {}
            for obj in self._get_task_relevant_objs():
                self.initial_positions[obj.name] = obj.get_position_orientation()
        
        # check table movement
        cur_positions = {}
        for obj in self._get_task_relevant_objs():
            cur_positions[obj.name] = obj.get_position_orientation()
        
        for key in self.initial_positions.keys():
            if 'table' in key: # if table is moved, directly terminate the episode 
                if np.linalg.norm(self.initial_positions[key][0] - cur_positions[key][0]) > 0.1:
                    return True
        
        # early termination when the robot get stuck, NOT WORKING NOW
        # if env_step == 0:
        #     # note that the poses are in the robot frame
        #     self.old_eef_left_pos = copy.deepcopy(ob_dict['eef_left_pos'])
        #     self.old_eef_left_quat = copy.deepcopy(ob_dict['eef_left_quat'])
        #     self.old_eef_right_pos = copy.deepcopy(ob_dict['eef_right_pos'])
        #     self.old_eef_right_quat = copy.deepcopy(ob_dict['eef_right_quat'])
        # # if the robot is stuck for n steps, early terminate the episode
        # if env_step > 150 and env_step % 100 == 0:
        #     print('env_step', env_step)
        #     # update the robot eef position
        #     cur_eef_left_pos = ob_dict['eef_left_pos']
        #     cur_eef_left_quat = ob_dict['eef_left_quat']
        #     cur_eef_right_pos = ob_dict['eef_right_pos']
        #     cur_eef_right_quat = ob_dict['eef_right_quat']
        #     left_eef_pos_diff = np.linalg.norm(self.old_eef_left_pos - cur_eef_left_pos) 
        #     left_eef_pos_nomove = left_eef_pos_diff < 0.02 
        #     left_eef_quat_diff = np.linalg.norm(self.old_eef_left_quat - cur_eef_left_quat)
        #     left_eef_quat_nomove = left_eef_quat_diff < 0.012
        #     right_eef_pos_diff = np.linalg.norm(self.old_eef_right_pos - cur_eef_right_pos) 
        #     right_eef_pos_nomove = right_eef_pos_diff < 0.02 
        #     right_eef_quat_diff = np.linalg.norm(self.old_eef_right_quat - cur_eef_right_quat)
        #     right_eef_quat_nomove = right_eef_quat_diff < 0.012

        #     self.old_eef_left_pos = copy.deepcopy(cur_eef_left_pos)
        #     self.old_eef_left_quat = copy.deepcopy(cur_eef_left_quat)
        #     self.old_eef_right_pos = copy.deepcopy(cur_eef_right_pos)
        #     self.old_eef_right_quat = copy.deepcopy(cur_eef_right_quat)

        #     if left_eef_pos_nomove and left_eef_quat_nomove and right_eef_pos_nomove and right_eef_quat_nomove:
        #         print('enter no move breakpoint')
        #         print('')
        #         print('left_eef_pos_nomove', left_eef_pos_diff, left_eef_pos_nomove)
        #         print('left_eef_quat_nomove', left_eef_quat_diff, left_eef_quat_nomove)
        #         print('right_eef_pos_nomove', right_eef_pos_diff, right_eef_pos_nomove)
        #         print('right_eef_quat_nomove', right_eef_quat_diff, right_eef_quat_nomove)
        #         print('')
        #         breakpoint()
        #         # return True
        #         return False
                
        return False

    def set_object_pose(self, obj_poses):
        """
        Set the object pose for the task relevant objects
        """
        if 'states' in obj_poses.keys(): obj_poses = obj_poses['states']
        if self.name.startswith("test_r1_cup"):
            task_relevant_objs = self._get_task_relevant_objs()
            for obj in task_relevant_objs:
                if 'table' not in obj.name:
                    obj.set_position_orientation(obj_poses[obj.name][:3], obj_poses[obj.name][3:])
            print('finishe setting object pose for r1 robot')


    # TODO: make it more generalizable
    # randomize the pose of all the task relevant objects in xy-pos and z-rot
    def _randomize_object_pose_D1(self, objs):
        # default D1 randomization
        # pos_magnitude = 0.1  # 10cm
        # rot_magnitude = np.pi / 12  # 15 degrees

        # for debugging
        # pos_magnitude = 0.001
        # rot_magnitude = np.pi / 10000 

        pos_magnitude = 0.1  # 10cm
        rot_magnitude = np.pi / 12  # 15 degrees

        print('position random range (meter)', pos_magnitude*2, ' orientation random range (degree)', rot_magnitude*2/np.pi*180)
        print('sample position in D1, breakpoint in _randomize_object_pose_D1')

        if self.name.startswith("test_r1_cup"):
            for obj in objs:
                if "table" not in obj.name:
                    pos, orn = obj.get_position_orientation()
                    pos_diff_xy = np.random.uniform(-pos_magnitude, pos_magnitude, size=2)
                    pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
                    pos+= pos_diff

                    orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
                    orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))
                    obj.set_position_orientation(pos, orn)
                    print('finish randomize obj pose for r1 robot', obj.name)

        elif self.name.startswith("test_tiago_cup"):

            for obj in objs:
                if "table" not in obj.name:
                    pos, orn = obj.get_position_orientation()
                    pos_diff_xy = np.random.uniform(-pos_magnitude, pos_magnitude, size=2)
                    pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
                    # pos += pos_diff
                    if 'coffee' in obj.name:
                        pos_diff[1] = 0.0 # make the y random val 0, coffee cup only randomize in the x range
                        pos+= pos_diff
                        offset = th.from_numpy(np.array([-0.1, 0.05, 0.0])).float()
                        offset[1] = 0.0 
                        pos += offset
                    # TODO: the paper cup randomization range is larger
                    if 'paper' in obj.name:
                        pos_diff[1] = 0.0 # make the y random val 0, paper cup only randomize in the x range
                        pos+= pos_diff
                        offset = th.from_numpy(np.array([-0.1, -0.05, 0.0])).float()
                        offset[1] = 0.0 
                        pos += offset

                    # TODO： without mobile motion， the target pose need to be very carefully selected
                    # pos += th.from_numpy(np.array([-.15, 0.0, 0]))
                    orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
                    orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))
                    obj.set_position_orientation(pos, orn)

    def _randomize_object_pose_D2(self, objs):
        pos_magnitude = 0.10  # 5cm
        rot_magnitude = np.pi / 12  # 15 degrees

        # for debugging
        # pos_magnitude = 0.001
        # rot_magnitude = np.pi / 10000  # 15 degrees

        for obj in objs:
            if "table" not in obj.name:
                pos, orn = obj.get_position_orientation()
                pos_diff_xy = np.random.uniform(-pos_magnitude, pos_magnitude, size=2)
                pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
                pos += pos_diff
                # TODO： without mobile motion， the target pose need to be very carefully selected
                pos += th.from_numpy(np.array([-.15, 0.0, 0]))
                orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
                orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))

                pos[1] = -pos[1] # mirror the position along the y-axis
                orn = T.mat2quat(T.euler2mat(th.tensor([0.0, 0.0, np.pi])) @ T.quat2mat(orn)) # add pi orientation along the y-axis
                obj.set_position_orientation(pos, orn)

    # TODO: make it more generalizable
    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        # obs, info = self.env.reset()

        # Reset the task
        self.env.task.reset(self.env)

        # Reset internal variables
        self.env._reset_variables()

        # Run a single simulator step to make sure we can grab updated observations
        og.sim.step()

        # Grab and return observations
        obs, _ = self.env.get_obs()

        info = {}

        print('check the env name', self.name)
        # breakpoint()

        # D0 is the original distribution (no randomization at all - deterministic reset)
        if self.name.endswith("D0"):
            if self.name.startswith("test_tiago_cup"):
                # hardcode for the test_tiago_cup task
                task_relevant_objs = self._get_task_relevant_objs()
                for obj in task_relevant_objs:
                    if 'coffee' in obj.name:
                        pos, orn = obj.get_position_orientation()
                        pos_diff = th.from_numpy(np.array([0.01, -0.01, 0.0])).float()
                        pos += pos_diff
                        obj.set_position_orientation(pos, orn)
            else:
                pass

        # D1 is the distribution with randomization in xy-pos and z-rot
        elif self.name.endswith("D1"):
            task_relevant_objs = self._get_task_relevant_objs()
            self._randomize_object_pose_D1(task_relevant_objs)

            # Step one time to update the scene and render a few times as well
            og.sim.step()
            for _ in range(5):
                og.sim.render()

            # Update the observation
            obs, info = self.env.get_obs()
        
        elif self.name.endswith("D2"):
            # for arm role change
            task_relevant_objs = self._get_task_relevant_objs()
            self._randomize_object_pose_D2(task_relevant_objs)

            # Step one time to update the scene and render a few times as well
            og.sim.step()
            for _ in range(5):
                og.sim.render()

            # Update the observation
            obs, info = self.env.get_obs()
        
        elif self.name.endswith("D3"):
            # for arm role change
            task_relevant_objs = self._get_task_relevant_objs()
            self._randomize_object_pose_D2(task_relevant_objs)

            # Step one time to update the scene and render a few times as well
            og.sim.step()
            for _ in range(5):
                og.sim.render()

            # Update the observation
            obs, info = self.env.get_obs()

        else:
            raise ValueError(f"Unknown environment name: {self.name}")
        
        # change to the new observation
        obs = self.get_obs_IL()

        return obs
    
    def wrap_env(self):
        self.env = create_wrapper(env=self.env)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        og.sim.load_state(th.from_numpy(state["states"]), serialized=True)
        og.sim.step()

        return self.get_obs_IL()
        # return self.env.get_obs()[0]

    # TODO: implement the case of "rgb_array" mode correctly, e.g. return the rendered image as a numpy array
    def render(self, mode="human", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        if mode == "human":
            og.sim.render()
        elif mode == "rgb_array":
            # # Add external sensor observations if they exist
            # if self.env._external_sensors is not None:
            #     external_obs = dict()
            #     external_info = dict()
            #     for sensor_name, sensor in self.env._external_sensors.items():
            #         external_obs[sensor_name], external_info[sensor_name] = sensor.get_obs()
            #     obs_sensor = external_obs
            # img_dim_4 = obs_sensor['external_sensor0']['rgb']
            # img = img_dim_4[:, :, :3] # 128x128x4 -> 128x128x3

            img_dim_4 = og.sim.viewer_camera._get_obs()[0]['rgb']
            img = img_dim_4[:, :, :3] # 720x1280x4 -> 720x1280x3
            img = np.array(img, dtype=np.uint8)
            return img
        else:
            return np.zeros((height if height else 128, width if width else 128, 3), dtype=np.uint8)

    def customize_physical_properties(self):
        """
        Setup the mass, friction specifically for each task
        """
        if self.name.startswith("test_r1_cup"):
            # Increase gripper friction
            state = og.sim.dump_state()
            og.sim.stop()
            target_friction = 2.0
            gripper_mat = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
                prim_path=f"{self.env.robots[0].prim_path}/gripper_mat",
                name="gripper_material",
                static_friction=target_friction,
                dynamic_friction=target_friction,
                restitution=None,
            )
            for links in self.env.robots[0].finger_links.values():
                for link in links:
                    for msh in link.collision_meshes.values():
                        msh.apply_physics_material(gripper_mat)
            og.sim.play()
            og.sim.load_state(state)

            print('finish setting up the gripper friction in test_r1')
        
        elif self.name.startswith("test_tiago_cup"):

            state = og.sim.dump_state()

            # change the density and friction of objects
            # TODO: uncomment the necessary parts
            # notebook = env.env.scene.object_registry("name", "notebook")
            # notebook.links['base_link'].density = 10

            # coffee_cup.links['base_link'].friction = 0.01 # friction is not in the link object

            # giftbox = env.scene.object_registry("name", "gift_box")
            # giftbox.links['base_link'].density = 100

            # coffee_cup = env.env.scene.object_registry("name", "coffee_cup")
            # coffee_cup.links['base_link'].density = 30

            # paper_cup = env.env.scene.object_registry("name", "paper_cup")
            # paper_cup.links['base_link'].density = 100

            og.sim.play()
            og.sim.load_state(state)
            print('finish setting up the density of objects in test_tiago_cup')
            breakpoint()

    def sensor_setup(self):
        """
        Setup the sensor position, orientation of the environment
        """
        if self.name.startswith("test_r1_cup"):

            # customize viewer for this task specifically
            # TODO: if use third person view, have to provide customized camera sensot for each
            sensor = og.sim.viewer_camera
            sensor.set_position_orientation(
                position=th.tensor([ 2.7668, -0.0084,  1.9879]),
                orientation=th.tensor([0.3260, 0.3297, 0.6300, 0.6229]),
            ) # viewer position
            sensor.add_modality("depth_linear")
            sensor.add_modality("rgb")
            sensor.image_height = 196
            sensor.image_width = 320

            # print('sensor intrinsic matrix', sensor.intrinsic_matrix)
            # print('sensor pose', sensor.get_position_orientation())
            self.K = np.array([
                [259.6039,   0.0000, 160.0000],
                [  0.0000, 278.5423,  98.0000],
                [  0.0000,   0.0000,   1.000]
                ]) 
            self.camera_position = th.tensor([ 2.7668, -0.0084,  1.9879])
            self.camera_quat = th.tensor([0.3260, 0.3297, 0.6300, 0.6229])
            self.world_to_cam_tf = T.pose2mat((self.camera_position, self.camera_quat)).numpy()
            self.sensor_max_depth = 4.0
            self.number_ponits_to_sample = 4096
            self.pcd_offset = np.array([ -4.116, 0.002,  -3.069])
            self.pcd_norm_range = np.array([0.9, 0.9, 0.9])
            self.clip_bbox_size = np.array([3, 1.5, 2])

            # TODO: maybe reduce the pcd range can be helpful
            # self.pcd_norm_range = np.array([1.0, 1.0, 1.0])
            # self.clip_bbox_size = np.array([2.5, 1.5, 1])
            
            # change the viewport output to the viewer camera
            viewer_prim_path = og.sim.viewer_camera.prim_path
            og.sim.viewer_camera.active_camera_path = viewer_prim_path #'/World/viewer_camera'

            # change external sensor 0 pose and resolution
            ext_sensor = self.env._external_sensors['external_sensor0']
            ext_sensor.set_position_orientation(
                position=th.tensor([ 1.7330, -0.0486,  1.5626]),
                orientation=th.tensor([0.3689, 0.3718, 0.6047, 0.5999]),
            )
            ext_sensor.add_modality("depth_linear")

            sensor_pose = sensor.get_position_orientation()
            # sensor_pose = th.cat([sensor_pose[0], sensor_pose[1]])
            sensor_info = {
                "pos": sensor_pose[0],
                "quat": sensor_pose[1],
                "K": sensor.intrinsic_matrix,
                "world_to_cam_tf": self.world_to_cam_tf,
                "image_height": sensor.image_height,
                "image_width": sensor.image_width,
                'sensor_max_depth': self.sensor_max_depth,
                'number_points_to_sample': self.number_ponits_to_sample,
                'pcd_offset': self.pcd_offset,
                'pcd_norm_range': self.pcd_norm_range,
                'clip_bbox_size': self.clip_bbox_size
            }

    
        elif self.name.startswith("test_tiago_cup"):

            self.K = np.array([
                [259.6039,   0.0000, 160.0000],
                [  0.0000, 280.2977,  90.0000],
                [  0.0000,   0.0000,   1.0000]
                ])
            self.camera_position = th.tensor([ 1.0304, -0.0309,  1.0272])
            self.camera_quat= th.tensor([0.2690, 0.2659, 0.6509, 0.6583])
            self.world_to_cam_tf = T.pose2mat((self.camera_position, self.camera_quat)).numpy()
            self.sensor_max_depth = 2.0
            self.number_ponits_to_sample = 2048


        else:
            # TODO: need to do sensor setup for other environments, check the incoming repo change
            print("Unknown environment name: ", self.name)
            breakpoint()
        return sensor_info

    def process_point_cloud(self, obs):
        """
        Get point cloud from the environment
        """
        if self.name.startswith("test_tiago_cup"):
            depth_img = obs['external::external_sensor0::depth_linear']
            raise ValueError(f"need customize to the new compute_point_cloud_from_rgbd")
            
        
        elif self.name.startswith("test_r1_cup"):
            # compute_pcd_time = time.time()
            for key in obs.keys():
                if 'depth_linear' in key:
                    # print('depth key', key)
                    depth = obs[key]
                elif 'rgb' in key:
                    # print('rgb key', key)
                    rgb = obs[key]
            rgbd = np.concatenate([rgb, depth[:,:,None]], axis=-1)
            pointcloud = compute_point_cloud_from_rgbd(
                rgbd=rgbd, 
                K=self.K, 
                pcd_offset=self.pcd_offset,
                pcd_norm_range=self.pcd_norm_range,
                clip_bbox_size=self.clip_bbox_size,
                cam_to_img_tf=None, 
                world_to_cam_tf=self.world_to_cam_tf, 
                pcd_step_vis=False, 
                max_depth=self.sensor_max_depth,
                sample_type='fps',
                num_points_to_sample=self.number_ponits_to_sample,
                clip_scene=True,
                with_color=self.with_color
                )
        
        return pointcloud

    def process_prop(self, obs):
        # base_qpos = obs['base_qpos'] #  3
        base_qvel = obs['base_qvel'] # 3
        trunk_qpos = obs['trunk_qpos'] # 4
        arm_left_qpos = obs['arm_left_qpos'] #  6
        arm_right_qpos = obs['arm_right_qpos'] #  6
        left_gripper_width = obs['gripper_left_qpos'].sum()[None] # 1
        right_gripper_width = obs['gripper_right_qpos'].sum()[None] # 1
        prop_state = np.concatenate((base_qvel, trunk_qpos, arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width)) # 21
        if 'r1' in self.name: assert prop_state.shape[0] == 21
        return prop_state

    def process_eef(self, obs):
        eef_left_pos = obs['eef_left_pos'] # 3
        eef_right_pos = obs['eef_right_pos'] # 3
        eef_left_quat = obs['eef_left_quat'] # 4
        eef_right_quat = obs['eef_right_quat'] # 4
        eef_state = np.concatenate((eef_left_pos, eef_right_pos, eef_left_quat, eef_right_quat)) # 14
        if 'r1' in self.name: assert eef_state.shape[0] == 14 # for r1 robot
        return eef_state
    
    def process_prop_eef(self, obs):
        # base_qpos = obs['base_qpos'] #  3
        base_qvel = obs['base_qvel'] # 3
        trunk_qpos = obs['trunk_qpos'] # 4
        arm_left_qpos = obs['arm_left_qpos'] #  6
        eef_left_pos = obs['eef_left_pos'] # 3
        eef_left_quat = obs['eef_left_quat'] # 4
        left_gripper_width = obs['gripper_left_qpos'].sum()[None] # 1
        arm_right_qpos = obs['arm_right_qpos'] #  6
        eef_right_pos = obs['eef_right_pos'] # 3
        eef_right_quat = obs['eef_right_quat'] # 4
        right_gripper_width = obs['gripper_right_qpos'].sum()[None] # 1

        prop_eef_state = np.concatenate((base_qvel, trunk_qpos, 
                                     arm_left_qpos, eef_left_pos, eef_left_quat, left_gripper_width, 
                                     arm_right_qpos, eef_right_pos, eef_right_quat, right_gripper_width)) # 35
        if 'r1' in self.name: assert prop_eef_state.shape[0] == 35 # for r1 robot
        return prop_eef_state

    def process_prop_eef_basepose(self, obs):
        base_qpos = obs['base_qpos'] #  3
        base_qvel = obs['base_qvel'] # 3
        trunk_qpos = obs['trunk_qpos'] # 4
        arm_left_qpos = obs['arm_left_qpos'] #  6
        eef_left_pos = obs['eef_left_pos'] # 3
        eef_left_quat = obs['eef_left_quat'] # 4
        left_gripper_width = obs['gripper_left_qpos'].sum()[None] # 1
        arm_right_qpos = obs['arm_right_qpos'] #  6
        eef_right_pos = obs['eef_right_pos'] # 3
        eef_right_quat = obs['eef_right_quat'] # 4
        right_gripper_width = obs['gripper_right_qpos'].sum()[None] # 1

        prop_eef_basepose_state = np.concatenate((base_qpos, base_qvel, trunk_qpos, 
                                     arm_left_qpos, eef_left_pos, eef_left_quat, left_gripper_width, 
                                     arm_right_qpos, eef_right_pos, eef_right_quat, right_gripper_width)) # 38
        if 'r1' in self.name: assert prop_eef_basepose_state.shape[0] == 38 # for r1 robot
        return prop_eef_basepose_state


    def process_base_vel_robot_frame(self, robot_prop_states):
        base_vel = copy.deepcopy(robot_prop_states['base_qvel'])
        base_vel_xy = base_vel[:2]
        base_vel_z = base_vel[2] # rotation along z-axis should not be changed
        base_vel_vec = th.cat([base_vel_xy, th.zeros(1)]) # 3
        base_vel_ori = th.Tensor([0, 0, 0, 1]) # 4
        base_link_pose = self.env.robots[0].get_position_orientation()
        # TODO: construct the frame attached to the base velocity 
        base_vel_vec_local, base_vel_ori_local = T.relative_pose_transform(base_vel_vec + base_link_pose[0], base_vel_ori, *base_link_pose)
        print('original base vel', base_vel_xy, 'base vel norm', th.norm(base_vel_xy))
        print('base vel in robot frame', base_vel_vec_local[:2], 'local base vel norm', th.norm(base_vel_vec_local))
        breakpoint()
        base_vel_local = th.cat([base_vel_vec_local[:2], th.Tensor([base_vel_z])]) 
        robot_prop_states['base_vel'] = base_vel_local
        print('breakpoint in transform the base vel to robot frame')
        print('TODO: still need to handle the z axis velocity, what does the position mean in the controller???')
        breakpoint()
        return robot_prop_states

    def process_obj_robot_frame(self):
        # process object states, tranform them into robot fixed frames

        base_link_pose = self.env.robots[0].get_position_orientation()
        if self.name.startswith("test_r1_cup"):
            # TODO: only work for test r1 cup, which is a dummy task
            obj_states = {}
            obj_list = []
            obj_list.append(self.env.scene.object_registry("name", "coffee_cup"))
            obj_list.append(self.env.scene.object_registry("name", "teacup"))
            # obj_list = [self.env.scene.object_registry("name", name) for name in ["coffee_cup", "teacup"]]
            for obj in obj_list:
                obj_name = "object::"+obj.name
                pos, ori = obj.get_position_orientation()
                local_pos, local_ori = T.relative_pose_transform(pos, ori, *base_link_pose)
                obj_states[obj_name] = np.concatenate([local_pos, local_ori])
        
        else:
            # get object states for tasks that are not dummy tasks
            obj_states = {}
            obj_bddl_names = [obj.bddl_inst for obj in self.env._task.object_scope.values()] # get object names
            for obj_name in obj_bddl_names:
                # TODO: here not checking whether the object exist in the scene, may need to handle this silimar to omnigibson/tasks/behavior_task.py
                pos, ori = self.env.task.object_scope[obj_name].get_position_orientation()
                local_pos, local_ori = T.relative_pose_transform(pos, ori, *base_link_pose)
                if 'agent' not in obj_name and 'robot' not in obj_name:
                    # remove the .n.01_1 suffix and only keep the object name
                    obj_name = "object::"+obj_name.split('.')[0]
                obj_states[obj_name] = np.concatenate([local_pos, local_ori])
        
        return obj_states

    def get_obs_IL(self, di=None):
        """
        Get observation for IL baselines
         - robot proprioceptive state
         - objects in the scene and their states
         - default observations
        """

        # customize observation for IL baselines
        obs_IL = {}

        obj_states = self.process_obj_robot_frame()
        obs_IL.update(obj_states)

        other_obs = self.get_observation(di) # get default observations
        obs_IL.update(other_obs)

        # TODO: after pulling the latest, the sensor camera info is not in other_obs
        if self.name.startswith("test_r1_cup"):
            # TODO: need to add back the camera info manually
            sensor_obs = og.sim.viewer_camera.get_obs()[0]
            sensor_obs = {'external::viewer::'+k:v for k,v in sensor_obs.items()}
            # external_sensor_obs = self.env._external_sensors['external_sensor0'].get_obs()[0]
            # external_sensor_obs = {'external::external_sensor0::'+k:v for k,v in external_sensor_obs.items()}
            # for k, v in sensor_obs.items():
            #     print('obs shape', k, v.shape)
            # obs_IL.update(sensor_obs)
        else:
            raise ValueError(f"need to customize camera sensor information for task: {self.name}")
        obs_IL.update(sensor_obs)
        # add point cloud only when policy rollout
        # policy_pcd_process_start_time = time.time()
        if self.policy_rollout:
            pcd = self.process_point_cloud(sensor_obs)
            if self.with_color:
                obs_IL['combined::color_point_cloud'] = pcd
            else:
                obs_IL['combined::point_cloud'] = pcd
        # print('policy_pcd_process_time', time.time()-policy_pcd_process_start_time)
        

        robot_prop_states = self.env.robots[0]._get_proprioception_dict()
        # TODO: need to add the base velocity in the robot frame
        # robot_prop_states = self.process_base_vel_robot_frame(robot_prop_states)
        print('check base vel')
        breakpoint()

        obs_IL.update(robot_prop_states)

        base_link_pose = self.env.robots[0].get_position_orientation()
        obs_IL.update({'base_link_pose': np.concatenate([base_link_pose[0], base_link_pose[1]])})
        print('breakpoint for update base link pose')
        breakpoint()

        prop_state = {'prop_state': self.process_prop(robot_prop_states)}
        obs_IL.update(prop_state)

        prop_eef_state = {'prop_eef_state': self.process_prop_eef(robot_prop_states)}
        obs_IL.update(prop_eef_state)

        prop_eef_basepose = {'prop_eef_basepose': self.process_prop_eef_basepose(robot_prop_states)}
        obs_IL.update(prop_eef_basepose)

        # eef_state = {'eef_state': self.process_eef(robot_prop_states)}
        # obs_IL.update(eef_state)

        return obs_IL

    def get_subtask_success_signal(self):
        # TODO: need to implement the subtask success signal
        subtask_signals = dict()
        if 'test_tiago_cup' in self.name:

            # the subtask signals are 
            # whether the coffee cup is grasped
            # whether the coffee cup is placed on the table
            # whether the paper cup is grasped
            
            # grasping logic:
            # TRUE = 1
            # UNKNOWN = 0
            # FALSE = -1

            # print('breakpoint in env_omnigibson get_subtask_success_signal')
            # breakpoint()

            # # TODO: need to change the logic 
            # subtask_signals["grasp_right"] = abs(int(self.robot.is_grasping(arm="right", candidate_obj=self.env.task.object_scope["coffee_cup.n.01_1"])))
            # subtask_signals["ungrasp_right"] = abs(1 - abs(int(self.robot.is_grasping(arm="right", candidate_obj=self.env.task.object_scope["coffee_cup.n.01_1"]))))

            # subtask_signals["grasp_left"] = abs(int(self.robot.is_grasping(arm="left", candidate_obj=self.env.task.object_scope["dixie_cup.n.01_1"])))
            # subtask_signals["ungrasp_left"] = abs(1-abs(int(self.robot.is_grasping(arm="left", candidate_obj=self.env.task.object_scope["dixie_cup.n.01_1"]))))
            
            # print('task name', self.name, 'subtask success signal not implemented')
            pass
        
        else: 
            # print('task name', self.name, 'subtask success signal not implemented')
            pass

        return subtask_signals
    
    def get_observation_list_IL(self):
        # return the list of observation keys for IL baselines
        obs = self.get_obs_IL()
        obs_list = list(obs.keys())
        return obs_list
    
    def get_observation(self, di=None):
        if di:
            return di

        obs, info = self.env.get_obs()
        return obs

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        state = og.sim.dump_state(serialized=True)
        return dict(states=state)

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        if self.name.startswith("test_r1_cup"):
            ontop = False
            coffee_cup_pos, _ = self.env.scene.object_registry("name", "coffee_cup").get_position_orientation()
            teacup_pos, _ = self.env.scene.object_registry("name", "teacup").get_position_orientation()
            # check whether the coffee cup is on top of the teacup
            xy_diff = np.linalg.norm(coffee_cup_pos[:2] - teacup_pos[:2])
            z_diff = np.abs(coffee_cup_pos[2] - teacup_pos[2])
            if xy_diff < 0.03 and z_diff < 0.03: ontop=True        
            if ontop:
                print('success detected', 'xy_diff', xy_diff, 'z_diff', z_diff)
            return {"task": ontop}
        return {"task": len(self.env.task._termination_conditions["predicate"].goal_status["unsatisfied"]) == 0}

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.OG_TYPE

    @property
    def version(self):
        """
        Returns version of robosuite used for this environment, eg. 1.2.0
        """
        return og.__version__

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(
            env_name=self.name,
            env_version=self.version,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs)
        )

    @classmethod
    def create_for_data_processing(
        cls,
        env_name,
        **kwargs,
    ):
        # Always flatten observation space for data processing
        kwargs["env"]["flatten_obs_space"] = True
        # breakpoint()
        return cls(env_name=env_name, **kwargs)

    @property
    def rollout_exceptions(self):
        return

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        return self.env

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)

    # Nothing below this is implemented yet - not needed for data generation
    def get_real_depth_map(self, depth_map):
        raise NotImplementedError

    def get_camera_intrinsic_matrix(self, camera_name, camera_height, camera_width):
        raise NotImplementedError

    def get_camera_extrinsic_matrix(self, camera_name):
        raise NotImplementedError

    def get_camera_transform_matrix(self, camera_name, camera_height, camera_width):
        raise NotImplementedError

    def get_reward(self):
        """
        Get current reward.
        """
        raise NotImplementedError

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        raise NotImplementedError

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        if 'tiago' in self.name:
            return 22
        elif 'r1' in self.name:
            return 21
        else:
            raise NotImplementedError
