"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import cv2
import json
import numpy as np
from copy import deepcopy

import omnigibson as og
import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB

import omnigibson.utils.transform_utils as T
from omnigibson import object_states
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives


import torch as th
import numpy as np


from omnigibson.macros import gm

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False

DEBUG = True

def hori_concatenate_image(images):
    # Ensure the images have the same height
    image1 = images[0]
    concatenated_image = image1
    for i in range(1, len(images)):
        image_i = images[i]
        if image1.shape[0] != image_i.shape[0]:
            # print("Images do not have the same height. Resizing the second image.")
            height = image1.shape[0]
            image_i = cv2.resize(image_i, (int(image_i.shape[1] * (height / image_i.shape[0])), height))

        # Concatenate the images side by side
        concatenated_image = np.concatenate((concatenated_image, image_i), axis=1)

    return concatenated_image


class EnvOmniGibson(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self,
        env_name,
        **kwargs,
    ):
        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)

        # Setting the objects (breakfast table, teacup, coffee_cup) to be more in the centre
        # Setting some default joint positions of the robot  
        kwargs["objects"][0]["position"] = [0.5, 0.0, 0.7]
        kwargs["objects"][1]["position"] = [0.5, 0.3, 0.8]
        kwargs["objects"][2]["position"] = [0.5, -0.2, 0.8]
        kwargs["robots"][0]["reset_joint_pos"][0] = -1.0
        kwargs["robots"][0]["reset_joint_pos"][10] = 0.0
        kwargs["robots"][0]["reset_joint_pos"][11] = 0.0
        kwargs["robots"][0]["obs_modalities"].append("depth_linear")
        # breakpoint()

        if og.sim is not None:
            og.sim.stop()
            og.clear()

        self.env = og.Environment(configs=kwargs)
        self.valid_env = True
        # TODO: uncomment the following lines for data generation.
        controller_config = {
            "base": {"name": "HolonomicBaseJointController", "motor_type": "position", "command_input_limits": None, "use_impedances": False},
            "trunk": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
            "arm_left": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
            "arm_right": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
            "gripper_left": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
            "gripper_right": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
            "camera": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
        }

        self.env.robots[0].reload_controllers(controller_config=controller_config)
        self.env.robots[0]._grasping_mode = "sticky"
        self.env.scene.update_initial_state()
        self.robot_name = self.env.robots[0].name

        # # remove later
        # breakpoint()

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

        self.primitive = StarterSemanticActionPrimitives(self.env, self.env.robots[0], enable_head_tracking=True)

        # Create CuRobo instance
        self.cmg = self.primitive._motion_generator

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
        obs, r, done, truncated, info = self.env.step(action)
        if video_writer:
            robot_name = self.env.robots[0].name
            ego_img = obs[f"{robot_name}::{robot_name}:eyes:Camera:0::rgb"]
            eef_left_img = obs[f"{robot_name}::{robot_name}:left_eef_link:Camera:0::rgb"]
            eef_right_img = obs[f"{robot_name}::{robot_name}:right_eef_link:Camera:0::rgb"]
            concatenated_img = hori_concatenate_image([ego_img, eef_left_img, eef_right_img])
            video_writer.append_data(concatenated_img)
        #     for env_idx, single_env in enumerate(self.env.envs):
        #         external_obs = single_env.external_sensors["external_sensor0"].get_obs()[0]["rgb"][:,:,:3].numpy()
        #         video_writer[env_idx].append_data(external_obs)
        return obs, r, done, info

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
        elif self.name.startswith("test_tiago_single_arm_cup"):
            return [self.env.scene.object_registry("name", name) for name in ["coffee_cup", "teacup", "breakfast_table"]]
        elif self.name.startswith("test_r1_cup"):
            return [self.env.scene.object_registry("name", name) for name in ["coffee_cup", "teacup", "breakfast_table"]]
        else:
            raise ValueError(f"Unknown environment name: {self.name}")

        return [self.env.task.object_scope[obj] for obj in obj_names]

    # TODO: make it more generalizable
    # randomize the pose of all the task relevant objects in xy-pos and z-rot
    def _randomize_object_pose(self, objs):

        # Sampling random object poses on table using OG API
        for obj in objs:
            if "table" not in obj.name:
                obj.states[object_states.OnTop].set_value(other=self.env.scene.object_registry("name", "breakfast_table"), new_value=True)

        # # Sampling random object poses on table using custom thresholds
        # pos_magnitude = [-0.1, 0.1] 
        # rot_magnitude = np.pi / 12  # 15 degrees

        # # for debugging
        # # pos_magnitude = 0.001
        # # rot_magnitude = np.pi / 10000  # 15 degrees

        # for obj in objs:
        #     if "table" not in obj.name:
        #         pos, orn = obj.get_position_orientation()
        #         pos_diff_xy = np.random.uniform(pos_magnitude[0], pos_magnitude[1], size=2)
        #         pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
        #         pos += pos_diff
        #         # TODO： without mobile motion， the target pose need to be very carefully selected
        #         # pos += th.from_numpy(np.array([-.15, 0.0, 0]))
        #         orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
        #         orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))
        #         obj.set_position_orientation(pos, orn)

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
        obs, info = self.env.reset()
        self.valid_env = True
        self.primitive.valid_env = True

        # Reset the robot to a specific position. Can remove this later
        self.env.robots[0].set_position_orientation(position=th.tensor([-1.0, 0.0, 0.0]))

        # D0 is the original distribution (no randomization at all - deterministic reset)
        if self.name.endswith("D0"):
            pass

        # D1 is the distribution with randomization in xy-pos and z-rot
        elif self.name.endswith("D1"):
            task_relevant_objs = self._get_task_relevant_objs()
            self._randomize_object_pose(task_relevant_objs)

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
        else:
            raise ValueError(f"Unknown environment name: {self.name}")

        return obs

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

        return self.env.get_obs()[0]

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
        else:
            return np.zeros((height if height else 128, width if width else 128, 3), dtype=np.uint8)
        
    def sensor_setup(self):
        """
        Setup the sensor position, orientation of the environment
        """
        sensor = self.env.robots[0].sensors[f"{self.robot_name}:eyes:Camera:0"]
        # sensor.image_height = 196
        # sensor.image_width = 320
        self.K = sensor.intrinsic_matrix
        # TODO: These are used in normalization of the point cloud, take a look at these values again!
        self.pcd_offset = np.array([0.0, 0.0, 0.0])
        self.pcd_norm_range = np.array([1.0, 1.0, 1.0])
        self.clip_bbox_size = np.array([10, 10, 10])
        self.world_to_cam_tf = np.eye(4)
        self.sensor_max_depth = 10.0
        self.number_ponits_to_sample = 4096

        sensor_info = {
            "K": self.K,
            "world_to_cam_tf": self.world_to_cam_tf,
            "image_height": sensor.image_height,
            "image_width": sensor.image_width,
            'sensor_max_depth': self.sensor_max_depth,
            'number_points_to_sample': self.number_ponits_to_sample,
            'pcd_offset': self.pcd_offset,
            'pcd_norm_range': self.pcd_norm_range,
            'clip_bbox_size': self.clip_bbox_size,
        }

        return sensor_info
    
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
    
    def get_obs_IL(self, di=None):
        """
        Get observation for IL baselines
         - robot proprioceptive state
         - objects in the scene and their states
         - default observations
        """

        # customize observation for IL baselines
        obs_IL = {}

        # obj_states = self.process_obj()
        # obs_IL.update(obj_states)

        other_obs = self.get_observation(di) # get default observations
        obs_IL.update(other_obs)
        
        robot_prop_states = self.env.robots[0]._get_proprioception_dict()
        obs_IL.update(robot_prop_states)

        prop_state = {'prop_state': self.process_prop(robot_prop_states)}
        obs_IL.update(prop_state)

        prop_eef_state = {'prop_eef_state': self.process_prop_eef(robot_prop_states)}
        obs_IL.update(prop_eef_state)

        prop_eef_basepose = {'prop_eef_basepose': self.process_prop_eef_basepose(robot_prop_states)}
        obs_IL.update(prop_eef_basepose)

        # eef_state = {'eef_state': self.process_eef(robot_prop_states)}
        # obs_IL.update(eef_state)

        return obs_IL

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

    # def is_success(self):
    #     """
    #     Check if the task condition(s) is reached. Should return a dictionary
    #     { str: bool } with at least a "task" key for the overall task success,
    #     and additional optional keys corresponding to other task criteria.
    #     """
    #     return {"task": len(self.env.task._termination_conditions["predicate"].goal_status["unsatisfied"]) == 0}
    
    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        # NOTE: Currently only using the final state to determine success. Verify satisfactory for all tasks.
        teacup_obj = self.env.scene.object_registry("name", "teacup")
        coffee_cup_obj = self.env.scene.object_registry("name", "coffee_cup")
        success = teacup_obj.states[object_states.Inside].get_value(coffee_cup_obj)
        return success

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
        raise NotImplementedError
