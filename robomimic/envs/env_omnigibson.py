"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy

import omnigibson as og
import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB
from omnigibson.envs import create_wrapper

import omnigibson.utils.transform_utils as T
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.action_primitives.curobo import CuroboEmbodimentSelection, CuRoboMotionGenerator


import torch as th
import numpy as np


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
            'base': {'name': 'JointController', 'use_delta_commands': False, "command_output_limits": "default"},
            'trunk': {'name': 'JointController', 'use_delta_commands': False, "command_output_limits": "default"},
            'arm_left': {'name': 'JointController', 'use_delta_commands': False, "command_output_limits": "default"},
            'arm_right': {'name': 'JointController', 'use_delta_commands': False, "command_output_limits": "default"},
            'gripper_left': {'name': 'MultiFingerGripperController', 'mode': 'binary'},
            'gripper_right': {'name': 'MultiFingerGripperController', 'mode': 'binary'},
            'camera': {'name': 'JointController', 'use_delta_commands': False, "command_output_limits": "default"},
        }
        self.env.robots[0].reload_controllers(controller_config=controller_config)
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
        self.cmg = CuRoboMotionGenerator(
            robot=self.env.robots[0],
            batch_size=1,
            use_cuda_graph=True,
        )

    def step(self, action):
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
        else:
            raise ValueError(f"Unknown environment name: {self.name}")

        return [self.env.task.object_scope[obj] for obj in obj_names]

    # TODO: make it more generalizable
    # randomize the pose of all the task relevant objects in xy-pos and z-rot
    def _randomize_object_pose(self, objs):
        pos_magnitude = 0.10  # 5cm
        rot_magnitude = np.pi / 12  # 15 degrees

        # for debugging
        # pos_magnitude = 0.001
        # rot_magnitude = np.pi / 10000  # 15 degrees

        for obj in objs:
            if 'table' not in obj.name:
                pos, orn = obj.get_position_orientation()
                pos_diff_xy = np.random.uniform(-pos_magnitude, pos_magnitude, size=2)
                pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
                pos += pos_diff
                # TODO： without mobile motion， the target pose need to be very carefully selected
                pos += th.from_numpy(np.array([-.15, 0.0, 0]))
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
            if 'table' not in obj.name:
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
