"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import cv2
import time
import json
import numpy as np
from copy import deepcopy

import omnigibson as og
import omnigibson.lazy as lazy
import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB

import omnigibson.utils.transform_utils as T
from omnigibson import object_states
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.objects.dataset_object import DatasetObject

from mimicgen.train_scripts.train_prep_data import compute_point_cloud_from_rgbd
from scipy.spatial.transform import Rotation as R

from enum import Enum
import torch as th
import numpy as np


from omnigibson.macros import gm

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False

DEBUG = False

class EnvErrTypes(str, Enum):
    ArmMPFailed = "ArmMPFailed"
    BaseMPFailed = "BaseMPFailed"
    BaseSamplingFailed = "BaseSamplingFailed"

def update_kwargs(kwargs):
    # RESOLUTION = (128, 450)
    RESOLUTION = (128, 128)

    # Explicity add the depth_linear and rgb modalities
    kwargs["robots"][0]["obs_modalities"].append("depth_linear")
    kwargs["robots"][0]["obs_modalities"].append("rgb")
    kwargs["robots"][0]["obs_modalities"].append("seg_instance")
    
    # Setting the camera height and width here because setting it later causes issues
    kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_height"] = RESOLUTION[0]
    kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_width"] = RESOLUTION[1]
    kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["horizontal_aperture"] = 25.0

    kwargs["robots"][0]["reset_joint_pos"] = [
            0.0000,
            0.0000,
            0.000,
            0.000,
            0.000,
            -0.0000, # 6 virtual base joint 
            0.5,
            -1.0,
            -0.8,
            -0.0000, # 4 torso joints
            -0.000,
            0.000,
            1.8944,
            1.8945,
            -0.9848,
            -0.9849,
            1.5612,
            1.5621,
            0.9097,
            0.9096,
            -1.5544,
            -1.5545,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
        ]

    # Always spawn robot at the origin with no rotation (this is to be compatible with curobo)
    kwargs["robots"][0]["position"] = [0.0, 0.0, 0.0]
    kwargs["robots"][0]["orientation"] = [0.0, 0.0, 0.0, 1.0]

def load_empty_scene(kwargs, add_distractor_objects=False):
    # RESOLUTION = (376, 1344)
    RESOLUTION = (128, 128)

    kwargs["scene"]["type"] = "Scene"
    # Setting the objects (breakfast table, teacup, coffee_cup) to be more in the centre
    # Setting some default joint positions of the robot  
    kwargs["objects"][0]["position"] = [1.0, 0.0, 0.7]
    kwargs["objects"][1]["position"] = [0.7, 0.3, 0.8]
    kwargs["objects"][2]["position"] = [0.7, -0.2, 0.8]
    kwargs["objects"][0]["scale"][0] = 1.5
    kwargs["robots"][0]["reset_joint_pos"][0] = -0.5
    # kwargs["robots"][0]["position"] = [-1.0, 0.0, 0.0]
    if kwargs["robots"][0]["type"] == "Tiago":
        kwargs["robots"][0]["reset_joint_pos"][10] = 0.0
        kwargs["robots"][0]["reset_joint_pos"][11] = 0.0
    if kwargs["robots"][0]["type"] == "R1":
        kwargs["robots"][0]["reset_joint_pos"][6:22] = [0.5, -1.0, -0.8, 0.0,     -0.141,      0.027,
            2.248,      2.550,     -0.983,     -1.416,      0.227,     -0.072,
            1.460,     -1.417,     -1.230,      1.214]
    # Explicity add the depth_linear and rgb modalities
    kwargs["robots"][0]["obs_modalities"].append("depth_linear")
    kwargs["robots"][0]["obs_modalities"].append("rgb")
    kwargs["robots"][0]["obs_modalities"].append("seg_instance")

    if kwargs["robots"][0]["type"] == "R1":
        # Setting the camera height and width here because setting it later causes issues
        kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_height"] = RESOLUTION[0]
        kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_width"] = RESOLUTION[1]
        kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["horizontal_aperture"] = 40.0

    if add_distractor_objects:
        kwargs["scene"]["load_object_categories"].append("straight_chair")

    return kwargs

def load_house_single_floor(kwargs):
    RESOLUTION = (256, 256)
    # kwargs["scene"] = {
    #     "type": "InteractiveTraversableScene",
    #     "scene_model": "house_single_floor",
    #     "load_room_instances": ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"],
    #     "not_load_object_categories": ["taboret", "fridge"],
    # }
    kwargs["robots"][0] = {
        "type": "R1",
        # "position": [9.0, 1.5,  1.0286],   # [5.2, -.8,  1.0286]
        # "orientation": [    -0.0000,      0.0000,      0.8734,     -0.4870],
        "name": "robot0",
        "action_normalize": False,
        "self_collisions": False,
        "obs_modalities": ["rgb", "depth_linear", "seg_instance"],
        # "default_reset_mode": "tuck",
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": RESOLUTION[0],
                    "image_width": RESOLUTION[1],
                    "horizontal_aperture": 40.0,
                },
            },
        },
        "reset_joint_pos": [
            0.0000,
            0.0000,
            0.000,
            0.000,
            0.000,
            -0.0000, # 6 virtual base joint 
            0.5,
            -1.0,
            -0.8,
            -0.0000, # 4 torso joints
            -0.000,
            0.000,
            1.8944,
            1.8945,
            -0.9848,
            -0.9849,
            1.5612,
            1.5621,
            0.9097,
            0.9096,
            -1.5544,
            -1.5545,
            0.0500,
            0.0500,
            0.0500,
            0.0500,
        ],
    }
    # # remove later
    # kwargs["objects"] = [
    #     DatasetObject(
    #         name="teacup_601",
    #         category="teacup",
    #         model="kccqwj",
    #         position=th.tensor([ 6.1515, -0.0625,  1.0921]),
    #         orientation=th.tensor([-1.3313e-06, -6.6665e-07, -6.6280e-01,  7.4880e-01])
    #     )
    # ]



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

    return np.array(concatenated_image)


class EnvOmniGibson(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self,
        env_name,
        **kwargs,
    ):
        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)
        self.add_distractor_objects = False
        self.single_arm = "right"

        breakpoint()
        update_kwargs(kwargs)
        # load_house_single_floor(kwargs)
        # load_empty_scene(kwargs)

        if og.sim is not None:
            og.sim.stop()
            og.clear()

        self.env = og.Environment(configs=kwargs)
        
        # Env parameters added by Arpit
        self.valid_env = True
        self.err = "None"
        self.obj_visible_at_start_of_manip = False
        self.IL_obs_keys = ["rgb", "depth_linear"]
        self.sampled_base_poses = {"failure": list(), "success": list()}
        
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
        # self.env.robots[0]._grasping_mode = "sticky"
        self.env.scene.update_initial_state()
        self.robot = self.env.robots[0]
        self.robot_name = self.env.robots[0].name

        self.customize_physical_properties()

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

        self.enable_head_tracking = False
        self.primitive = StarterSemanticActionPrimitives(self.env, self.env.robots[0], enable_head_tracking=self.enable_head_tracking, curobo_batch_size=10)

        # Create CuRobo instance
        self.cmg = self.primitive._motion_generator

        self.policy_rollout = False
        self.with_color = False

        self.global_env_step = 0

        # This is done only for the r1_pick_cup task
        if "r1_pick_cup" in env_name:
            floor = self.env.scene.object_registry("name", "floors_ptwlei_0")
            # floor2 = self.env.scene.object_registry("name", "floors_ifmioj_0")
            breakfast_table = self.env.scene.object_registry("name", "breakfast_table_6")
            temp_state = og.sim.dump_state(serialized=False)
            og.sim.stop()
            floor.scale = th.tensor([1.8, 1.0, 1.0])
            # floor2.scale = th.tensor([1.8, 1.0, 1.0])
            breakfast_table.scale = th.tensor([1.668, 1.038, 0.994])
            og.sim.play()
            og.sim.load_state(temp_state)
            for _ in range(10): og.sim.step()


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
            ego_img = obs[f"{robot_name}::{robot_name}:eyes:Camera:0::rgb"].numpy()[:, :, :3]
            # eef_left_img = obs[f"{robot_name}::{robot_name}:left_eef_link:Camera:0::rgb"]
            # eef_right_img = obs[f"{robot_name}::{robot_name}:right_eef_link:Camera:0::rgb"]
            viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'].numpy()[:, :, :3]
            concatenated_img = hori_concatenate_image([ego_img, viewer_img])
            video_writer.append_data(concatenated_img)
        #     for env_idx, single_env in enumerate(self.env.envs):
        #         external_obs = single_env.external_sensors["external_sensor0"].get_obs()[0]["rgb"][:,:,:3].numpy()
        #         video_writer[env_idx].append_data(external_obs)

        # replace the observation with newly added IL obs function 
        obs, obs_info = self.get_obs_IL()
        
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
        elif self.name.startswith("test_tiago_single_arm_cup"):
            return [self.env.scene.object_registry("name", name) for name in ["coffee_cup", "teacup", "breakfast_table"]]
        elif self.name.startswith("test_r1_cup"):
            return [self.env.scene.object_registry("name", name) for name in ["coffee_cup", "teacup", "breakfast_table"]]
        elif self.name.startswith("r1_put_away_cup"):
            return [self.env.scene.object_registry("name", name) for name in ["coffee_cup", "teacup", "breakfast_table"]]
        elif self.name.startswith("r1_tidy_table"):
            return [self.env.scene.object_registry("name", name) for name in ["teacup_601", "drop_in_sink_awvzkn_0"]]
        elif self.name.startswith("r1_pick_cup"):
            return [self.env.scene.object_registry("name", name) for name in ["coffee_cup_7", "breakfast_table_6"]]
        else:
            raise ValueError(f"Unknown environment name: {self.name}")

        return [self.env.task.object_scope[obj] for obj in obj_names]

    # TODO: make it more generalizable
    # randomize the pose of all the task relevant objects in xy-pos and z-rot
    def _randomize_object_pose_D0(self, objs):

        # Sampling random object poses on table using custom thresholds
        pos_magnitude = [-0.1, 0.1] 
        rot_magnitude = np.pi / 12  # 15 degrees

        # for debugging
        # pos_magnitude = 0.001
        # rot_magnitude = np.pi / 10000  # 15 degrees

        for obj in objs:
            if "table" not in obj.name:
                pos, orn = obj.get_position_orientation()
                pos_diff_xy = np.random.uniform(pos_magnitude[0], pos_magnitude[1], size=2)
                pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
                pos += pos_diff
                # TODO： without mobile motion， the target pose need to be very carefully selected
                # pos += th.from_numpy(np.array([-.15, 0.0, 0]))
                orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
                orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))
                obj.set_position_orientation(pos, orn)

    def _randomize_object_pose_D2(self, objs):
        # pos_magnitude = 0.10  # 5cm
        # rot_magnitude = np.pi / 12  # 15 degrees

        # # for debugging
        # # pos_magnitude = 0.001
        # # rot_magnitude = np.pi / 10000  # 15 degrees

        # for obj in objs:
        #     if "table" not in obj.name:
        #         pos, orn = obj.get_position_orientation()
        #         pos_diff_xy = np.random.uniform(-pos_magnitude, pos_magnitude, size=2)
        #         pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
        #         pos += pos_diff
        #         # TODO： without mobile motion， the target pose need to be very carefully selected
        #         pos += th.from_numpy(np.array([-.15, 0.0, 0]))
        #         orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
        #         orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))

        #         pos[1] = -pos[1] # mirror the position along the y-axis
        #         orn = T.mat2quat(T.euler2mat(th.tensor([0.0, 0.0, np.pi])) @ T.quat2mat(orn)) # add pi orientation along the y-axis
        #         obj.set_position_orientation(pos, orn)

        # # Randomize height of table
        # breakfast_table = self.env.scene.object_registry("name", "breakfast_table")
        # breakfast_table_current_scale = breakfast_table.scale
        # z_scale = np.random.uniform(0.8, 1.2)
        # # print(f"z_scale: {z_scale}")
        # temp_state = og.sim.dump_state(serialized=False)
        # og.sim.stop()
        # breakfast_table.scale = th.tensor([breakfast_table_current_scale[0], breakfast_table_current_scale[1], 1.0 * z_scale])
        # og.sim.play()
        # og.sim.load_state(temp_state)
        # breakfast_table.keep_still()
        # for _ in range(10): og.sim.step()

        # # debugging
        # coffee_cup = self.env.scene.object_registry("name", "coffee_cup")
        # x_pos = np.random.uniform(0.67, 0.71)
        # y_pos = np.random.uniform(-0.5, 0.5)
        # current_coffee_cup_pos = coffee_cup.get_position()
        # coffee_cup.set_position_orientation(position=th.tensor([x_pos, y_pos, 0.9]))
        
        # # Sampling random object poses on table using OG API
        # for obj in objs:
        #     if "table" not in obj.name:
        #         obj.states[object_states.OnTop].set_value(other=self.env.scene.object_registry("name", "breakfast_table"), new_value=True)

        bar = self.env.scene.object_registry("name", "bar_udatjt_0")
        bar_current_scale = bar.scale
        z_scale = 0.7
        # z_scale = np.random.uniform(0.8, 1.2)
        temp_state = og.sim.dump_state(serialized=False)
        og.sim.stop()
        bar.scale = th.tensor([bar_current_scale[0], bar_current_scale[1], 1.0 * z_scale])
        og.sim.play()
        og.sim.load_state(temp_state)
        bar.keep_still()
        bar.set_position_orientation(position=th.tensor([7.287, 0.189, 0.40]))
        for _ in range(10): og.sim.step()

        # For house_single_floor scene
        for obj in objs:
            if "table" not in obj.name:
                obj.states[object_states.OnTop].set_value(other=bar, new_value=True)

        # teacup = self.env.scene.object_registry("name", "teacup")
        # x_range = np.random.uniform(-0.2, 0.2)
        # teacup.set_position_orientation(position=th.tensor([ 6.700 + x_range, 0.024,  0.739]), orientation=th.tensor([    -0.000,      0.000,      0.858,      0.514]))


    # TODO: make it more generalizable
    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        obs, info = self.env.reset()
        self.global_env_step = 0
        if not self.policy_rollout:
            self.valid_env = True
            # self.primitive.valid_env = True
            self.primitive.mp_err = "None"
            self.err = "None"
            self.obj_visible_at_start_of_manip = False

        # # Reset the robot to a specific position. TODO: Make this general
        # self.env.robots[0].set_position_orientation(position=th.tensor([-0.5, 0.0, 0.0]))
        self.env.robots[0].set_position_orientation(position=th.tensor([-0.863, -0.26, 0]))


        # # stack cup task in house_single_floor scene
        # self.robot.set_position_orientation(position=th.tensor([9.0, 1.5,  0.2]), orientation=th.tensor([-0.0000, 0.0000, 0.8734, -0.4870]))
        # self.robot.set_joint_positions(th.tensor([-0.3681,  1.2081, -0.2686,  1.5397,  0.9159, -1.5726]), indices=self.robot.arm_control_idx["left"])
        # self.robot.set_joint_positions( th.tensor([0.3681,  1.2081, -0.2686,  1.5397,  0.9159, -1.5726]), indices=self.robot.arm_control_idx["right"])
        # self.robot.reset_joint_pos = th.tensor([
        #                 0.0000,
        #                 0.0000,
        #                 0.000,
        #                 0.000,
        #                 0.000,
        #                 -0.0000, # 6 virtual base joint 
        #                 0.5,
        #                 -1.0,
        #                 -0.8,
        #                 -0.0000, # 4 torso joints
        #                 -0.3681,
        #                 0.3681,
        #                 1.2081,
        #                 1.2081,
        #                 -0.2686,
        #                 -0.2686,
        #                 1.5397,
        #                 1.5397,
        #                 0.9159,
        #                 0.9159,
        #                 -1.5726,
        #                 -1.5726,
        #                 0.0500,
        #                 0.0500,
        #                 0.0500,
        #                 0.0500,
        #             ],)


        if self.add_distractor_objects:
            # Set chair poses
            chair_0 = self.env.scene.object_registry("name", "straight_chair_amgwaw_0")
            chair_0.set_position_orientation(position=th.tensor([-0.0725,  0.0028,  0.4485]), orientation=th.tensor([ 0.0016,  0.0020, -0.1448,  0.9895]))

            chair_1 = self.env.scene.object_registry("name", "straight_chair_amgwaw_1")
            chair_1.set_position_orientation(position=th.tensor([ 0.4266, -1.0887,  0.4484]), orientation=th.tensor([ 2.0414e-05, -1.7101e-03,  9.9991e-01, -1.3466e-02]))

            chair_3 = self.env.scene.object_registry("name", "straight_chair_eospnr_0")
            chair_3.set_position_orientation(position=th.tensor([-0.5871,  2.2136,  0.4930]))
            chair_4 = self.env.scene.object_registry("name", "straight_chair_eospnr_1")
            chair_4.set_position_orientation(position=th.tensor([-1.2552,  2.3325,  0.4930]))
            for _ in range(20): og.sim.step()

        # D0 is the distribution with randomization in xy-pos and z-rot
        if self.name.endswith("D0"):
            task_relevant_objs = self._get_task_relevant_objs()
            self._randomize_object_pose_D0(task_relevant_objs)

            # Step one time to update the scene and render a few times as well
            og.sim.step()
            for _ in range(5):
                og.sim.render()

            # Update the observation
            obs, info = self.env.get_obs()

        # D1 is randomization all over the furniture
        elif self.name.endswith("D1"):
            # # for arm role change
            task_relevant_objs = self._get_task_relevant_objs()
            self._randomize_object_pose_D2(task_relevant_objs)

            # Step one time to update the scene and render a few times as well
            og.sim.step()
            for _ in range(5):
                og.sim.render()

            # Update the observation
            obs, info = self.env.get_obs()
        
        # D2 has ranomization with obstacles
        elif self.name.endswith("D2"):
            pass
        else:
            raise ValueError(f"Unknown environment name: {self.name}")

        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([-3.0856,  0.1110,  3.4114]),
            orientation=th.tensor([-0.3543,  0.3566,  0.6132, -0.6093]),
        )
        
        for _ in range(50): og.sim.step()
        
        # change to the new observation
        obs, obs_info = self.get_obs_IL()
        
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
        # There is probably a bug in og.sim.load_state where sometimes the state is not loaded correctly.
        # Empirically, I see that this happens when the robot is in a collision state with the object that is not reset correctly
        # but I might be wrong. For some reason this fix works.
        table_obj = self.env.scene.object_registry("name", "breakfast_table") 
        table_obj.set_position_orientation(position=th.tensor([0.0, -2.0, 0.7]))
        for _ in range(20): og.sim.step()

        og.sim.load_state(th.from_numpy(state["states"]).to(th.float32), serialized=True)

        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([-3.0856,  0.1110,  3.4114]),
            orientation=th.tensor([-0.3543,  0.3566,  0.6132, -0.6093]),
        )
        
        for _ in range(20): og.sim.step()

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
        else:
            # return np.zeros((height if height else 128, width if width else 128, 3), dtype=np.uint8)
            robot_name = self.env.robots[0].name
            obs, info = self.env.get_obs()
            ego_img = obs[f"{robot_name}::{robot_name}:eyes:Camera:0::rgb"].numpy()[:, :, :3]
            viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'].numpy()[:, :, :3]
            # breakpoint()
            # eef_left_img = obs[f"{robot_name}::{robot_name}:left_eef_link:Camera:0::rgb"]
            # eef_right_img = obs[f"{robot_name}::{robot_name}:right_eef_link:Camera:0::rgb"]
            concatenated_img = hori_concatenate_image([ego_img, viewer_img])
            return concatenated_img
            # video_writer.append_data(concatenated_img)
        
    def customize_physical_properties(self):
        # breakpoint()
        # Increase gripper friction
        state = og.sim.dump_state()
        og.sim.stop()
        target_friction = 4.0
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
        
    def sensor_setup(self):
        """
        Setup the sensor position, orientation of the environment
        """
        sensor = self.env.robots[0].sensors[f"{self.robot_name}:eyes:Camera:0"]
        # sensor.image_height = 128
        # sensor.image_width = 128
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

        # left_eef_sensor = self.env.robots[0].sensors[f"{self.robot_name}:left_eef_link:Camera:0"]
        # left_eef_sensor.image_height = 200
        # left_eef_sensor.image_width = 200

        # right_eef_sensor = self.env.robots[0].sensors[f"{self.robot_name}:right_eef_link:Camera:0"]
        # right_eef_sensor.image_height = 200
        # right_eef_sensor.image_width = 200

        return sensor_info
    
    def process_point_cloud(self, obs):
        """
        Get point cloud from the environment
        """
        
        # compute_pcd_time = time.time()
        # breakpoint()
        for key in obs.keys():
            if 'eyes:Camera:0::depth_linear' in key:
                # print('depth key', key)
                depth = obs[key]
            elif 'eyes:Camera:0::rgb' in key:
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

        # temp_start_time = time.time()
        other_obs, info = self.get_observation(di) # get default observations
        # retain only the relevant obs keys for IL policy
        for k in other_obs.keys():
            if k.split("::")[-1] in self.IL_obs_keys:
                if "seg" in k:
                    obs_IL[k] = other_obs[k].cpu()
                    breakpoint()
                else:
                    obs_IL[k] = other_obs[k]
        # obs_IL.update(other_obs)
        # obs_time = time.time() - temp_start_time 

        if self.policy_rollout:
            pcd = self.process_point_cloud(other_obs)
            if self.with_color:
                obs_IL['combined::color_point_cloud'] = pcd
            else:
                obs_IL['combined::point_cloud'] = pcd
        
        # add robot sensor poses
        for k in self.robot.sensors:
            sensor_pose = self.robot.sensors[k].get_position_orientation()
            obs_IL.update({f"{k}_pose": np.concatenate([sensor_pose[0], sensor_pose[1]])})

        
        # temp_start_time = time.time()
        robot_prop_states = self.env.robots[0]._get_proprioception_dict()
        obs_IL.update(robot_prop_states)
        # print("Time taken for getting obs and proprio: {:.2f} and {:.2f} seconds".format(obs_time, time.time() - temp_start_time))

        prop_state = {'prop_state': self.process_prop(robot_prop_states)}
        obs_IL.update(prop_state)

        prop_eef_state = {'prop_eef_state': self.process_prop_eef(robot_prop_states)}
        obs_IL.update(prop_eef_state)

        prop_eef_basepose = {'prop_eef_basepose': self.process_prop_eef_basepose(robot_prop_states)}
        obs_IL.update(prop_eef_basepose)

        # eef_state = {'eef_state': self.process_eef(robot_prop_states)}
        # obs_IL.update(eef_state)

        base_link_pose = self.env.robots[0].get_position_orientation()
        obs_IL.update({'base_link_pose': np.concatenate([base_link_pose[0], base_link_pose[1]])})

        return obs_IL, info

    def get_observation(self, di=None):
        if di:
            return di

        obs, info = self.env.get_obs()
        return obs, info

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
        # # NOTE: Currently only using the final state to determine success. Verify satisfactory for all tasks.
        # teacup_obj = self.env.scene.object_registry("name", "teacup")
        coffee_cup_obj = self.env.scene.object_registry("name", "coffee_cup_7")
        # success = teacup_obj.states[object_states.Inside].get_value(coffee_cup_obj)
        
        # # if teacup is grasped
        # success = teacup_obj.states[object_states.Touching].get_value(other=self.env.robots[0])

        # # if coffee_cup is grasped
        success = coffee_cup_obj.states[object_states.Touching].get_value(other=self.env.robots[0])

        return {"task": success}

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
        if 'tiago' in self.name:
            return 22
        elif 'r1' in self.name:
            return 21
        else:
            raise NotImplementedError