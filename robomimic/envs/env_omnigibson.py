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
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

# from mimicgen.train_scripts.train_prep_data import compute_point_cloud_from_rgbd
from scipy.spatial.transform import Rotation as R
import fpsample
import open3d as o3d

from enum import Enum
import torch as th
import numpy as np
import gym
import time
import copy


from omnigibson.macros import gm

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False

DEBUG = True

class EnvErrTypes(str, Enum):
    ArmMPFailed = "ArmMPFailed"
    BaseMPFailed = "BaseMPFailed"
    BaseSamplingFailed = "BaseSamplingFailed"

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

def fps_downsample(color_pcd, num_points_to_sample):
    if color_pcd.shape[0] > num_points_to_sample:
        pc = color_pcd[:, 3:]
        color_img = color_pcd[:, :3]
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, num_points_to_sample, h=5)
        pc = pc[kdline_fps_samples_idx]
        color_img = color_img[kdline_fps_samples_idx]
        color_pcd = np.concatenate([color_img, pc], axis=-1)
    else:
        raise ValueError("color_pcd shape is smaller than num_points_to_sample")
    return color_pcd

def pcd_vis(pc):
    # visualize with open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3)) 
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])
    print('number points', pc.shape[0])

def color_pcd_vis(color_pcd):
    # visualize with open3D
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(color_pcd[:, :3])
    pcd.points = o3d.utility.Vector3dVector(color_pcd[:,3:]) 
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])
    print('number points', color_pcd.shape[0])

class EnvOmniGibson(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self,
        env_name,
        policy_rollout=False,
        manipulation_only=False,
        real_robot_mode=False,
        **kwargs,
    ):
        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)
        self.add_distractor_objects = False
        self.single_arm = "right"
        self.policy_rollout = policy_rollout
        self.with_color = True
        self.manipulation_only = manipulation_only
        self.real_robot_mode = real_robot_mode
        self.init_nav_manip = False
        self.debug_from_saved_state = False
        self.retract_type = "retract_to_canonical_pose_maintain_orn"     # Options: ["retract_to_canonical_pose", "retract_to_start_of_arm_mp"]

        if self.name.startswith("r1_pick_cup"):
            self.update_params_r1_pick_cup(kwargs)
        if self.name.startswith("r1_tidy_table"):
            self.update_params_r1_tidy_table(kwargs)
        if self.name.startswith("r1_dishes_away"):
            self.update_params_r1_dishes_away(kwargs)
        # Some general updates to kwargs. Always call this after the task specific updates in the previous lines
        self.update_kwargs(kwargs)

        if self.policy_rollout:
            # customize the environment for policy rollout
            # 1. rewrite the robot pos to the same one when teleoperation
            # kwargs["robots"][0]["position"] = robot_pos
            # kwargs["robots"][0]["orientation"] = robot_quat

            if self.name.startswith("r1_pick_cup"):
                # 2. set the bbox for ego-centric pcd range
                self.x_range = [0.0, 2.3]
                self.y_range = [-0.5, 0.5]
                self.z_range = [0.7, 2.0]

                # 3. speficy the table height, and cup heigth, and intrinsic matrix
                self.cup_mask_height = 0.95

                if self.manipulation_only:
                    self.intrinsic_matrix = np.array([
                        [174.08,   0.000, 128.000],
                        [  0.000, 174.08, 128.000],
                        [  0.000,   0.000,   1.000]])
                    self.table_mask_height = 0.755 # this is used when fps the pcd in two groups

                if self.init_nav_manip:
                    # the initial nav+manip 2 demos has differernt intrinsic matrix and resolution
                    self.intrinsic_matrix = np.array([
                        [87.04,   0.000, 64.000],
                        [  0.000, 87.04, 64.000],
                        [  0.000,   0.000,   1.000]])
                    self.table_mask_height = 0.77
                    
                    # TODO: hacky!!, this resolution only works for the first 2 demos got from mobile manipulation
                    kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_height"] = 128
                    kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_width"] = 128
                    

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

        # Perform any post env creation setup
        if self.name.startswith("r1_pick_cup"):
            self.update_env_post_creation_r1_pick_cup()

        # self.env.robots[0]._grasping_mode = "sticky"
        self.env.scene.update_initial_state()
        self.robot = self.env.robots[0]
        self.robot_name = self.env.robots[0].name

        self.customize_physical_properties()
        self.sensor_info = self.sensor_setup()

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

        # Call reset so that robot is set to its initial pose as curobo warmup (base) depends on that (due to the joint limits of the base)
        self.env.robots[0].set_position_orientation(position=th.tensor(self.reset_base_pose[0]), orientation=th.tensor(self.reset_base_pose[1]))
        for _ in range(5): og.sim.step()

        if self._init_kwargs['init_curobo']:
        # if not self.policy_rollout:
            # Head tracking with soft visibility constraint requires use_cuda_graph=False
            self.primitive = StarterSemanticActionPrimitives(
                self.env,
                self.env.robots[0],
                enable_head_tracking=self.enable_head_tracking,
                curobo_batch_size=6,
                # curobo_use_cuda_graph=not self.enable_head_tracking,
                curobo_use_cuda_graph=False,
                use_base_pose_hack=False,
                real_robot_mode=self.real_robot_mode,
            )

            # Create CuRobo instance
            self.cmg = self.primitive._motion_generator


        self.global_env_step = 0    

        # Obtain reset left and right eef pose and eyes pose
        self.left_eef_reset_pose_wrt_robot = self.robot.get_relative_eef_pose(arm="left") 
        self.right_eef_reset_pose_wrt_robot = self.robot.get_relative_eef_pose(arm="right") 
        eyes_reset_pose_wrt_world = self.robot.links["eyes"].get_position_orientation()
        robot_pose_wrt_world = self.robot.get_position_orientation()
        self.eyes_reset_pose_wrt_robot = T.mat2pose(th.linalg.inv(T.pose2mat(robot_pose_wrt_world)) @ T.pose2mat(eyes_reset_pose_wrt_world))


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
        elif self.name.startswith("r1_dishes_away"):
            return [self.env.scene.object_registry("name", name) for name in ["bar_gjeoer_0", "shelf_pfusrd_1", "plate_603", "plate_602", "plate_601"]]
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

        elif self.name.startswith("r1_pick_cup"):
            task_relevant_objs = self._get_task_relevant_objs()
            for obj in task_relevant_objs:
                if 'table' not in obj.name:
                    if obj.name == 'coffee_cup_7':
                        # print("")
                        # print('initial coffee cup position', obj.get_position_orientation())
                        obj.set_position_orientation(obj_poses["coffee_cup"][:3], obj_poses["coffee_cup"][3:])
                        # pose = [1.034, -0.1601,  0.7769]  
                        # 6.359e-03 -1.849e-04  9.996e-01 -2.692e-02
                        # print('after reset coffee cup position', obj.get_position_orientation())
                        # print("")
            for _ in range(5): og.sim.step()
            for _ in range(5): og.sim.render()
            print('finishe setting object pose for r1 robot')

    # TODO: make it more generalizable
    # randomize the pose of all the task relevant objects in xy-pos and z-rot
    def _randomize_object_pose_D0(self, objs):

        # Sampling random object poses using custom thresholds
        if self.name.startswith("r1_pick_cup"):
            pos_magnitude = [-0.15, 0.15] 
            rot_magnitude = np.pi / 12  # 15 degrees
        elif self.name.startswith("r1_dishes_away"):
            pos_magnitude = [-0.1, 0.1] 
            rot_magnitude = 0.01
        elif self.name.startswith("r1_tidy_table"):
            pos_magnitude = [-0.15, 0.15] 
            rot_magnitude = np.pi / 12  # 15 degrees


        for obj in objs:
            if all(keyword not in obj.name for keyword in ["table", "shelf", "bar", "sink"]):
                pos, orn = obj.get_position_orientation()
                state = og.sim.dump_state()
                while True:
                    pos_diff_xy = np.random.uniform(pos_magnitude[0], pos_magnitude[1], size=2)
                    pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
                    new_pos = pos + pos_diff
                    # TODO： without mobile motion， the target pose need to be very carefully selected
                    # pos += th.from_numpy(np.array([-.15, 0.0, 0]))
                    orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
                    new_orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))
                    obj.set_position_orientation(new_pos, new_orn)
                    for _ in range(10):
                        og.sim.step()
                    cond = self._get_relevant_initial_condition(obj)
                    assert cond is not None, f"Condition not found for object {obj.name}"
                    if cond.evaluate():
                        break
                    og.sim.load_state(state)

    def _get_relevant_initial_condition(self, obj):
        for cond in self.env.task.activity_initial_conditions:
            if self.env.task.object_scope[cond.body[1]].unwrapped == obj:
                # cond is a HEAD condition
                # cond.children[0] is the actual binary predicate
                return cond.children[0]
        return None

    def _randomize_object_pose_D1(self, objs):
        for obj in objs:
            if "table" not in obj.name:
                state = og.sim.dump_state()
                while True:
                    cond = self._get_relevant_initial_condition(obj)
                    assert cond is not None, f"Condition not found for object {obj.name}"
                    if cond.sample(True):
                        break
                    og.sim.load_state(state)

    # def _randomize_object_pose_D1(self, objs):
    #     # pos_magnitude = 0.10  # 5cm
    #     # rot_magnitude = np.pi / 12  # 15 degrees

    #     # # for debugging
    #     # # pos_magnitude = 0.001
    #     # # rot_magnitude = np.pi / 10000  # 15 degrees

    #     # for obj in objs:
    #     #     if "table" not in obj.name:
    #     #         pos, orn = obj.get_position_orientation()
    #     #         pos_diff_xy = np.random.uniform(-pos_magnitude, pos_magnitude, size=2)
    #     #         pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
    #     #         pos += pos_diff
    #     #         # TODO： without mobile motion， the target pose need to be very carefully selected
    #     #         pos += th.from_numpy(np.array([-.15, 0.0, 0]))
    #     #         orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
    #     #         orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))

    #     #         pos[1] = -pos[1] # mirror the position along the y-axis
    #     #         orn = T.mat2quat(T.euler2mat(th.tensor([0.0, 0.0, np.pi])) @ T.quat2mat(orn)) # add pi orientation along the y-axis
    #     #         obj.set_position_orientation(pos, orn)

    #     # # Randomize height of table
    #     # breakfast_table = self.env.scene.object_registry("name", "breakfast_table")
    #     # breakfast_table_current_scale = breakfast_table.scale
    #     # z_scale = np.random.uniform(0.8, 1.2)
    #     # # print(f"z_scale: {z_scale}")
    #     # temp_state = og.sim.dump_state(serialized=False)
    #     # og.sim.stop()
    #     # breakfast_table.scale = th.tensor([breakfast_table_current_scale[0], breakfast_table_current_scale[1], 1.0 * z_scale])
    #     # og.sim.play()
    #     # og.sim.load_state(temp_state)
    #     # breakfast_table.keep_still()
    #     # for _ in range(10): og.sim.step()

    #     # # debugging
    #     # coffee_cup = self.env.scene.object_registry("name", "coffee_cup")
    #     # x_pos = np.random.uniform(0.67, 0.71)
    #     # y_pos = np.random.uniform(-0.5, 0.5)
    #     # current_coffee_cup_pos = coffee_cup.get_position()
    #     # coffee_cup.set_position_orientation(position=th.tensor([x_pos, y_pos, 0.9]))

    #     # # Sampling random object poses on table using OG API
    #     # for obj in objs:
    #     #     if "table" not in obj.name:
    #     #         obj.states[object_states.OnTop].set_value(other=self.env.scene.object_registry("name", "breakfast_table"), new_value=True)

    #     bar = self.env.scene.object_registry("name", "bar_udatjt_0")
    #     bar_current_scale = bar.scale
    #     z_scale = 0.7
    #     # z_scale = np.random.uniform(0.8, 1.2)
    #     temp_state = og.sim.dump_state(serialized=False)
    #     og.sim.stop()
    #     bar.scale = th.tensor([bar_current_scale[0], bar_current_scale[1], 1.0 * z_scale])
    #     og.sim.play()
    #     og.sim.load_state(temp_state)
    #     bar.keep_still()
    #     bar.set_position_orientation(position=th.tensor([7.287, 0.189, 0.40]))
    #     for _ in range(10): og.sim.step()

    #     # For house_single_floor scene
    #     for obj in objs:
    #         if "table" not in obj.name:
    #             obj.states[object_states.OnTop].set_value(other=bar, new_value=True)

    #     # teacup = self.env.scene.object_registry("name", "teacup")
    #     # x_range = np.random.uniform(-0.2, 0.2)
    #     # teacup.set_position_orientation(position=th.tensor([ 6.700 + x_range, 0.024,  0.739]), orientation=th.tensor([    -0.000,      0.000,      0.858,      0.514]))

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

        if self.debug_from_saved_state:
            import pickle
            state = pickle.load(open("/home/arpit/test_projects/mimicgen/random_files/start_of_last_nav2.pickle", "rb"))
            og.sim.load_state(state)
            for _ in range(5): og.sim.step()
            fridge = self.env.scene.object_registry("name", "fridge_dszchb_0")
            self.env.scene.remove_object(obj=fridge)
            for _ in range(5): og.sim.step()

            self.cmg.update_obstacles()
        else:
            # Reset the robot to a specific pose (Note that this is different from the spawned pose because curobo requires robot to be spawned at origin)
            self.env.robots[0].set_position_orientation(position=th.tensor(self.reset_base_pose[0]), orientation=th.tensor(self.reset_base_pose[1]))

        # for static manipulation only
        if self.manipulation_only:
            if self.real_robot_mode:
                init_joint_pos = th.tensor([
                    0.5, -0.430, 0.004, 0.007, 0.007, 0.259, # Base
                    1.3, -2.3, -1.2, 0.0, # Torso
                    0.0, 0.0, 1.894, 1.894, -0.985, -0.985, 1.561, 1.562, 0.910, 0.910, -1.554, -1.554, # Arms
                    0.050, 0.050, 0.050, 0.050]) # Grippers
            else:
                init_joint_pos = th.tensor([
                    0.332, -0.430, 0.004, 0.007,  0.007, 0.259, # Base
                    1.427, -1.658, -0.543, 0.051, # Torso
                    -0.000, -0.000, 1.894, 1.894, -0.985, -0.985, 1.561, 1.562, 0.910, 0.910, -1.554, -1.554, # Arms
                    0.050, 0.050, 0.050, 0.050]) # Grippers
            self.robot.set_joint_positions(init_joint_pos)
            # self.env.robots[0].set_position_orientation(position=th.tensor([0.332, -0.430, 0.0]))
        # else:
        #     self.env.robots[0].set_position_orientation(position=th.tensor([-0.863, -0.26, 0.0]))

        # If loading a saved state, don't do randomization for all objects. Choose accodgin to what you want
        if self.debug_from_saved_state:
            pass
        
        # D0 is the distribution with randomization in xy-pos and z-rot
        elif self.name.endswith("D0"):
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
            self._randomize_object_pose_D1(task_relevant_objs)

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
            position=th.tensor([ 7.040, -1.375,  2.365]),
            orientation=th.tensor([0.382, 0.188, 0.400, 0.812]),
        )
        
        if self.policy_rollout:
            # customize the viewer camera for policy rollout
            ext_sensor = self.env._external_sensors['external_sensor2']
            ext_sensor.set_position_orientation(position=th.tensor([1.9230, -0.2432,  1.4854]), orientation=th.tensor([0.3403, 0.3626, 0.6326, 0.5937]),)
        
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
        """
        Setup the mass, friction specifically for each task
        """
        # Change the color of the robot to be black.
        for material in self.robot.materials:
            material.diffuse_color_constant = th.tensor([0.0, 0.0, 0.0])

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
        
        elif self.name.startswith("r1_pick_cup"):
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
        
        elif self.name.startswith("r1_dishes_away"):
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
        
        elif self.name.startswith("r1_tidy_table"):
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

        else:
            raise ValueError(f"Unknown environment name: {self.name}, need to customize the physical properties")
        
    def sensor_setup(self):
        """
        Setup the sensor position, orientation of the environment
        """
        all_sensor_info = {}
        for sensor_name, sensor in self.env.robots[0].sensors.items():
            # sensor = self.env.robots[0].sensors[f"{self.robot_name}:eyes:Camera:0"]
            # TODO: These are used in normalization of the point cloud, take a look at these values again!
            self.pcd_offset = np.array([0.0, 0.0, 0.0])
            self.pcd_norm_range = np.array([1.0, 1.0, 1.0])
            self.clip_bbox_size = np.array([10, 10, 10])
            self.world_to_cam_tf = np.eye(4)
            self.sensor_max_depth = 2.0
            self.number_ponits_to_sample = 4096

            # self.pcd_offset = np.array([ -4.116, 0.002,  -3.069])
            # self.pcd_norm_range = np.array([0.9, 0.9, 0.9])
            # self.clip_bbox_size = np.array([3, 1.5, 2])

            # # TODO: maybe reduce the pcd range can be helpful
            # # self.pcd_norm_range = np.array([1.0, 1.0, 1.0])
            # # self.clip_bbox_size = np.array([2.5, 1.5, 1])
            
            # # change the viewport output to the viewer camera
            # viewer_prim_path = og.sim.viewer_camera.prim_path
            # og.sim.viewer_camera.active_camera_path = viewer_prim_path #'/World/viewer_camera'

            # # change external sensor 0 pose and resolution
            # ext_sensor = self.env._external_sensors['external_sensor0']
            # ext_sensor.set_position_orientation(
            #     position=th.tensor([ 1.7330, -0.0486,  1.5626]),
            #     orientation=th.tensor([0.3689, 0.3718, 0.6047, 0.5999]),
            # )
            # ext_sensor.add_modality("depth_linear")

            sensor_info = {
                "K": sensor.intrinsic_matrix,
                "world_to_cam_tf": self.world_to_cam_tf,
                "image_height": sensor.image_height,
                "image_width": sensor.image_width,
                'sensor_max_depth': self.sensor_max_depth,
                'number_points_to_sample': self.number_ponits_to_sample,
                'pcd_offset': self.pcd_offset,
                'pcd_norm_range': self.pcd_norm_range,
                'clip_bbox_size': self.clip_bbox_size,
            }
            all_sensor_info[sensor_name] = sensor_info

        return all_sensor_info
    
        # the following can be deleted in the future
        # elif self.name.startswith("test_tiago_cup"):

        #     self.K = np.array([
        #         [259.6039,   0.0000, 160.0000],
        #         [  0.0000, 280.2977,  90.0000],
        #         [  0.0000,   0.0000,   1.0000]
        #         ])
        #     self.camera_position = th.tensor([ 1.0304, -0.0309,  1.0272])
        #     self.camera_quat= th.tensor([0.2690, 0.2659, 0.6509, 0.6583])
        #     self.world_to_cam_tf = T.pose2mat((self.camera_position, self.camera_quat)).numpy()
        #     self.sensor_max_depth = 2.0
        #     self.number_ponits_to_sample = 2048

    def depth_to_pcd(
            self,
            depth,
            pose,
            base_link_pose,
            K,
            max_depth=2,
        ):

        # get the homogeneous transformation matrix from quaternion
        pos = pose[:3]
        quat = pose[3:]
        rot = R.from_quat(quat)  # scipy expects [x, y, z, w]
        rot_add = R.from_euler('x', np.pi).as_matrix() # handle the cam_to_img transformation
        rot_matrix = rot.as_matrix() @ rot_add   # 3x3 rotation matrix
        world_to_cam_tf = np.eye(4)
        world_to_cam_tf[:3, :3] = rot_matrix
        world_to_cam_tf[:3, 3] = pos

        # filter depth
        mask = depth > max_depth
        depth[mask] = 0
        h, w = depth.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij", sparse=False)
        assert depth.min() >= 0
        u = x
        v = y
        uv = np.dstack((u, v, np.ones_like(u))) # (img_width, img_height, 3)

        Kinv = np.linalg.inv(K)
        
        pc = depth.reshape(-1, 1) * (uv.reshape(-1, 3) @ Kinv.T)
        pc = pc.reshape(h, w, 3)
        pc = np.concatenate([pc.reshape(-1, 3), np.ones((h * w, 1))], axis=-1)  # shape (H*W, 4)

        world_to_robot_tf = T.pose2mat((th.from_numpy(base_link_pose[:3]), th.from_numpy(base_link_pose[3:]))).numpy()
        robot_to_world_tf = np.linalg.inv(world_to_robot_tf)
        pc = (pc @ world_to_cam_tf.T @ robot_to_world_tf.T)[:, :3].reshape(h, w, 3)

        return pc

    def process_fused_point_cloud(self, obs):
 
        base_link_pose = obs['base_link_pose'] # (7,)

        # TODO: now assuming the camera intrinsic matrix are the same for all the cameras!!

        eye_rgb = obs['robot_r1::robot_r1:eyes:Camera:0::rgb'][...,:3] # (resolution 0, resolution 1, 3)
        eye_depth = obs['robot_r1::robot_r1:eyes:Camera:0::depth_linear'] # (resolution 0, resolution 1, 1)
        eye_pose = obs['robot_r1:eyes:Camera:0_pose'] # (7,)
        eye_cam_pcd = self.depth_to_pcd(eye_depth, eye_pose, base_link_pose, self.intrinsic_matrix, max_depth=self.sensor_max_depth)
        eye_cam_rgbd = np.concatenate([eye_rgb/255.0, eye_cam_pcd], axis=-1).reshape(-1,6)

        left_cam_rgb = obs['robot_r1::robot_r1:left_eef_link:Camera:0::rgb'][...,:3]
        left_cam_depth = obs['robot_r1::robot_r1:left_eef_link:Camera:0::depth_linear']
        left_cam_pose = obs['robot_r1:left_eef_link:Camera:0_pose']
        left_cam_pcd = self.depth_to_pcd(left_cam_depth, left_cam_pose, base_link_pose, self.intrinsic_matrix, max_depth=self.sensor_max_depth)
        left_cam_rgbd = np.concatenate([left_cam_rgb/255.0, left_cam_pcd], axis=-1).reshape(-1,6)

        right_cam_rgb = obs['robot_r1::robot_r1:right_eef_link:Camera:0::rgb'][...,:3]
        right_cam_depth = obs['robot_r1::robot_r1:right_eef_link:Camera:0::depth_linear']
        right_cam_pose = obs['robot_r1:right_eef_link:Camera:0_pose']
        right_cam_pcd = self.depth_to_pcd(right_cam_depth, right_cam_pose, base_link_pose, self.intrinsic_matrix, max_depth=self.sensor_max_depth)
        right_cam_rgbd = np.concatenate([right_cam_rgb/255.0, right_cam_pcd], axis=-1).reshape(-1,6)
   
        color_pcd = np.concatenate([eye_cam_rgbd, left_cam_rgbd, right_cam_rgbd], axis=0)

        # clip point cloud with a bounding box
        mask = (color_pcd[:, 3] > self.x_range[0]) & (color_pcd[:, 3] < self.x_range[1]) & (
            color_pcd[:, 4] > self.y_range[0]) & (color_pcd[:, 4] < self.y_range[1]) & (
            color_pcd[:, 5] > self.z_range[0]) & (color_pcd[:, 5] < self.z_range[1])
        color_pcd = color_pcd[mask]
        
        # split the pcd based on table and not table and then do down sample differently
        table_mask = color_pcd[:, 5] < self.table_mask_height
        table_pcd = color_pcd[table_mask]
        not_table_pcd = color_pcd[~table_mask]
        table_ratio = 0.3
        table_samples = int(self.number_ponits_to_sample * table_ratio)
        not_table_samples = self.number_ponits_to_sample - table_samples

        if table_pcd.shape[0] < table_samples:
            print('table pcd shape is smaller than table samples')
            breakpoint()
        table_pcd_ds = fps_downsample(table_pcd, table_samples)
        not_table_pcd_ds = fps_downsample(not_table_pcd, not_table_samples)
        color_pcd = np.concatenate([table_pcd_ds, not_table_pcd_ds], axis=0)
         
        return color_pcd

    def process_point_cloud(self, obs):
        """
        Get point cloud from the environment
        """
        # compute_pcd_time = time.time()
        # breakpoint()

        if self.name.startswith("r1_pick_cup"):
            pointcloud = self.process_fused_point_cloud(obs)

        elif self.name.startswith("test_r1_cup"):
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
                K=self.intrinsic_matrix, 
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

        # add robot sensor poses
        for k in self.robot.sensors:
            sensor_pose = self.robot.sensors[k].get_position_orientation()
            obs_IL.update({f"{k}_pose": np.concatenate([sensor_pose[0], sensor_pose[1]])})

        
        # temp_start_time = time.time()
        robot_prop_states = self.env.robots[0]._get_proprioception_dict()
        # TODO: need to add the base velocity in the robot frame
        # robot_prop_states = self.process_base_vel_robot_frame(robot_prop_states)
        # print('check base vel')
        # breakpoint()

        obs_IL.update(robot_prop_states)
        # print("Time taken for getting obs and proprio: {:.2f} and {:.2f} seconds".format(obs_time, time.time() - temp_start_time))

        base_link_pose = self.env.robots[0].get_position_orientation()
        obs_IL.update({'base_link_pose': np.concatenate([base_link_pose[0], base_link_pose[1]])})

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

        eyes_pose = self.robot.links["eyes"].get_position_orientation()
        obs_IL.update({'eyes_pose': np.concatenate([eyes_pose[0], eyes_pose[1]])})

        if self.policy_rollout:
            pcd = self.process_point_cloud(obs_IL)
            if self.with_color:
                obs_IL['combined::color_point_cloud'] = pcd
            else:
                obs_IL['combined::point_cloud'] = pcd
        
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
        unsuccess_bddl = len(self.env.task._termination_conditions["predicate"].goal_status["unsatisfied"])
        success_bddl = unsuccess_bddl == 0
        result = {"task": success_bddl}

        # Additional success criteria
        if self.name.startswith("r1_pick_cup"):
            # # NOTE: Currently only using the final state to determine success. Verify satisfactory for all tasks.
            # teacup_obj = self.env.scene.object_registry("name", "teacup")
            coffee_cup_obj = self.env.scene.object_registry("name", "coffee_cup_7")
            # success = teacup_obj.states[object_states.Inside].get_value(coffee_cup_obj)

            # # if teacup is grasped
            # success = teacup_obj.states[object_states.Touching].get_value(other=self.env.robots[0])

            # # if coffee_cup is grasped
            success_touching = coffee_cup_obj.states[object_states.Touching].get_value(other=self.env.robots[0])
            # get coffee_cup object position
            coffee_cup_pos = coffee_cup_obj.get_position_orientation()[0][2]
            success_lift = coffee_cup_pos > 0.82
            result.update({
                "touching": success_touching,
                "bddl": success_bddl,
                "lift": success_lift,
            })
        # TODO: Implement this
        elif self.name.startswith("r1_dishes_away"):
            plate_601_obj = self.env.scene.object_registry("name", "plate_601")
            plate_602_obj = self.env.scene.object_registry("name", "plate_602")
            plate_603_obj = self.env.scene.object_registry("name", "plate_603")
            result.update({"bddl": False})

        return result

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
        
    def update_kwargs(self, kwargs):
        # RESOLUTION = (128, 450)
        RESOLUTION = (256, 256)

        # Explicity add the depth_linear and rgb modalities
        kwargs["robots"][0]["obs_modalities"].append("depth_linear")
        kwargs["robots"][0]["obs_modalities"].append("rgb")
        # kwargs["robots"][0]["obs_modalities"].append("seg_instance")
        
        # Setting the camera height and width here because setting it later causes issues
        kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_height"] = RESOLUTION[0]
        kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_width"] = RESOLUTION[1]
        kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["horizontal_aperture"] = 25.0

        # Untucked reset joint positions. The torso is different from the default R1 untucked position
        kwargs["robots"][0]["reset_joint_pos"] = [
                0.0000,
                0.0000,
                0.000,
                0.000,
                0.000,
                -0.0000, # 6 virtual base joint 
                1.375 if self.real_robot_mode else 0.5,
                -2.195 if self.real_robot_mode else -1.0,
                -0.96 if self.real_robot_mode else -0.8,
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
        # Tucked reset joint positions. The torso is different from the default R1 tucked position
        # kwargs["robots"][0]["reset_joint_pos"] = [
        #         0.0000,
        #         0.0000,
        #         0.000,
        #         0.000,
        #         0.000,
        #         -0.0000, # 6 virtual base joint 
        #         0.5,
        #         -1.0,
        #         -0.8,
        #         -0.0000, # 4 torso joints
        #         -0.000,
        #         0.000,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0,
        #         0.0500,
        #         0.0500,
        #         0.0500,
        #         0.0500,
        #     ]

        # Always spawn robot at the origin with no rotation (this is to be compatible with curobo)
        kwargs["robots"][0]["position"] = [0.0, 0.0, 0.0]
        kwargs["robots"][0]["orientation"] = [0.0, 0.0, 0.0, 1.0]

    def update_params_r1_pick_cup(self, kwargs):
        self.reset_base_pose = (th.tensor([-0.863, -0.26, 0]), th.tensor([0.0, 0.0, 0.0, 1.0]))

    
    def update_params_r1_tidy_table(self, kwargs):
        # kwargs["scene"] = {
        #     "type": "InteractiveTraversableScene",
        #     "scene_model": "house_single_floor",
        #     "load_room_instances": ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"],
        #     "not_load_object_categories": ["taboret", "fridge"],
        # }
        kwargs["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
        self.reset_base_pose = (kwargs["robots"][0]["position"], kwargs["robots"][0]["orientation"])

    
    def update_params_r1_dishes_away(self, kwargs):
        kwargs["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
        # For the task of dishes away, we don't load the fridge
        kwargs["scene"]["not_load_object_categories"] = ["fridge"]
        self.reset_base_pose = (kwargs["robots"][0]["position"], kwargs["robots"][0]["orientation"])

    def update_env_post_creation_r1_pick_cup(self):
        floor = self.env.scene.object_registry("name", "floors_ptwlei_0")
        # floor2 = self.env.scene.object_registry("name", "floors_ifmioj_0")
        # breakfast_table = self.env.scene.object_registry("name", "breakfast_table_6")
        temp_state = og.sim.dump_state(serialized=False)
        og.sim.stop()
        floor.scale = th.tensor([1.8, 1.0, 1.0])
        # floor2.scale = th.tensor([1.8, 1.0, 1.0])
        # breakfast_table.scale = th.tensor([1.668, 1.038, 0.994])
        og.sim.play()
        og.sim.load_state(temp_state)
        og.sim.step()
