import torch
import numpy as np
import json
import os
import sys
import pdb 
from scipy.spatial.transform import Rotation as R
import yaml
# from ur_env.rotations import pose2quat
from env.rotations import pose2quat

import argparse

from rekep.environment import R2D2Env
# from rekep.ik_solver_ur5 import UR5IKSolver
from rekep.ik_solver_piper import PiPERIKSolver
from rekep.subgoal_solver import SubgoalSolver
from rekep.path_solver import PathSolver
from rekep.visualizer import Visualizer
# from ur_env.ur5_env import RobotEnv
from env.piper_env import RobotEnv
from rekep.utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

# from r2d2_vision import R2D2Vision
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="xFormers is not available")

import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

@timer_decorator
class MainR2D2:
    def __init__(self, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # TODO Update the default reset_joint_pos for PIPER
        self.reset_joint_pos = np.array([-0.023413960133687794, -1.9976251761065882, 1.7851085662841797, 4.942904949188232, -1.5486105124102991, -1.5801880995379847])  # UR5 home position

        # self.vision = R2D2Vision(visualize=self.visualize)

        self.robot_env = RobotEnv(test_mode=True)
        self.env = R2D2Env(global_config['env'])
        
        ik_solver = PiPERIKSolver(
            reset_joint_pos=self.reset_joint_pos,
            world2robot_homo=None,
        )
        # initialize solvers
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, self.reset_joint_pos)
        self.path_solver = PathSolver(global_config['path_solver'], ik_solver, self.reset_joint_pos)
        self.visualizer = Visualizer(global_config['visualizer'])

        if visualize:
            self.visualizer = Visualizer(global_config['visualizer'])
            self.data_path = "/home/franka/R2D2_3dhat/images/current_images" # TODO ???

    @timer_decorator
    def perform_task(self, instruction, obj_list=None, rekep_program_dir=None):
        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        # obj_list = ['scissors']
        
        if rekep_program_dir is None:
            pass
            # realworld_rekep_program_dir = self.vision.perform_task(instruction, obj_list, data_path, 3)
        else:
            realworld_rekep_program_dir = rekep_program_dir
        # ====================================
        self._execute(realworld_rekep_program_dir)

    @timer_decorator
    def _execute(self, rekep_program_dir):
        # Load program info and constraints
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        
        # Register initial keypoints
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        
        # Load all stage constraints
        self.constraint_fns = self._load_constraints(rekep_program_dir)
        
        # bookkeeping of which keypoints can be moved in the optimization
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

        # Generate action sequences for all stages
        self.all_actions = []
        # pdb.set_trace()
        # Process each stage sequentially
        if 1:
            # Read stage from robot state file
            with open('./robot_state.json', 'r') as f:
                robot_state = json.load(f)
                stage = robot_state.get('rekep_stage', 1)  # !!! @Tianyou Default to stage 1 if not found
            # store robot state in rekep_program_dir
            with open(os.path.join(rekep_program_dir, f'robot_state_{stage}.json'), 'w') as f:
                json.dump(robot_state, f, indent=4)
                 # Get current state
            scene_keypoints = self.env.get_keypoint_positions()
            print(f"Camera frame keypoints: {scene_keypoints}")
            
            # Transform keypoints from camera to world coordinates
            scene_keypoints = self.transform_keypoints_to_world(scene_keypoints)
            print(f"World frame keypoints: {scene_keypoints}")
            
            self.keypoints = np.concatenate([[self.ur_get_ee_location()], scene_keypoints], axis=0)
            self.curr_ee_pose = self.ur_get_ee_pose()
            self.curr_joint_pos = self.ur_get_joint_pos()
            self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size']) # TODO ???
            self.collision_points = self.env.get_collision_points()
            
            # stage = input(f"Enter stage number (1-{self.program_info['num_stages']}): ")
            stage = int(stage)
            if stage > self.program_info['num_stages']:
                print(f"{bcolors.FAIL}Stage {stage} is out of bounds, skipping\n{bcolors.ENDC}")
                return
            self._update_stage(stage)
            # if self.stage > 1:
            #     path_constraints = self.constraint_fns[self.stage]['path']
            #     for constraints in path_constraints:
            #         violation = constraints(self.keypoints[0], self.keypoints[1:])
            #         if violation > self.config['constraint_tolerance']:
            #             # backtrack = True
            #             print(f"\n\n\nConstraint violation: {violation}\n\n\n")
            #             break
       
            # Generate actions for this stage
            next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter, maintain_orientation=True)
            print(f"Next subgoal: {next_subgoal}")
            next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
            self.first_iter = False


            # pdb.set_trace()
            # Add gripper actions based on stage type
            # True or False from metadata.json
            # if self.is_grasp_stage: 
            #     next_path[-1, 7] = self.env.get_gripper_close_action() # Todo aliagn shape?
                
            # elif self.is_release_stage:
            #     next_path[-1, 7] = self.env.get_gripper_open_action() 
                
            self.all_actions.append(next_path)

        # Combine all action sequences
        combined_actions = np.concatenate(self.all_actions, axis=0)

        # # 计算夹爪偏移量 (在保存到action.json前减去偏移)
        # for i in range(len(combined_actions)):
        #     # 获取当前位姿 (x, y, z, qx, qy, qz, qw, gripper)
        #     pose = combined_actions[i]
        #     position = pose[:3]
        #     quat = pose[3:7]  # 四元数 [qx, qy, qz, qw]
            
        #     # 从四元数创建旋转矩阵
        #     rotation_matrix = R.from_quat(quat).as_matrix()
            
        #     # 应用手性校正
        #     rot_correct = np.array([
        #         [-1, 0, 0],
        #         [0, -1, 0],
        #         [0, 0, 1]
        #     ])
        #     rotation_corrected = rotation_matrix @ rot_correct
            
        #     # 计算沿校正后EE z轴的偏移
        #     z_offset = np.array([0, 0, 0.16])  # z轴方向0.16m
        #     z_offset_world = rotation_corrected @ z_offset
            
        #     # 从位置中减去偏移量
        #     combined_actions[i, :3] = position - z_offset_world

        if self.stage <= self.program_info['num_stages']: 
            # self.env.sleep(2.0)
            save_path = os.path.join('./outputs', 'action.json') # TODO: save by stage?
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump({"ee_action_seq": combined_actions.tolist(), "stage": stage}, f, indent=4)

            with open(os.path.join(rekep_program_dir, f'stage{stage}_actions.json'), 'w') as f:
                json.dump({"ee_action_seq": combined_actions.tolist(), "stage": stage}, f, indent=4)
            print(f"{bcolors.OKGREEN}Actions saved to {save_path}\n and added to {rekep_program_dir}\n{bcolors.ENDC}")
            return
        else:
            print(f"{bcolors.OKGREEN}All stages completed\n\n{bcolors.ENDC}")
            # TODO: return to reset pose?
            # self.env.reset()
            return  

    def ur_get_joint_pos(self):
        joint_pos = self.robot_env.robot.get_joint_positions()
        return joint_pos
    
    def ur_get_ee_location(self):
        ee_pos = self.robot_env.robot.get_tcp_pose()
        return ee_pos[:3]
    
    def ur_get_ee_pose(self):
        ee_pos = self.robot_env.robot.get_tcp_pose()
        pos = pose2quat(ee_pos)
        # Extract quaternion components in [qx, qy, qz, qw] order
        quat = np.array([pos[4], pos[5], pos[6], pos[3]])  # Modified index order
        return np.concatenate([pos[:3], quat])

    def _load_constraints(self, rekep_program_dir):
        """Helper to load all stage constraints"""
        constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            constraint_fns[stage] = stage_dict
        return constraint_fns

    @timer_decorator
    def _get_next_subgoal(self, from_scratch, maintain_orientation=True):
        # pdb.set_trace()
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        print(f"Subgoal constraints: {subgoal_constraints}")
        print(f"Path constraints: {path_constraints}")
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            self.sdf_voxels,
                                                            self.collision_points,
                                                            self.is_grasp_stage,
                                                            self.curr_joint_pos,
                                                            from_scratch=from_scratch)
        
        # If maintain_orientation is True, keep the orientation from curr_ee_pose
        if maintain_orientation:
            subgoal_pose[3:7] = self.curr_ee_pose[3:7]
            print("Maintaining end-effector orientation from current pose")
        
        # Apply offset at subgoal generation time
        # if self.is_grasp_stage:
        # 获取位姿中的位置和角度部分
        position = subgoal_pose[:3]
        quat = subgoal_pose[3:7]  # 四元数 [qx, qy, qz, qw]
        
        # 从四元数创建旋转矩阵
        rotation_matrix = R.from_quat(quat).as_matrix()
        
        # 应用手性校正
        rot_correct = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        rotation_corrected = rotation_matrix @ rot_correct
        
        # 计算沿校正后EE z轴的偏移
        z_offset = np.array([0, 0, 0.16])  # z轴方向0.16m
        z_offset_world = rotation_corrected @ z_offset
        
        # 从位置中减去偏移量
        subgoal_pose[:3] = position - z_offset_world
            
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose, self.data_path)
        return subgoal_pose

    @timer_decorator
    def _get_next_path(self, next_subgoal, from_scratch):
        # pdb.set_trace()
        print(f"Start solving path from {self.curr_ee_pose} to {next_subgoal}")
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.keypoints,
                                                    self.keypoint_movable_mask,
                                                    path_constraints,
                                                    self.sdf_voxels,
                                                    self.collision_points,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        # 插值在这里
        processed_path = self._process_path(path)
        
        if self.visualize:
            self.visualizer.visualize_path(processed_path, self.data_path)
            
        return processed_path
    
    # TODO: check action sequence
    @timer_decorator
    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        
        # Create action sequence (now with 7 dimensions for 6 joints + gripper)
        ee_action_seq = np.zeros((dense_path.shape[0], 8)) 
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action() 
        return ee_action_seq

    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        # can only be grasp stage or release stage or none
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.robot_env.robot.control_gripper(close=False)
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        # pdb.set_trace()
        print("Grasp action")
    
    def _execute_release_action(self):
        print("Release action")
        
    def transform_keypoints_to_world_c2e2b(self, keypoints):
        """
        将关键点从相机坐标系转换到机器人基坐标系
        分两步转换：
        1. 相机坐标系 → end effector坐标系（通过eye-on-hand标定外参）
        2. end effector坐标系 → 机器人基坐标系（通过当前机械臂正向运动学）
        """
        # 转换为numpy数组
        keypoints = np.array(keypoints)
        
        # 第一步：相机坐标系 → end effector坐标系
        # 加载相机到end effector的外参
        ee2camera = self.load_camera_extrinsics()
        
        # 转换为齐次坐标
        keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
        
        # 第二步：end effector坐标系 → 机器人基坐标系
        # 获取当前end effector在基坐标系中的位姿（来自正向运动学）
        ee_pose = self.ur_get_ee_pose()  # [x, y, z, qx, qy, qz, qw]
        print(f"EE位姿: {ee_pose}")
        
        quat = np.array([ee_pose[3],ee_pose[4], ee_pose[5], ee_pose[6]])  # [qx,qy,qz,qw]
        rotation = R.from_quat(quat).as_matrix()  # 使用正确的四元数顺序
        
        # Apply handedness correction - reverse X and Z axes
        rot_correct = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        rotation_corrected = rotation @ rot_correct

        # Create transformation matrix
        base2ee = np.eye(4)
        base2ee[:3, :3] = rotation_corrected
        base2ee[:3, 3] = ee_pose[:3]
        
        # Camera frame
        camera_frame_incorrect = base2ee @ ee2camera

        camera_axes_correction = np.array([
            [0, 0, 1],  # New x-axis is old z-axis
            [-1, 0, 0], # New y-axis is old x-axis
            [0, -1, 0]  # New z-axis is negative old y-axis
    ])

        # Apply the correction to the camera frame rotation part
        camera_frame = camera_frame_incorrect.copy()
        camera_frame[:3, :3] = camera_frame_incorrect[:3, :3] @ camera_axes_correction


        # 应用变换
        base_coords_homogeneous = (camera_frame @ keypoints_homogeneous.T).T
        
        # 转换为非齐次坐标
        base_coords = base_coords_homogeneous[:, :3] / base_coords_homogeneous[:, 3, np.newaxis]
        
        return base_coords
    def transform_keypoints_to_world(self, keypoints):
        """
        将关键点从相机坐标系转换到机器人基座坐标系
        使用 Eye-to-Hand 标定结果 + 坐标轴对齐修正

        PiPer机械臂基座坐标系：
            X 向前，Y 向左，Z 向上
        RealSense光学坐标系：
            X 向右，Y 向下，Z 向前
        """
        keypoints = np.array(keypoints)
        keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))

        # # 加载 base → camera 的外参矩阵
        T_camera_to_base = self.load_camera_extrinsics()
     
        # position = np.array([0.29309614, 0.49516889, 0.42916608])
        # quaternion = np.array([0.06269216, 0.92432117, -0.36740451, 0.08193897])  # [x, y, z, w]

        # # Convert quaternion to rotation matrix
        # rotation_matrix = R.from_quat(quaternion).as_matrix()

        # # Create a 4x4 homogeneous transformation matrix
        # T_camera_to_base = np.eye(4)
        # T_camera_to_base[:3, :3] = rotation_matrix
        # T_camera_to_base[:3, 3] = position

        # 应用变换
        base_coords_homogeneous = (T_camera_to_base @ keypoints_homogeneous.T).T
        base_coords = base_coords_homogeneous[:, :3] / base_coords_homogeneous[:, 3, np.newaxis]

        return base_coords

    def load_camera_intrinsics(self):
        # D435i default intrinsics
        class RS_Intrinsics:
            def __init__(self):
                self.fx = 489.424683  # focal length x
                self.fy = 489.424683 # focal length y
                self.ppx = 325.761810 # principal point x
                self.ppy = 212.508759  # principal point y
        intrinsics = RS_Intrinsics()
        depth_scale = 0.001  # D435i default depth scale, 1mm
        
        # Convert to matrix format
        intrinsics_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        
        return intrinsics_matrix, depth_scale

    def load_camera_extrinsics(self):
        extrinsics_path = './configs/camera.yaml' 
        with open(extrinsics_path, 'r') as f:
            extrinsics_data = yaml.safe_load(f)
        qx = extrinsics_data['transformation']['qx']
        qy = extrinsics_data['transformation']['qy'] 
        qz = extrinsics_data['transformation']['qz']
        qw = extrinsics_data['transformation']['qw']
        
        # 转换为Scipy需要的[x,y,z,w]格式
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        
        # Extract transformation parameters
        tx = extrinsics_data['transformation']['x']
        ty = extrinsics_data['transformation']['y']
        tz = extrinsics_data['transformation']['z']
        
        # Create 4x4 transformation matrix
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rot
        extrinsics[:3, 3] = [tx, ty, tz]
        
        return extrinsics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', type=str, required=False, help='Instruction for the task')
    parser.add_argument('--rekep_program_dir', type=str, required=False, help='keypoint constrain proposed folder')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()

    # args.instruction = "Put the green package in the drawer, the robot is already grasping the package and the package is already aligned with the drawer opening."
    # args.obj_list = ['cloth']

    vlm_query_dir = "./vlm_query/"

    vlm_dirs = [os.path.join(vlm_query_dir, d) for d in os.listdir(vlm_query_dir) 
                if os.path.isdir(os.path.join(vlm_query_dir, d))]
    
    if vlm_dirs:
        newest_rekep_dir = max(vlm_dirs, key=os.path.getmtime)
        print(f"\033[92mUsing most recent directory: {newest_rekep_dir}\033[0m")
    else:
        print("No directories found under vlm_query")
        sys.exit(1)

    main = MainR2D2(visualize=args.visualize)
    main.perform_task(instruction=args.instruction, rekep_program_dir=newest_rekep_dir)
