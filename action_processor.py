import torch
import numpy as np
import json
import os
import sys
import pdb 
from scipy.spatial.transform import Rotation as R
import yaml
from typing import Optional, List, Tuple, Dict, Any
import argparse
import time
import warnings

# Local imports
from env.rotations import pose2quat
from rekep.environment import R2D2Env
from rekep.ik_solver_piper import PiPERIKSolver
from rekep.subgoal_solver import SubgoalSolver
from rekep.path_solver import PathSolver
from rekep.visualizer import Visualizer
from rekep.environment_new import ReKepEnv
from rekep.utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="xFormers is not available")


def timer_decorator(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper


class CameraIntrinsics:
    """Camera intrinsic parameters."""
    def __init__(self, fx: float = 382.06, fy: float = 382.06, 
                 ppx: float = 321.79, ppy: float = 235.10):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy


class ActionProcessor:
    """Robot action processing system for ReKep manipulation tasks.
    
    This class handles the execution of robotic manipulation tasks by:
    - Loading ReKep program constraints
    - Planning subgoals and paths
    - Generating action sequences
    - Coordinating with robot hardware
    """
    
    def __init__(self, 
                 config_path: str = "./configs/config.yaml",
                 camera_config_path: str = "./configs/camera_config.yaml",
                 visualize: bool = False,
                 camera_intrinsics: Optional[CameraIntrinsics] = None,
                 cotracker_api_url: str = "http://localhost:5000",
                 camera_instance = None):
        """Initialize the ActionProcessor.
        
        Args:
            config_path: Path to configuration file
            visualize: Whether to enable visualization
            camera_intrinsics: Camera intrinsic parameters
            cotracker_api_url: CoTracker API服务器地址
        """
        self.config_path = config_path
        self.camera_config_path = camera_config_path
        self.visualize = visualize
        self.cotracker_api_url = cotracker_api_url
        self.camera_instance = camera_instance
        
        # 加载相机配置
        self.camera_config = self._load_camera_config()
        
        # 设置相机内参
        if camera_intrinsics is None:
            intrinsics = self.camera_config['intrinsics']
            # 统一使用彩色相机内参，realsense获取时已经将深度对齐到了RGB上。
            color_intrinsics = intrinsics['color']
            self.camera_intrinsics = CameraIntrinsics(
                fx=color_intrinsics['fx'],
                fy=color_intrinsics['fy'],
                ppx=color_intrinsics['ppx'],
                ppy=color_intrinsics['ppy']
            )
        else:
            self.camera_intrinsics = camera_intrinsics
        
        # Load configuration
        global_config = get_config(config_path=config_path)
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        
        # Set random seeds
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['seed'])
        
        # Robot configuration
        self.reset_joint_pos = np.array([
            -86.853, -1.560, -0.762, 2.939, 22.294, 0.000
        ]) * np.pi / (180 * 1000)  # PiPER home position (converted from 0.001 degrees to radians)
        
        # Initialize components
        self._initialize_components(global_config)
        
        # Performance monitoring
        self._timers = {}
        
        # Initialize program_info to avoid KeyError
        self.program_info = {}
        
    def _initialize_components(self, global_config: Dict[str, Any]) -> None:
        """Initialize robot environment and solvers."""
        # 初始化CoTracker客户端
        cotracker_client = None
        if self.cotracker_api_url:
            try:
                from keypoint_cotracker import CoTrackerClient
                cotracker_client = CoTrackerClient(base_url=self.cotracker_api_url)
                print(f"CoTracker客户端初始化成功: {self.cotracker_api_url}")
            except Exception as e:
                print(f"CoTracker客户端初始化失败: {e}")
        
        # 使用 ReKepEnv 替代 R2D2Env - 传入相机实例和cotracker_client避免资源冲突
        self.env = ReKepEnv(
            test_mode=False,
            camera_instance=self.camera_instance,
            cotracker_client=cotracker_client
        )
        print("使用 ReKepEnv 环境（共享相机和CoTracker实例）")
        
        # IK solver
        ik_solver = PiPERIKSolver(
            reset_joint_pos=self.reset_joint_pos,
            world2robot_homo=None,
        )
        
        # Motion planning solvers
        self.subgoal_solver = SubgoalSolver(
            global_config['subgoal_solver'], ik_solver, self.reset_joint_pos
        )
        self.path_solver = PathSolver(
            global_config['path_solver'], ik_solver, self.reset_joint_pos
        )
        
        # Visualization
        if self.visualize:
            self.visualizer = Visualizer(global_config['visualizer'])
            self.data_path = "/path"  # TODO: make configurable
    
    def _timer_log(self, operation: str, duration: float) -> None:
        """Log operation timing."""
        self._timers[operation] = duration
        print(f"\033[94m{operation}: {duration:.2f}s\033[0m")
    
    @timer_decorator
    def process_action_task(self, 
                           instruction: str,
                           rekep_program_dir: str,
                           stage: Optional[int] = None,
                           output_dir: str = './outputs') -> str:
        """Process a complete action task for robotic manipulation.
        
        Args:
            instruction: Task instruction for the robot
            rekep_program_dir: Directory containing ReKep program constraints
            stage: Specific stage to execute (if None, defaults to stage 1)
            output_dir: Directory for output files
            
        Returns:
            Path to the generated action file
        """
        start_time = time.time()
        
        # 设置rekep_program_dir到环境中，以便在创建机器人状态字典时使用
        self.env.rekep_program_dir = rekep_program_dir
        # 确保robot对象也能访问到rekep_program_dir
        self.env.robot.rekep_program_dir = rekep_program_dir
        
        # Execute the task
        action_file_path = self._execute_task(rekep_program_dir, stage, output_dir)
        
        total_time = time.time() - start_time
        print(f"\033[92mTotal action processing time: {total_time:.2f} seconds\033[0m")
        
        return action_file_path
    
    @timer_decorator
    def _execute_task(self, 
                     rekep_program_dir: str, 
                     stage: Optional[int] = None,
                     output_dir: str = './outputs') -> str:
        """Execute the robotic manipulation task.
        
        Args:
            rekep_program_dir: Directory containing ReKep program
            stage: Stage number to execute
            output_dir: Output directory for action files
            
        Returns:
            Path to generated action file
        """
        # Load program info and constraints
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        
        # 只在第一阶段注册初始关键点，后续阶段使用已注册的关键点
        if stage is None or stage == 1:
            # Register initial keypoints
            self.env.register_keypoints(self.program_info['init_keypoint_positions'],self.program_info['keypoints_2d_coordinates'])
            print(f"Registering initial keypoints for stage 1")
        else:
            print(f"Using existing keypoints for stage {stage}")
        
        # Load all stage constraints
        self.constraint_fns = self._load_constraints(rekep_program_dir)
        
        # Initialize keypoint movable mask
        self.keypoint_movable_mask = np.ones(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable
        
        # Determine stage to execute
        if stage is None or stage ==1:
            # Default to stage 1 (no longer dependent on robot_state.json)
            stage = 1
            # 从robot_state.json读取初始关节角度用于保存状态
            try:
                with open('robot_state.json', 'r') as f:
                    initial_state = json.load(f)
                    # 优先使用initial_position中的关节角度
                    if 'initial_position' in initial_state and 'joint_positions' in initial_state['initial_position']:
                        self.init_joint_pos = np.array(initial_state['initial_position']['joint_positions'])
                        print(f"从robot_state.json的initial_position读取初始关节角度: {self.init_joint_pos}")
                    else:
                        # 兼容旧版本，使用reset_joint_pos
                        self.init_joint_pos = np.array(initial_state['joint_info']['reset_joint_pos'])
                        print(f"从robot_state.json的joint_info.reset_joint_pos读取初始关节角度: {self.init_joint_pos}")
            except (FileNotFoundError, KeyError) as e:
                print(f"无法读取robot_state.json,使用默认初始关节角度: {e}")
                self.init_joint_pos = self.reset_joint_pos
            print(f"{bcolors.OKBLUE}No stage specified, defaulting to stage 1{bcolors.ENDC}")
        
        stage = int(stage)
        
        # Get current robot state
        # 如果还没有获取过关键点位置,则获取一次并缓存
        if not hasattr(self, '_cached_scene_keypoints_3d'):
            self._cached_scene_keypoints_3d = self.env.get_keypoint_positions()
        scene_keypoints_3d = self._cached_scene_keypoints_3d
        print(f"Camera frame keypoints (3D): {scene_keypoints_3d}")
        
        # Transform keypoints from camera to world coordinates
        world_keypoints_3d = self.transform_keypoints_to_world(scene_keypoints_3d)
        print(f"World frame keypoints (3D): {world_keypoints_3d}")
        

        # Get current robot state directly from robot (no file dependency)
        self.curr_ee_location = self.env.get_ee_location()  # 3D position [x,y,z] in meters
        tcp_pose = self.env.robot.get_tcp_pose()  # [x,y,z,rx,ry,rz] - position + rotation
        self.curr_joint_pos = self.get_joint_pos()  # Joint angles in radians
        
        # Extract orientation from TCP pose (rotation components)
        ee_orientation_euler_deg = tcp_pose[3:6]  # [rx, ry, rz] rotation angles in degrees
        
        # Convert to radians for consistency with robot_state.json
        ee_orientation_euler = np.radians(ee_orientation_euler_deg)  # [rx, ry, rz] in radians
        
        # Convert euler angles to quaternion for compatibility with execute_endpose_action.py
        # scipy quaternion format: [x, y, z, w]
        ee_orientation_quat = R.from_euler('xyz', ee_orientation_euler_deg, degrees=True).as_quat()
        # Create complete pose in quaternion format [x,y,z,qx,qy,qz,qw]
        self.curr_ee_pose = np.concatenate([tcp_pose[:3], ee_orientation_quat])
        
        # Combine end-effector and scene keypoints (now both in meters)
        self.keypoints = np.concatenate([[self.curr_ee_location], world_keypoints_3d], axis=0)
        print(f"Combined keypoints(end-effector and scene keypoints): {self.keypoints}")
        
        # 获取原始数据用于raw_data字段
        try:
            # 获取当前末端位姿原始数据
            arm_end_pose_msgs = self.env.robot.piper.GetArmEndPoseMsgs()
            end_pose_raw = {
                'X_axis': arm_end_pose_msgs.end_pose.X_axis,
                'Y_axis': arm_end_pose_msgs.end_pose.Y_axis,
                'Z_axis': arm_end_pose_msgs.end_pose.Z_axis,
                'RX_axis': arm_end_pose_msgs.end_pose.RX_axis,
                'RY_axis': arm_end_pose_msgs.end_pose.RY_axis,
                'RZ_axis': arm_end_pose_msgs.end_pose.RZ_axis,
                'position_unit': '0.001 mm',
                'orientation_unit': '0.001 degrees'
            }
            
            # 获取当前关节角度原始数据
            joint_msgs = self.env.robot.piper.GetArmJointMsgs()
            joint_angles_raw = {
                'joint_1': joint_msgs.joint_state.joint_1,
                'joint_2': joint_msgs.joint_state.joint_2,
                'joint_3': joint_msgs.joint_state.joint_3,
                'joint_4': joint_msgs.joint_state.joint_4,
                'joint_5': joint_msgs.joint_state.joint_5,
                'joint_6': joint_msgs.joint_state.joint_6,
                'unit': '0.001 degrees'
            }
        except Exception as e:
            print(f"获取原始数据时发生错误: {e}，使用模拟数据")
            # 使用模拟数据
            # 将米转换为0.001毫米
            pos_factor = 1000000
            # 将弧度转换为0.001度
            rot_factor = 180 * 1000 / np.pi
            
            end_pose_raw = {
                'X_axis': int(self.curr_ee_location[0] * pos_factor),
                'Y_axis': int(self.curr_ee_location[1] * pos_factor),
                'Z_axis': int(self.curr_ee_location[2] * pos_factor),
                'RX_axis': int(ee_orientation_euler[0] * rot_factor),
                'RY_axis': int(ee_orientation_euler[1] * rot_factor),
                'RZ_axis': int(ee_orientation_euler[2] * rot_factor),
                'position_unit': '0.001 mm',
                'orientation_unit': '0.001 degrees'
            }
            
            joint_angles_raw = {
                'joint_1': int(self.curr_joint_pos[0] * rot_factor),
                'joint_2': int(self.curr_joint_pos[1] * rot_factor),
                'joint_3': int(self.curr_joint_pos[2] * rot_factor),
                'joint_4': int(self.curr_joint_pos[3] * rot_factor),
                'joint_5': int(self.curr_joint_pos[4] * rot_factor),
                'joint_6': int(self.curr_joint_pos[5] * rot_factor),
                'unit': '0.001 degrees'
            }
        
        # Store complete robot state in rekep_program_dir using real-time data
        robot_state = {
            'timestamp': time.time(),
            'rekep_stage': stage,
            'joint_info': {
                'joint_positions': self.curr_joint_pos.tolist(),
                'reset_joint_pos': self.init_joint_pos.tolist()
            },
            'initial_position': {
                'joint_positions': self.init_joint_pos.tolist(),
                'ee_position': self.curr_ee_location.tolist(),
                'ee_orientation': ee_orientation_euler.tolist()
            },
            'ee_info': {
                'position': self.curr_ee_location.tolist(),
                'orientation': {
                    'quaternion': ee_orientation_quat.tolist(),
                    'euler': ee_orientation_euler.tolist(),
                    'unit': 'radians',
                    'description': 'End effector orientation in radians [rx, ry, rz]'
                }
            },
            'gripper_info': {
                'state': 0.0  # 默认为打开状态
            },
            'safety_info': {
                'collision_status': 'false',
                'safety_status': 'normal',
                'errors': []
            },
            'control_info': {
                'control_mode': 'position',
                'operation_mode': 'auto'
            },
            'misc': {},
            'raw_data': {
                'joint_angles_raw': joint_angles_raw,
                'end_pose_raw': end_pose_raw
            },
            'keypoints': self.keypoints[1:].tolist() # 不包括末端位置关键点
        }
        with open(os.path.join(rekep_program_dir, f'robot_state_{stage}.json'), 'w') as f:
            json.dump(robot_state, f, indent=4)
        self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
        self.collision_points = self.env.get_collision_points()
        
        # Validate stage
        if stage > self.program_info['num_stages']:
            print(f"{bcolors.FAIL}Stage {stage} is out of bounds, skipping\n{bcolors.ENDC}")
            return ""
        
        # Update stage information
        self._update_stage(stage)
        
        # Generate actions for this stage
        next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter, maintain_orientation=True)
        print(f"Next subgoal: {next_subgoal}")
        next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
        self.first_iter = False
        
        # Save actions
        action_file_path = self._save_actions(next_path, stage, rekep_program_dir, output_dir)
        
        return action_file_path
    
    def _save_actions(self, 
                     actions: np.ndarray, 
                     stage: int, 
                     rekep_program_dir: str, 
                     output_dir: str) -> str:
        """Save action sequence to files.
        
        Args:
            actions: Action sequence array
            stage: Current stage number
            rekep_program_dir: ReKep program directory
            output_dir: Output directory
            
        Returns:
            Path to saved action file
        """
        if stage <= self.program_info['num_stages']:
            # Save to main output directory
            save_path = os.path.join(output_dir, 'action.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            action_data = {
                "ee_action_seq": actions.tolist(), 
                "stage": stage
            }
            
            with open(save_path, 'w') as f:
                json.dump(action_data, f, indent=4)
            
            # Save to rekep program directory
            stage_save_path = os.path.join(rekep_program_dir, f'stage{stage}_actions.json')
            with open(stage_save_path, 'w') as f:
                json.dump(action_data, f, indent=4)
            
            print(f"{bcolors.OKGREEN}Actions saved to {save_path}\n and added to {rekep_program_dir}\n{bcolors.ENDC}")
            return save_path
        else:
            print(f"{bcolors.OKGREEN}All stages completed\n\n{bcolors.ENDC}")
            return ""
    
    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions."""
        return self.env.get_joint_pos()
    
    def get_ee_location(self) -> np.ndarray:
        """Get current end-effector location."""
        return self.env.get_ee_location()
    
    def get_ee_pose(self) -> np.ndarray:
        """Get current end-effector pose."""
        return self.env.get_ee_pose()
    
    def _load_constraints(self, rekep_program_dir: str) -> Dict[int, Dict[str, List]]:
        """Load all stage constraints from ReKep program directory.
        
        Args:
            rekep_program_dir: Directory containing constraint files
            
        Returns:
            Dictionary mapping stage numbers to constraint functions
        """
        constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)
                stage_dict[constraint_type] = (
                    load_functions_from_txt(load_path, get_grasping_cost_fn) 
                    if os.path.exists(load_path) else []
                )
            constraint_fns[stage] = stage_dict
        return constraint_fns
    
    @timer_decorator
    def _get_next_subgoal(self, from_scratch: bool, maintain_orientation: bool = True) -> np.ndarray:
        """Generate next subgoal for the current stage.
        
        Args:
            from_scratch: Whether to solve from scratch
            maintain_orientation: Whether to maintain current orientation
            
        Returns:
            Subgoal pose array
        """
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        
        print(f"Subgoal constraints: {subgoal_constraints}")
        print(f"Path constraints: {path_constraints}")
        print(self.keypoint_movable_mask)
        subgoal_pose, debug_dict = self.subgoal_solver.solve(
            self.curr_ee_pose,
            self.keypoints,
            self.keypoint_movable_mask,
            subgoal_constraints,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.is_grasp_stage,
            self.curr_joint_pos,
            from_scratch=from_scratch
        )
        
        # Maintain orientation if requested
        if maintain_orientation:
            subgoal_pose[3:7] = self.curr_ee_pose[3:7]
            print("Maintaining end-effector orientation from current pose")
        
        # Log debug information
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        
        # Visualize if enabled
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose, self.data_path)
        
        return subgoal_pose
    
    @timer_decorator
    def _get_next_path(self, next_subgoal: np.ndarray, from_scratch: bool) -> np.ndarray:
        """Generate path to the next subgoal.
        
        Args:
            next_subgoal: Target subgoal pose
            from_scratch: Whether to solve from scratch
            
        Returns:
            Processed path array
        """
        print(f"Start solving path from {self.curr_ee_pose} to {next_subgoal}")
        
        path_constraints = self.constraint_fns[self.stage]['path']
        print(f"Debug: curr_ee_pose shape: {self.curr_ee_pose.shape}")
        print(f"Debug: curr_ee_pose content: {self.curr_ee_pose}")
        path, debug_dict = self.path_solver.solve(
            self.curr_ee_pose,
            next_subgoal,
            self.keypoints,
            self.keypoint_movable_mask,
            path_constraints,
            self.sdf_voxels,
            self.collision_points,
            self.curr_joint_pos,
            from_scratch=from_scratch
        )
        
        print_opt_debug_dict(debug_dict)
        
        # Process and interpolate path
        processed_path = self._process_path(path)
        
        # Visualize if enabled
        if self.visualize:
            self.visualizer.visualize_path(processed_path, self.data_path)
        
        return processed_path
    
    @timer_decorator
    def _process_path(self, path: np.ndarray) -> np.ndarray:
        """Process and interpolate the path.
        
        Args:
            path: Raw path array
            
        Returns:
            Processed action sequence
        """
        # Spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        
        num_steps = get_linear_interpolation_steps(
            full_control_points[0], full_control_points[-1],
            self.config['interpolate_pos_step_size'],
            self.config['interpolate_rot_step_size']
        )
        
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        
        # Create action sequence (8 dimensions: 7 for pose + 1 for gripper)
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        
        # Set gripper actions based on stage type
        if self.is_grasp_stage:
            # For grasping stage: keep gripper open during movement, close at the end
            ee_action_seq[:-1, 7] = self.env.get_gripper_open_action()  # Keep open during movement
            ee_action_seq[-1, 7] = self.env.get_gripper_close_action()   # Close at the end
            print(f"抓取阶段：路径中保持夹爪打开，最后一步关闭夹爪")
        elif self.is_release_stage:
            # For release stage: keep gripper closed during movement, open at the end
            ee_action_seq[:-1, 7] = self.env.get_gripper_close_action()  # Keep closed during movement
            ee_action_seq[-1, 7] = self.env.get_gripper_open_action()    # Open at the end
            print(f"释放阶段：路径中保持夹爪关闭，最后一步打开夹爪")
        else:
            # For other stages: maintain current gripper state
            ee_action_seq[:, 7] = self.env.get_gripper_null_action()
            print(f"普通移动阶段：保持夹爪状态不变")
        
        return ee_action_seq
    
    def _update_stage(self, stage: int) -> None:
        """Update stage-specific information.
        
        Args:
            stage: Stage number to update to
        """
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        
        # Validate stage type
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        
        # Prepare gripper for grasping stage
        if self.is_grasp_stage:
            self.env.control_gripper(close=False)
        
        # Initialize stage state
        self.action_queue = []
        # TODO 使用真实的抓取检测
        self._update_keypoint_movable_mask_mock()
        if stage == 1:
            self.first_iter = True
    
    def _load_camera_config(self) -> Dict[str, Any]:
        """加载相机配置文件
        
        Returns:
            Dict: 相机配置
        """
        try:
            with open(self.camera_config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载相机配置文件失败: {e}")
            # 返回默认配置
            return {
                'resolution': {'width': 640, 'height': 480, 'fps': 30},
                'intrinsics': {
                    'color': {
                        'fx': 606.60, 'fy': 605.47,
                        'ppx': 323.69, 'ppy': 247.12
                    },
                    'depth_scale': 0.001
                },
                'processing': {'resize_width': 640, 'resize_height': 480}
            }
    
    def _update_keypoint_movable_mask(self) -> None:
        """Update which keypoints can be moved in optimization."""
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)
    
    def _update_keypoint_movable_mask_mock(self) -> None:
        """Mock version: Update which keypoints can be moved based on previous grasp stages.
        
        This mock version directly sets keypoints as movable if they were grasped 
        in previous stages, without relying on the environment's grasping state.
        """
        # Reset all keypoints to not movable (except ee)
        for i in range(1, len(self.keypoint_movable_mask)):
            self.keypoint_movable_mask[i] = False
        
        
        # Check all previous stages for grasped keypoints
        for stage_idx in range(self.stage-1):
            grasp_keypoint_idx = self.program_info['grasp_keypoints'][stage_idx]
            if grasp_keypoint_idx != -1:  # -1 means no grasp in this stage
                # Convert to keypoint_movable_mask index (add 1 because first is ee)
                mask_idx = grasp_keypoint_idx + 1
                if mask_idx < len(self.keypoint_movable_mask):
                    self.keypoint_movable_mask[mask_idx] = True
                    print(f"Mock: Setting keypoint {grasp_keypoint_idx} as movable (grasped in stage {stage_idx + 1})")
    
    def transform_keypoints_to_world(self, keypoints: np.ndarray, hand_in_eye:bool=True) -> np.ndarray:
        """Transform keypoints from camera coordinate system to robot base coordinate system.
        
        Uses Eye-to-Hand calibration results with coordinate axis alignment correction.
        
        PiPer robot base coordinate system:
            X forward, Y left, Z up
        RealSense optical coordinate system:
            X right, Y down, Z forward
            
        Args:
            keypoints: Keypoints in camera coordinate system (3D)
            hand_in_eye: Whether the hand is in the eye coordinate system, default is True
            
        Returns:
            Keypoints in robot base coordinate system
        """
        if hand_in_eye:
            # 注意：这里由于标定的坐标系不相同，需要一些转换，具体参考标定结果。
            # 直接使用原始关键点,不做坐标系修正
            keypoints_standard = keypoints.copy()
            # 加载手眼标定结果
            T_standard_gripper2cam = self.load_camera_extrinsics()
            R_standard_gripper2cam = T_standard_gripper2cam[:3, :3]
            t_standard_gripper2cam = T_standard_gripper2cam[:3, 3]

            # 获取当前机械臂末端底座位姿
            gripper_base_pose = self.env.robot.get_gripper_base_pose()  # [x,y,z,rx,ry,rz]
            x,y,z,rx,ry,rz = gripper_base_pose
            rx,ry,rz = np.deg2rad([rx,ry,rz])
            R_base2gripper = R.from_euler('xyz', [rx,ry,rz]).as_matrix()
            t_base2gripper = np.array([x,y,z]).reshape(3,1)
            
            # 将所有关键点转换为齐次坐标形式
            keypoints_cam = np.array(keypoints_standard)
            keypoints_cam_homogeneous = np.hstack((keypoints_cam, np.ones((keypoints_cam.shape[0], 1))))

            # 构建从相机到机械臂基座的变换矩阵
            T_gripper2cam = np.eye(4)
            T_gripper2cam[:3, :3] = R_standard_gripper2cam
            T_gripper2cam[:3, 3] = t_standard_gripper2cam.reshape(3)

            T_base2gripper = np.eye(4)
            T_base2gripper[:3, :3] = R_base2gripper
            T_base2gripper[:3, 3] = t_base2gripper.reshape(3)

            # 一次性完成所有变换
            T_total = T_base2gripper @ T_gripper2cam
            base_coords_homogeneous = (T_total @ keypoints_cam_homogeneous.T).T
            base_coords = base_coords_homogeneous[:, :3]
                
        else:
            keypoints = np.array(keypoints)
            keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
            
            # Load base → camera extrinsic matrix
            T_camera_to_base = self.load_camera_extrinsics()
            
            # Apply transformation
            base_coords_homogeneous = (T_camera_to_base @ keypoints_homogeneous.T).T
            base_coords = base_coords_homogeneous[:, :3] / base_coords_homogeneous[:, 3, np.newaxis]
        
        return base_coords
    
    def load_camera_intrinsics(self) -> Tuple[np.ndarray, float]:
        """Load camera intrinsic parameters.
        
        Returns:
            Tuple of (intrinsics_matrix, depth_scale)
        """
        intrinsics = self.camera_intrinsics
        depth_scale = 0.001  # D435i default depth scale, 1mm
        
        # Convert to matrix format
        intrinsics_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        
        return intrinsics_matrix, depth_scale
    
    def load_camera_extrinsics(self) -> np.ndarray:
        """Load camera extrinsic parameters from configuration file.
        
        Returns:
            4x4 transformation matrix from camera to base
        """
        extrinsics_path = './configs/camera_config.yaml'
        with open(extrinsics_path, 'r') as f:
            extrinsics_data = yaml.safe_load(f)
        
        # Extract quaternion and position
        qx = extrinsics_data['transformation']['qx']
        qy = extrinsics_data['transformation']['qy']
        qz = extrinsics_data['transformation']['qz']
        qw = extrinsics_data['transformation']['qw']
        
        # Convert to rotation matrix (Scipy format: [x,y,z,w])
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        
        # Extract translation
        tx = extrinsics_data['transformation']['x']
        ty = extrinsics_data['transformation']['y']
        tz = extrinsics_data['transformation']['z']
        
        # Create 4x4 transformation matrix
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rot
        extrinsics[:3, 3] = [tx, ty, tz]
        
        return extrinsics
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance timing metrics.
        
        Returns:
            Dictionary of operation timings
        """
        return self._timers.copy()

# Convenience functions for backward compatibility and ease of use
def create_action_processor(config_path: str = "./configs/config.yaml",
                           visualize: bool = False,
                           camera_intrinsics: Optional[CameraIntrinsics] = None) -> ActionProcessor:
    """Create and return an ActionProcessor instance.
    
    Args:
        config_path: Path to configuration file
        visualize: Whether to enable visualization
        camera_intrinsics: Camera intrinsic parameters
    
    Returns:
        ActionProcessor: Initialized action processor instance
    """
    return ActionProcessor(
        config_path=config_path,
        visualize=visualize,
        camera_intrinsics=camera_intrinsics
    )


def process_action_task(instruction: str,
                       rekep_program_dir: str,
                       stage: Optional[int] = None,
                       visualize: bool = False,
                       config_path: str = "./configs/config.yaml",
                       output_dir: str = './outputs') -> str:
    """Process an action task for robotic manipulation.
    
    Args:
        instruction: Task instruction for the robot
        rekep_program_dir: Directory containing ReKep program constraints
        stage: Specific stage to execute (if None, defaults to stage 1)
        visualize: Whether to visualize planning steps
        config_path: Path to configuration file
        output_dir: Directory for output files
    
    Returns:
        str: Path to the generated action file
    """
    processor = ActionProcessor(config_path=config_path, visualize=visualize)
    return processor.process_action_task(
        instruction=instruction,
        rekep_program_dir=rekep_program_dir,
        stage=stage,
        output_dir=output_dir
    )


# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Action Processor")
    parser.add_argument('--instruction', type=str, required=False, 
                       help='Instruction for the task')
    parser.add_argument('--rekep_program_dir', type=str, required=False, 
                       help='ReKep program directory with constraints')
    parser.add_argument('--stage', type=int, required=False,
                       help='Specific stage to execute')
    parser.add_argument('--visualize', action='store_true', 
                       help='Visualize each solution before executing')
    parser.add_argument('--config', type=str, default="./configs/config.yaml",
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for action files')
    args = parser.parse_args()
    
    # Auto-detect most recent ReKep program directory if not specified
    if args.rekep_program_dir is None:
        vlm_query_dir = "./vlm_query/"
        vlm_dirs = [os.path.join(vlm_query_dir, d) for d in os.listdir(vlm_query_dir) 
                    if os.path.isdir(os.path.join(vlm_query_dir, d))]
        
        if vlm_dirs:
            args.rekep_program_dir = max(vlm_dirs, key=os.path.getmtime)
            print(f"\033[92mUsing most recent directory: {args.rekep_program_dir}\033[0m")
        else:
            print("No directories found under vlm_query")
            sys.exit(1)
    
    # Process the action task
    action_file = process_action_task(
        instruction=args.instruction or "Execute robotic manipulation task",
        rekep_program_dir=args.rekep_program_dir,
        stage=args.stage,
        visualize=args.visualize,
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    if action_file:
        print(f"\033[92mAction file generated: {action_file}\033[0m")
    else:
        print("\033[91mFailed to generate action file\033[0m")
