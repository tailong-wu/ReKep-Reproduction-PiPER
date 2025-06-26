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
    def __init__(self, fx: float = 489.424683, fy: float = 489.424683, 
                 ppx: float = 325.761810, ppy: float = 212.508759):
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
            self.camera_intrinsics = CameraIntrinsics(
                fx=intrinsics['fx'],
                fy=intrinsics['fy'],
                ppx=intrinsics['ppx'],
                ppy=intrinsics['ppy']
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
            # 移动机械臂到初始位置
            # 从robot_state.json读取初始关节角度
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
            # 回到预设初始位置，便于Action以适应场景的第一个动作出发
            self.env.robot.move_to_joint_positions(self.init_joint_pos)
            print(f"{bcolors.OKBLUE}No stage specified, defaulting to stage 1{bcolors.ENDC}")
        
        stage = int(stage)
        
        # Get current robot state
        scene_keypoints = self.env.get_keypoint_positions()
        print(f"Camera frame keypoints: {scene_keypoints}")
        
        # Transform keypoints from camera to world coordinates
        scene_keypoints = self.transform_keypoints_to_world(scene_keypoints)
        print(f"World frame keypoints: {scene_keypoints}")
        
        # Get current robot state directly from robot (no file dependency)
        self.curr_ee_location = self.env.get_ee_location()  # 3D position [x,y,z] in meters
        tcp_pose = self.env.robot.get_tcp_pose()  # [x,y,z,rx,ry,rz] - position + rotation
        self.curr_joint_pos = self.get_joint_pos()  # Joint angles in radians
        
        # Extract orientation from TCP pose (rotation components)
        ee_orientation_euler = tcp_pose[3:6]  # [rx, ry, rz] rotation angles in degrees
        
        # Convert euler angles to quaternion for compatibility with execute_endpose_action.py
        # scipy quaternion format: [x, y, z, w]
        # ee_orientation_quat = R.from_euler('xyz', ee_orientation_euler, degrees=True).as_quat()
        # ee_orientation_quat = R.from_euler('yzx', np.radians(ee_orientation_euler), degrees=False).as_quat()
        ee_orientation_quat = R.from_euler('yzx', ee_orientation_euler, degrees=True).as_quat()
        # Create complete pose in quaternion format [x,y,z,qx,qy,qz,qw]
        self.curr_ee_pose = np.concatenate([tcp_pose[:3], ee_orientation_quat])
        
        # Combine end-effector and scene keypoints (now both in meters)
        self.keypoints = np.concatenate([[self.curr_ee_location], scene_keypoints], axis=0)
        print(f"Combined keypoints(end-effector and scene keypoints): {self.keypoints}")
        
        # Store complete robot state in rekep_program_dir using real-time data
        robot_state = {
            'rekep_stage': stage,
            'ee_info':{'ee_position': self.curr_ee_location.tolist(),
                        'ee_orientation': ee_orientation_quat.tolist()},  # Quaternion [x,y,z,w]
            'joint_positions': self.curr_joint_pos.tolist(),
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
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        
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
        self._update_keypoint_movable_mask()
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
                    'fx': 489.424683, 'fy': 489.424683,
                    'ppx': 325.761810, 'ppy': 212.508759,
                    'depth_scale': 0.001
                },
                'processing': {'resize_width': 640, 'resize_height': 480}
            }
    
    def _update_keypoint_movable_mask(self) -> None:
        """Update which keypoints can be moved in optimization."""
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)
    
    def transform_keypoints_to_world(self, keypoints: np.ndarray) -> np.ndarray:
        """Transform keypoints from camera coordinate system to robot base coordinate system.
        
        Uses Eye-to-Hand calibration results with coordinate axis alignment correction.
        
        PiPer robot base coordinate system:
            X forward, Y left, Z up
        RealSense optical coordinate system:
            X right, Y down, Z forward
            
        Args:
            keypoints: Keypoints in camera coordinate system
            
        Returns:
            Keypoints in robot base coordinate system
        """
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
