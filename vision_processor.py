"""Vision Processor Module for ReKep Robotic Manipulation System

This module provides a clean interface for vision processing in robotic manipulation tasks.
It encapsulates object detection, segmentation, keypoint proposal, and constraint generation.

Author: Tony Wang, University of Pennsylvania
Refactored for better modularity and maintainability.
"""

import os
import torch
import numpy as np
import cv2
import json
import time
import yaml
import warnings
from typing import Optional, Tuple, List, Dict, Any

from rekep.keypoint_proposal import KeypointProposer
from rekep.constraint_generation import ConstraintGenerator
from rekep.perception.gdino import GroundingDINO
from rekep.utils import bcolors, get_config

warnings.filterwarnings("ignore", category=UserWarning)


class CameraIntrinsics:
    """Camera intrinsic parameters for depth-to-pointcloud conversion."""
    
    def __init__(self, fx: float, fy: float, ppx: float, ppy: float):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy


class VisionProcessor:
    """Main vision processing class for robotic manipulation tasks.
    
    This class handles the complete vision pipeline including:
    - Object detection and segmentation
    - Depth processing and point cloud generation
    - Keypoint proposal
    - Constraint generation for motion planning
    """
    
    # Default camera intrinsics for different RealSense models
    # 统一使用彩色相机内参（原深度已经对齐到彩色相机上）
    # 彩色相机内参
    COLOR_INTRINSICS = CameraIntrinsics(fx=606.60, fy=605.47, ppx=323.69, ppy=247.12)
    
    def __init__(self, 
                 config_path: str = "./configs/config.yaml",
                 camera_config_path: str = "./configs/camera_config.yaml",
                 visualize: bool = True,
                 camera_intrinsics: Optional[CameraIntrinsics] = None,
                 depth_scale: float = 0.001,
                 seed: Optional[int] = None):
        """Initialize the vision processor.
        
        Args:
            config_path: Path to the configuration file
            visualize: Whether to enable visualization
            camera_intrinsics: Camera intrinsic parameters (defaults to D435i)
            depth_scale: Depth scale factor (default: 0.001 for 1mm)
            seed: Random seed for reproducibility
        """
        self.config_path = config_path
        self.camera_config_path = camera_config_path
        self.visualize = visualize
        self.depth_scale = depth_scale
        
        # Load configuration
        global_config = get_config(config_path=config_path)
        self.config = global_config['main']
        
        # Load camera configuration
        self.camera_config = self._load_camera_config()
        
        # Set camera intrinsics from config or use provided ones
        if camera_intrinsics is None:
            intrinsics = self.camera_config['intrinsics']
            # 统一使用彩色相机内参
            color_intrinsics = intrinsics['color']
            self.camera_intrinsics = CameraIntrinsics(
                fx=color_intrinsics['fx'],
                fy=color_intrinsics['fy'],
                ppx=color_intrinsics['ppx'],
                ppy=color_intrinsics['ppy']
            )
            self.depth_scale = intrinsics['depth_scale']
        else:
            self.camera_intrinsics = camera_intrinsics
        # Set random seeds
        if seed is None:
            seed = self.config.get('seed', 42)
        self._set_random_seeds(seed)
        
        # Initialize components
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        
        # Store config for lazy initialization of constraint_generator
        self._constraint_generator_config = global_config['constraint_generator']
        self._constraint_generator = None
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _load_camera_config(self) -> Dict[str, Any]:
        """Load camera configuration from YAML file.
        
        Returns:
            Dictionary containing camera configuration
        """
        try:
            with open(self.camera_config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading camera config: {e}")
            # Return default configuration
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
    
    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _timer_log(self, func_name: str, duration: float) -> None:
        """Log function execution time."""
        print(f"Function {func_name} took {duration:.2f} seconds to execute")
    
    def load_rgb_depth_data(self, data_path: str, use_varied_camera: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Load RGB and depth data from files.
        
        Args:
            data_path: Directory containing the image files
            use_varied_camera: Whether to use varied camera data (vs fixed camera)
            
        Returns:
            Tuple of (rgb_image, depth_image)
        """
        # 使用标准命名格式
        if use_varied_camera:
            color_path = os.path.join(data_path, 'varied_camera_raw.png')
            depth_path = os.path.join(data_path, 'varied_camera_depth.npy')
        print(f"\033[92mLoading RGB from: {color_path}\033[0m")
        print(f"\033[92mLoading depth from: {depth_path}\033[0m")
        
        # Load and convert RGB image
        bgr = cv2.imread(color_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not load RGB image from {color_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Load depth image
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Could not load depth image from {depth_path}")
        depth = np.load(depth_path)
        
        # Resize images to match configuration if needed
        target_width = self.camera_config['processing']['resize_width']
        target_height = self.camera_config['processing']['resize_height']
        
        if rgb.shape[1] != target_width or rgb.shape[0] != target_height:
            print(f"\033[93mResizing RGB image from {rgb.shape[1]}x{rgb.shape[0]} to {target_width}x{target_height}\033[0m")
            rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        if depth.shape[1] != target_width or depth.shape[0] != target_height:
            print(f"\033[93mResizing depth image from {depth.shape[1]}x{depth.shape[0]} to {target_width}x{target_height}\033[0m")
            depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        
        print(f"\033[92mRGB shape: {rgb.shape}, Depth shape: {depth.shape}\033[0m")
        return rgb, depth
    
    def detect_objects(self, color_path: str, output_dir: str = './data/') -> Tuple[List, np.ndarray]:
        """Detect objects using Grounding DINO.
        
        Args:
            color_path: Path to the color image
            output_dir: Directory to save detection results
            
        Returns:
            Tuple of (bounding_boxes, masks)
        """
        start_time = time.time()
        
        print(f"\033[92mRunning Dino-X object detection\033[0m")
        print(f"\033[92mUsing image: {color_path}\033[0m")
        
        # 确保图像文件存在
        if not os.path.exists(color_path):
            raise FileNotFoundError(f"Could not find image file: {color_path}")
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        gdino = GroundingDINO()
        predictions = gdino.get_dinox(color_path)
        bboxes, masks = gdino.visualize_bbox_and_mask(predictions, color_path, output_dir)
        
        # Process masks
        masks = masks.astype(bool)
        masks = np.stack(masks, axis=0)
        
        print(f"\033[92mDetected {len(masks)} objects\033[0m")
        print(f"\033[92mMask shape: {masks[0].shape if len(masks) > 0 else 'No masks'}\033[0m")
        
        self._timer_log("detect_objects", time.time() - start_time)
        return bboxes, masks
    
    def depth_to_pointcloud(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth image to 3D point cloud.
        
        Args:
            depth: Depth image array
            
        Returns:
            Point cloud array of shape (height * width, 3)
        """
        start_time = time.time()
        
        height, width = depth.shape
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        
        points = np.zeros((height * width, 3))
        valid_mask = depth > 0
        
        # Convert pixel coordinates to 3D coordinates
        x = (u[valid_mask].flatten() - self.camera_intrinsics.ppx) / self.camera_intrinsics.fx
        y = (v[valid_mask].flatten() - self.camera_intrinsics.ppy) / self.camera_intrinsics.fy
        z = depth[valid_mask].flatten() * self.depth_scale
        
        x = np.multiply(x, z)
        y = np.multiply(y, z)
        
        valid_indices = np.where(valid_mask.flatten())[0]
        points[valid_indices] = np.stack((x, y, z), axis=-1)
        
        print(f"\033[92mGenerated point cloud with shape: {points.shape}\033[0m")
        self._timer_log("depth_to_pointcloud", time.time() - start_time)
        
        return points
    
    def generate_keypoints(self, rgb: np.ndarray, points: np.ndarray, masks: np.ndarray) -> Tuple[List, np.ndarray, List]:
        """Generate keypoints for manipulation planning.
        
        Args:
            rgb: RGB image
            points: 3D point cloud
            masks: Object segmentation masks
            
        Returns:
            Tuple of (keypoints, projected_image, 2D pixel coordinates)
        """
        start_time = time.time()
        
        # Get keypoints and extract 2D coordinates from keypoint proposer
        keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, masks)
        
        # Extract 2D pixel coordinates from candidate_pixels in keypoint proposer
        # We need to access the candidate_pixels from the last run
        candidate_pixels_2d = []
        if hasattr(self.keypoint_proposer, '_last_candidate_pixels'):
            candidate_pixels_2d = self.keypoint_proposer._last_candidate_pixels.tolist()
        
        print(f'{bcolors.HEADER}Generated {len(keypoints)} keypoints{bcolors.ENDC}')
        self._timer_log("generate_keypoints", time.time() - start_time)
        
        return keypoints, projected_img, candidate_pixels_2d
    
    @property
    def constraint_generator(self):
        """Lazy initialization of constraint generator."""
        if self._constraint_generator is None:
            print(f"{bcolors.OKBLUE}Initializing constraint generator...{bcolors.ENDC}")
            self._constraint_generator = ConstraintGenerator(self._constraint_generator_config)
        return self._constraint_generator
    

    
    def save_visualization(self, image: np.ndarray, output_path: str = './data/rekep_with_keypoints.png') -> None:
        """Save visualization image.
        
        Args:
            image: Image to save
            output_path: Output file path
        """
        if self.visualize:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.axis('on')
            plt.title('Annotated Image with Keypoints')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"\033[92mVisualization saved to: {output_path}\033[0m")
    
    def process_vision_task(self, 
                          instruction: Optional[str] = None,
                          data_path: str = "./data/realsense_captures",
                          obj_list: Optional[str] = None,
                          use_varied_camera: bool = True,
                          output_dir: str = './data/',
                          generate_constraints: bool = True) -> str:
        """Process a complete vision task for robotic manipulation.
        
        Args:
            instruction: Task instruction for the robot
            data_path: Path to directory containing RGB-D data
            obj_list: Optional list of objects to detect (currently unused)
            use_varied_camera: Whether to use varied camera data
            output_dir: Directory for output files
            
        Returns:
            Path to the generated ReKep program directory
        """
        start_time = time.time()
        
        # Load RGB-D data
        rgb, depth = self.load_rgb_depth_data(data_path, use_varied_camera)
        
        # 使用标准命名格式
        color_path = os.path.join(data_path, 
                            'varied_camera_raw.png' if use_varied_camera else 'fixed_camera_raw.png')
        print(f"\033[92m使用标准命名格式进行对象检测: {color_path}\033[0m")
        
        # 保存RGB和深度图像到输出目录，以便后续处理
        output_rgb_path = os.path.join(output_dir, 'varied_camera_raw.png' if use_varied_camera else 'fixed_camera_raw.png')
        output_depth_path = os.path.join(output_dir, 'varied_camera_depth.npy' if use_varied_camera else 'fixed_camera_depth.npy')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存RGB图像
        cv2.imwrite(output_rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # 保存深度图像
        np.save(output_depth_path, depth)
        
        print(f"\033[92m已保存RGB图像到: {output_rgb_path}\033[0m")
        print(f"\033[92m已保存深度图像到: {output_depth_path}\033[0m")
        
        # 检测对象
        bboxes, masks = self.detect_objects(color_path, output_dir)
        
        # Generate point cloud
        points = self.depth_to_pointcloud(depth)
        
        # Generate keypoints
        keypoints ,projected_img, keypoints_2d = self.generate_keypoints(rgb, points, masks)
        
        # Save visualization - 如果不生成约束，将在后面保存到场景目录
        if generate_constraints:
            self.save_visualization(projected_img)
        # 如果不生成约束，可视化图像会在后面保存到场景目录
        
        # TODO transform keypoints_3d_cam to world coordinates(keypoints_3d_world)
        # keypoints_3d_world = self.transform_keypoint_camera_to_world(keypoints)
        
        # 准备初始metadata
        metadata = {
            'init_keypoint_positions': keypoints,
            'num_keypoints': len(keypoints)
        }
        
        # Add 2D keypoint coordinates to metadata if available
        if keypoints_2d is not None:
            metadata['keypoints_2d_coordinates'] = keypoints_2d
            print(f'{bcolors.OKGREEN}Added {len(keypoints_2d)} 2D keypoint coordinates to metadata{bcolors.ENDC}')
        
        # Generate constraints
        if generate_constraints:
            if instruction is None:
                raise ValueError("instruction is required when generate_constraints=True")
            rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
            print(f'{bcolors.HEADER}Constraints generated and saved in {rekep_program_dir}{bcolors.ENDC}')
        else:
            # 即使不生成约束，也要保存metadata到output_dir（v2c_demo场景目录）
            from datetime import datetime
            if instruction:
                fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
            else:
                fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_vision_task"
            task_dir = os.path.join(output_dir, fname)
            
            print(f"\033[94m正在创建任务目录: {task_dir}\033[0m")
            try:
                os.makedirs(task_dir, exist_ok=True)
                print(f"\033[92m任务目录创建成功: {task_dir}\033[0m")
            except Exception as e:
                print(f"\033[91m创建任务目录失败: {e}\033[0m")
                raise
            
            # 保存查询图像到场景目录
            image_path = os.path.join(task_dir, 'query_img.png')
            cv2.imwrite(image_path, projected_img[..., ::-1])
            
            # 保存投影图像到场景目录
            projected_img_path = os.path.join(task_dir, 'projected_keypoints.png')
            cv2.imwrite(projected_img_path, cv2.cvtColor(projected_img, cv2.COLOR_RGB2BGR))
            
            # 保存metadata到场景目录
            import json
            for k, v in metadata.items():
                if isinstance(v, np.ndarray):
                    metadata[k] = v.tolist()
            with open(os.path.join(task_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata saved to {os.path.join(task_dir, 'metadata.json')}")
            print(f"Projected keypoints image saved to {projected_img_path}")
            
            rekep_program_dir = task_dir
        
        total_time = time.time() - start_time
        print(f"\033[92mTotal processing time: {total_time:.2f} seconds\033[0m")
        
        return rekep_program_dir

# Convenience functions for backward compatibility and ease of use
def create_vision_processor(config_path: str = "./configs/config.yaml",
                          visualize: bool = True,
                          camera_intrinsics: Optional[CameraIntrinsics] = None) -> VisionProcessor:
    """Create and return a VisionProcessor instance.
    
    Args:
        config_path: Path to configuration file
        visualize: Whether to enable visualization
        camera_intrinsics: Camera intrinsic parameters
    
    Returns:
        VisionProcessor: Initialized vision processor instance
    """
    return VisionProcessor(
        config_path=config_path,
        visualize=visualize,
        camera_intrinsics=camera_intrinsics
    )


def process_vision_task(instruction: Optional[str] = None,
                       data_path: str = "./data/realsense_captures",
                       obj_list: Optional[str] = None,
                       visualize: bool = True,
                       config_path: str = "./configs/config.yaml") -> str:
    """Process a vision task for robotic manipulation.
    
    Args:
        instruction: Task instruction for the robot
        data_path: Path to directory containing RGB-D data
        obj_list: Optional list of objects to detect
        visualize: Whether to visualize keypoints
        config_path: Path to configuration file
    
    Returns:
        str: Path to the generated ReKep program directory
    """
    processor = VisionProcessor(config_path=config_path, visualize=visualize)
    return processor.process_vision_task(
        instruction=instruction,
        data_path=data_path,
        obj_list=obj_list
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision Processor for ReKep Robotic Manipulation")
    parser.add_argument('--instruction', type=str, required=False, 
                       help='Instruction for the task')
    parser.add_argument('--obj_list', type=str, required=False, 
                       help='String list of objects to detect')
    parser.add_argument('--data_path', type=str, required=False, 
                       help='Path to directory containing RGB-D frames')
    parser.add_argument('--visualize', action='store_true', 
                       help='Visualize keypoints on the image')
    parser.add_argument('--config', type=str, default="./configs/config.yaml",
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Set default values
    instruction = args.instruction or "Put the corn in the frying pan."
    data_path = args.data_path or "./data/realsense_captures"
    visualize = args.visualize if args.visualize is not None else True
    
    # Process the vision task
    rekep_program_dir = process_vision_task(
        instruction=instruction,
        data_path=data_path,
        obj_list=args.obj_list,
        visualize=visualize,
        config_path=args.config
    )
    
    print(f"\033[92mReKep program directory: {rekep_program_dir}\033[0m")