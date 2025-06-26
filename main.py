#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
ReKep机器人操作主程序
整合相机、视觉处理、动作规划和执行的完整流程

Author: tailong-wu
Date: 2025
"""

import os
import sys
import json
import time
import traceback
import logging
import numpy as np
import requests
import yaml
from scipy.spatial.transform import Rotation as R
from typing import Optional, List, Dict, Any
from piper_sdk import C_PiperInterface_V2
# from env.piper_mock_interface import Mock_C_PiperInterface_V2 as C_PiperInterface_V2 # 先进行无实机调试

# 导入自定义模块
from real_camera_refactored import RealSenseCamera, capture_single_frame
from vision_processor import VisionProcessor
from action_processor import ActionProcessor
from keypoint_cotracker import CoTrackerClient


class ReKepRobotSystem:
    """ReKep机器人操作系统主类"""
    
    def __init__(self, 
                 instruction: str = "Put the corn in the frying pan.",
                 config_path: str = "./configs/config.yaml",
                 data_path: str = "./data/realsense_captures",
                 output_dir: str = "./outputs",
                 ik_api_url: str = "http://localhost:8000/solve-ik/",
                 cotracker_api_url: str = "http://localhost:5000",
                 debug_mode: bool = False,
                 log_level: str = "INFO",
                 use_cache: bool = True,
                 use_existing_vlm_query: Optional[str] = None):
        """
        初始化ReKep机器人系统
        
        Args:
            instruction: 任务指令
            config_path: 配置文件路径
            data_path: 数据保存路径
            output_dir: 输出目录
            ik_api_url: IK求解器API地址
            cotracker_api_url: CoTracker API地址
            debug_mode: 是否启用调试模式（显示详细错误信息）
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
            use_cache: 是否使用缓存的vlm_query，True为使用历史文件夹，False为生成新的
            use_existing_vlm_query: 指定使用的vlm_query目录路径，如果为None则自动检测最新的
        """
        self.instruction = instruction
        self.config_path = config_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.ik_api_url = ik_api_url
        self.cotracker_api_url = cotracker_api_url
        self.debug_mode = debug_mode
        self.use_cache = use_cache
        self.use_existing_vlm_query = use_existing_vlm_query
        
        # 创建必要的目录
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 配置日志系统
        self._setup_logging(log_level)
        
        # 初始化组件
        self.camera = None
        self.vision_processor = None
        self.action_processor = None
        self.cotracker_client = None
        self.piper_robot = None
        
        # 状态变量
        self.current_stage = 1
        self.rekep_program_dir = None
        self.keypoints_2d = None
        self.total_stages = 0
        
        self.logger.info("初始化ReKep机器人系统")
        print(f"\033[92m初始化ReKep机器人系统\033[0m")
        print(f"任务指令: {self.instruction}")
    
    def _setup_logging(self, log_level: str):
        """配置日志系统"""
        # 创建日志目录
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志格式
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        
        # 创建logger
        self.logger = logging.getLogger('ReKepRobotSystem')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 文件处理器
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"rekep_robot_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(file_handler)
        
        # 控制台处理器（仅在调试模式下显示详细信息）
        if self.debug_mode:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(console_handler)
    
    def _handle_error(self, error: Exception, context: str, reraise: bool = False):
        """统一的错误处理方法"""
        error_msg = f"{context}: {str(error)}"
        
        # 记录到日志
        self.logger.error(error_msg, exc_info=True)
        
        # 控制台输出
        print(f"\033[91m{error_msg}\033[0m")
        
        # 调试模式下显示详细堆栈信息
        if self.debug_mode:
            print(f"\033[93m详细错误信息:\033[0m")
            print(f"\033[93m错误类型: {type(error).__name__}\033[0m")
            print(f"\033[93m错误位置: {traceback.format_exc()}\033[0m")
        
        if reraise:
            raise error
    
    def initialize_components(self) -> bool:
        """初始化所有组件"""
        try:
            self.logger.info("开始初始化组件")
            print("\033[94m正在初始化组件...\033[0m")
            
            # 初始化相机
            self.logger.debug("初始化相机")
            self.camera = RealSenseCamera(save_dir=self.data_path)
            
            # 初始化视觉处理器
            self.logger.debug("初始化视觉处理器")
            self.vision_processor = VisionProcessor(
                config_path=self.config_path,
                camera_config_path="./configs/camera_config.yaml",
                visualize=True
            )
            
            # 初始化动作处理器
            self.logger.debug("初始化动作处理器")
            self.action_processor = ActionProcessor(
                config_path=self.config_path,
                camera_config_path="./configs/camera_config.yaml",
                visualize=False,
                cotracker_api_url=self.cotracker_api_url,
                camera_instance=self.camera
            )
            
            # 初始化关键点跟踪器
            self.logger.debug("初始化关键点跟踪器")
            self.cotracker_client = CoTrackerClient(
                base_url=self.cotracker_api_url,
                camera_config_path="./configs/camera_config.yaml"
            )
            
            # 初始化机器人
            self.logger.debug("初始化机器人接口")
            self.piper_robot = C_PiperInterface_V2("can0")
            self.piper_robot.ConnectPort()
            self._enable_robot()
            
            self.logger.info("所有组件初始化完成")
            print("\033[92m所有组件初始化完成\033[0m")
            return True
            
        except Exception as e:
            self._handle_error(e, "组件初始化失败")
            return False
    
    def _enable_robot(self):
        """使能机械臂"""
        print("\033[94m正在使能机械臂...\033[0m")
        enable_flag = False
        timeout = 10
        start_time = time.time()
        
        while not enable_flag:
            elapsed_time = time.time() - start_time
            enable_flag = (
                self.piper_robot.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and
                self.piper_robot.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and
                self.piper_robot.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and
                self.piper_robot.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and
                self.piper_robot.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and
                self.piper_robot.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            )
            
            self.piper_robot.EnableArm(7)
            self.piper_robot.GripperCtrl(0, 1000, 0x01, 0)
            
            if elapsed_time > timeout:
                raise RuntimeError("机械臂使能超时")
            
            if not enable_flag:
                time.sleep(1)
        
        print("\033[92m机械臂使能成功\033[0m")
        
        # 机械臂回零点
        self._reset_to_home()
    
    def _reset_to_home(self):
        """将机械臂回到零点位置"""
        print("\033[94m正在将机械臂回到零点位置...\033[0m")
        
        try:
            # PiPER机械臂默认复位关节位置 (弧度)
            reset_joint_positions_rad = np.array([
                -86.853, -1.560, -0.762, 2.939, 22.294, 0.000
            ]) * np.pi / (180 * 1000)  # 从0.001度转换为弧度
            
            # 转换为控制器需要的单位 (0.001度)
            joint_factor = 180 * 1000 / np.pi
            reset_joint_positions_mdeg = [round(angle * joint_factor) for angle in reset_joint_positions_rad]
            
            print(f"目标零点位置 (rad): {reset_joint_positions_rad}")
            print(f"发送给控制器 (0.001deg): {reset_joint_positions_mdeg}")
            
            # 切换到关节控制模式
            self.piper_robot.MotionCtrl_2(0x01, 0x01, 100, 0x00)
            
            # 等待机械臂到达零点位置
            self._wait_for_joints(reset_joint_positions_rad, reset_joint_positions_mdeg, gripper_val=0)
            
            print("\033[92m机械臂已回到零点位置\033[0m")
            
        except Exception as e:
            print(f"\033[91m机械臂回零点失败: {e}\033[0m")
            raise
    
    def capture_initial_scene(self) -> bool:
        """捕获初始场景图像和深度数据"""
        print("\033[94m正在捕获初始场景...\033[0m")
        
        try:
            # 检查相机是否已经在流式传输，如果没有则启动
            if not self.camera.is_streaming:
                if not self.camera.start_streaming():
                    print("\033[91m无法启动相机\033[0m")
                    return False
            
            self.camera.print_intrinsics()
            
            # 等待相机流稳定
            time.sleep(3)
            
            # 尝试多次获取帧，增加超时时间
            max_retries = 5
            for attempt in range(max_retries):
                print(f"尝试获取帧 {attempt + 1}/{max_retries}...")
                color_image, depth_image = self.camera.get_frames(timeout_ms=5000)  # 增加到5秒超时
                
                if color_image is not None and depth_image is not None:
                    print("\033[92m成功获取图像数据\033[0m")
                    break
                    
                if attempt < max_retries - 1:
                    print(f"第{attempt + 1}次尝试失败，等待1秒后重试...")
                    time.sleep(1)
            else:
                print("\033[91m多次尝试后仍无法获取图像数据\033[0m")
                return False
            
            # 保存图像
            saved_files = self.camera.save_images(color_image, depth_image)
            print(f"\033[92m场景图像已保存: {saved_files}\033[0m")
            
            return True
                
        except Exception as e:
            print(f"\033[91m捕获场景失败: {e}\033[0m")
            return False
    
    def process_vision_task(self) -> bool:
        """处理视觉任务，获取关键点和约束"""
        print("\033[94m正在处理视觉任务...\033[0m")
        
        try:
            if not self.use_cache:
                # 不使用缓存，直接生成新的vlm_query
                print("\033[93m不使用缓存，重新生成ReKep程序\033[0m")
                self.rekep_program_dir = self.vision_processor.process_vision_task(
                    instruction=self.instruction,
                    data_path=self.data_path,
                    use_varied_camera=True,
                    output_dir=self.data_path
                )
            else:
                # 使用缓存模式
                # 优先使用指定的vlm_query目录
                if self.use_existing_vlm_query and os.path.exists(self.use_existing_vlm_query):
                    self.rekep_program_dir = self.use_existing_vlm_query
                    print(f"\033[92m使用指定的ReKep程序目录: {self.rekep_program_dir}\033[0m")
                else:
                    # 检查是否存在本地已求解的vlm_query目录
                    vlm_query_dir = "./vlm_query"
                    if os.path.exists(vlm_query_dir):
                        # 获取最新的vlm_query子目录
                        vlm_dirs = [os.path.join(vlm_query_dir, d) for d in os.listdir(vlm_query_dir)
                                   if os.path.isdir(os.path.join(vlm_query_dir, d))]
                        
                        if vlm_dirs:
                            # 使用最新的目录
                            self.rekep_program_dir = max(vlm_dirs, key=os.path.getmtime)
                            print(f"\033[92m使用本地已求解的ReKep程序目录: {self.rekep_program_dir}\033[0m")
                        else:
                            print("\033[93m未找到vlm_query子目录，重新生成ReKep程序\033[0m")
                            # 重新生成ReKep程序
                            self.rekep_program_dir = self.vision_processor.process_vision_task(
                                instruction=self.instruction,
                                data_path=self.data_path,
                                use_varied_camera=True,
                                output_dir=self.data_path
                            )
                    else:
                        print("\033[93m未找到vlm_query目录，重新生成ReKep程序\033[0m")
                        # 重新生成ReKep程序
                        self.rekep_program_dir = self.vision_processor.process_vision_task(
                            instruction=self.instruction,
                            data_path=self.data_path,
                            use_varied_camera=True,
                            output_dir=self.data_path
                        )
            
            # 读取metadata获取2D关键点信息
            metadata_path = os.path.join(self.rekep_program_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.keypoints_2d = metadata.get('keypoints_2d_coordinates', [])
                    self.total_stages = metadata.get('num_stages', 1)
                    print(f"\033[92m获取到 {len(self.keypoints_2d)} 个2D关键点\033[0m")
                    print(f"\033[92m总共 {self.total_stages} 个阶段\033[0m")
            
            print(f"\033[92m视觉处理完成，ReKep程序目录: {self.rekep_program_dir}\033[0m")
            return True
            
        except Exception as e:
            print(f"\033[91m视觉处理失败: {e}\033[0m")
            return False
    
    def initialize_keypoint_tracking(self) -> bool:
        """初始化关键点跟踪"""
        print("\033[94m正在初始化关键点跟踪...\033[0m")
        
        try:
            # 转换关键点格式 [y,x] -> [x,y]
            if self.keypoints_2d:
                keypoints_xy = np.array([[kp[1], kp[0]] for kp in self.keypoints_2d], dtype=np.float32)
                print(f"\033[92m转换关键点格式: {self.keypoints_2d} -> {keypoints_xy.tolist()}\033[0m")
            else:
                # 使用默认关键点
                keypoints_xy = np.array([[320, 200], [320, 280]], dtype=np.float32)
                print("\033[93m使用默认关键点\033[0m")
             
            # 获取机械臂末端位置并转换为图像像素坐标
            try:
                # 获取末端位置（世界坐标系）
                ee_pos = self.action_processor.get_ee_location()
                print(f"\033[92m获取到机械臂末端位置(世界坐标系): {ee_pos}\033[0m")
                
                # 使用已初始化的相机捕获当前帧
                color_image, depth_image = self.camera.get_frames()
                if color_image is None:
                    print("\033[91m无法捕获当前帧\033[0m")
                else:
                        # 加载相机内参
                        intrinsics_matrix, depth_scale = self.action_processor.load_camera_intrinsics()
                        fx = intrinsics_matrix[0, 0]
                        fy = intrinsics_matrix[1, 1]
                        cx = intrinsics_matrix[0, 2]
                        cy = intrinsics_matrix[1, 2]
                        
                        # 加载相机外参（世界坐标系到相机坐标系的变换）
                        camera_extrinsics = self.action_processor.load_camera_extrinsics()
                        
                        # 创建末端位置的齐次坐标
                        ee_pos_homogeneous = np.append(ee_pos, 1.0)
                        
                        # 将末端位置从世界坐标系转换到相机坐标系
                        # 注意：camera_extrinsics是从相机到世界的变换，需要求逆
                        camera_to_world = camera_extrinsics
                        world_to_camera = np.linalg.inv(camera_to_world)
                        ee_pos_camera_homogeneous = world_to_camera @ ee_pos_homogeneous
                        ee_pos_camera = ee_pos_camera_homogeneous[:3] / ee_pos_camera_homogeneous[3]
                        
                        print(f"\033[92m机械臂末端位置(相机坐标系): {ee_pos_camera}\033[0m")
                        
                        # 将3D点投影到图像平面
                        if ee_pos_camera[2] > 0:  # 确保点在相机前方
                            ee_pixel_x = int(cx + (ee_pos_camera[0] * fx) / ee_pos_camera[2])
                            ee_pixel_y = int(cy + (ee_pos_camera[1] * fy) / ee_pos_camera[2])
                            
                            # 确保像素坐标在图像范围内
                            if 0 <= ee_pixel_x < 640 and 0 <= ee_pixel_y < 480:
                                # 添加末端位置作为关键点
                                ee_keypoint = np.array([[ee_pixel_x, ee_pixel_y]], dtype=np.float32)
                                keypoints_xy = np.vstack([keypoints_xy, ee_keypoint])
                                print(f"\033[92m添加机械臂末端位置作为关键点: [{ee_pixel_x}, {ee_pixel_y}]\033[0m")
                            else:
                                print(f"\033[93m机械臂末端位置投影超出图像范围: [{ee_pixel_x}, {ee_pixel_y}]\033[0m")
                        else:
                            print(f"\033[93m机械臂末端位置在相机后方，无法投影\033[0m")
            except Exception as e:
                print(f"\033[91m获取机械臂末端位置失败: {e}\033[0m")
            
            # 使用环境中的 register_keypoints_for_tracking 方法进行关键点注册
            if hasattr(self.action_processor, 'env') and hasattr(self.action_processor.env, 'register_keypoints_for_tracking'):
                # 设置环境中的 keypoints_2d
                self.action_processor.env.keypoints_2d = keypoints_xy.tolist()
                
                # 使用 ReKepEnv 的 register_keypoints_for_tracking 方法
                success = self.action_processor.env.register_keypoints_for_tracking()
                if success:
                    print("\033[92m关键点注册成功\033[0m")
                    return True
                else:
                    print("\033[91m关键点注册失败\033[0m")
                    return False
            else:
                # 回退到使用 CoTrackerClient 直接注册
                print("\033[93m环境不支持关键点注册，使用 CoTrackerClient 直接注册\033[0m")
                
                # 使用已初始化的相机捕获当前帧
                color_image, depth_image = self.camera.get_frames()
                if color_image is None:
                    print("\033[91m无法捕获当前帧\033[0m")
                    return False
                
                # 注册关键点
                status_code, response = self.cotracker_client.register_frame(color_image, keypoints_xy)
                if status_code == 200:
                    print("\033[92m关键点注册成功\033[0m")
                    return True
                else:
                    print(f"\033[91m关键点注册失败: {status_code}, {response}\033[0m")
                    return False
                
        except Exception as e:
            print(f"\033[91m关键点跟踪初始化失败: {e}\033[0m")
            return False
    
    def solve_stage_actions(self, stage: int) -> Optional[str]:
        """求解指定阶段的动作序列"""
        print(f"\033[94m正在求解第 {stage} 阶段动作...\033[0m")
        
        try:
            # 使用动作处理器生成动作序列
            action_file_path = self.action_processor.process_action_task(
                instruction=self.instruction,
                rekep_program_dir=self.rekep_program_dir,
                stage=stage,
                output_dir=self.output_dir
            )
            
            if action_file_path:
                print(f"\033[92m第 {stage} 阶段动作序列已生成: {action_file_path}\033[0m")
                return action_file_path
            else:
                print(f"\033[93m第 {stage} 阶段已完成或无需动作\033[0m")
                return None
                
        except Exception as e:
            print(f"\033[91m第 {stage} 阶段动作求解失败: {e}\033[0m")
            return None
    
    def convert_to_joint_actions(self, action_file_path: str) -> bool:
        """将末端位姿动作序列转换为关节动作序列"""
        print("\033[94m正在转换为关节动作序列...\033[0m")
        
        try:
            # 读取动作数据
            with open(action_file_path, 'r') as f:
                action_data = json.load(f)
            
            ee_action_seq = action_data['ee_action_seq']
            
            # 准备API请求
            request_data = {"ee_action_seq": ee_action_seq}
            
            # 发送请求到IK求解器
            start_time = time.time()
            response = requests.post(self.ik_api_url, json=request_data, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                # 保存关节动作序列
                joint_action_path = os.path.join(self.output_dir, 'joint_action.json')
                with open(joint_action_path, 'w') as f:
                    json.dump(result, f, indent=4)
                
                print(f"\033[92mIK求解完成，耗时: {end_time - start_time:.2f}秒\033[0m")
                print(f"\033[92m关节动作序列已保存: {joint_action_path}\033[0m")
                return True
            else:
                print(f"\033[91mIK求解失败: {response.status_code}, {response.text}\033[0m")
                return False
                
        except Exception as e:
            print(f"\033[91m关节动作转换失败: {e}\033[0m")
            return False
    
    def execute_joint_actions(self) -> bool:
        """执行关节动作序列"""
        print("\033[94m正在执行关节动作序列...\033[0m")
        
        try:
            # 读取关节动作数据
            joint_action_path = os.path.join(self.output_dir, 'joint_action.json')
            with open(joint_action_path, 'r') as f:
                action_data = json.load(f)
            
            joint_action_seq = action_data['joint_angle_seq']
            joint_factor = 180 * 1000 / np.pi  # 转换因子: rad -> 0.001deg
            
            # 切换到关节控制模式
            self.piper_robot.MotionCtrl_2(0x01, 0x01, 100, 0x00)
            
            for i, action in enumerate(joint_action_seq):
                print(f"\n\033[96m执行第 {i+1}/{len(joint_action_seq)} 个动作\033[0m")
                
                joint_angles_rad = np.array(action[:6])
                joints_mdeg = [round(angle * joint_factor) for angle in joint_angles_rad]
                
                print(f"目标关节角度 (rad): {joint_angles_rad}")
                print(f"发送给控制器 (0.001deg): {joints_mdeg}")
                
                # 等待机械臂到达目标位置
                self._wait_for_joints(joint_angles_rad, joints_mdeg, gripper_val=0)
                
                # 执行动作后进行图像采集和关键点跟踪
                self._track_keypoints_after_action(i + 1)
                
                # 每个动作执行后更新机器人状态
                self._update_robot_state()
                
                # 等待用户确认继续
                input("\033[93m按 Enter 键继续下一个动作...\033[0m")
            
            print("\033[92m所有动作执行完毕\033[0m")
            
            # 更新 robot_state.json 文件
            self._update_robot_state()
            
            return True
            
        except Exception as e:
            print(f"\033[91m动作执行失败: {e}\033[0m")
            return False
    
    def _wait_for_joints(self, target_joints_rad: np.ndarray, joints_mdeg: List[int], 
                        gripper_val: int, tolerance: float = 0.1):
        """等待机械臂到达指定关节角度"""
        # 尝试使用environment_new.py中的方法
        if hasattr(self.action_processor, 'env') and hasattr(self.action_processor.env, 'robot') and \
           hasattr(self.action_processor.env.robot, '_wait_for_joints'):
            # 使用封装好的方法等待关节到达目标位置
            self.action_processor.env.robot._wait_for_joints(target_joints_rad, joints_mdeg, gripper_val, tolerance)
            return
        
        # 备用方案：使用原有逻辑
        factor_to_rad = np.pi / (180 * 1000)
        
        while True:
            # 控制夹爪
            if gripper_val == 0:
                self.piper_robot.GripperCtrl(50*1000, 1000, 0x01, 0)  # 张开
            else:
                self.piper_robot.GripperCtrl(0*1000, 1000, 0x01, 0)   # 闭合
            
            # 发送关节控制指令
            self.piper_robot.JointCtrl(
                joints_mdeg[0], joints_mdeg[1], joints_mdeg[2],
                joints_mdeg[3], joints_mdeg[4], joints_mdeg[5]
            )
            
            # 获取当前关节角度
            joint_msgs = self.piper_robot.GetArmJointMsgs()
            current_joints_rad = np.array([
                joint_msgs.joint_state.joint_1 * factor_to_rad,
                joint_msgs.joint_state.joint_2 * factor_to_rad,
                joint_msgs.joint_state.joint_3 * factor_to_rad,
                joint_msgs.joint_state.joint_4 * factor_to_rad,
                joint_msgs.joint_state.joint_5 * factor_to_rad,
                joint_msgs.joint_state.joint_6 * factor_to_rad,
            ])
            
            joint_error = np.linalg.norm(current_joints_rad - target_joints_rad)
            print(f"\r关节角度误差: {joint_error:.4f}", end="", flush=True)
            
            if joint_error < tolerance:
                print("\n\033[92m已到达目标关节角度\033[0m")
                break
            
            time.sleep(0.1)
    
    def _track_keypoints_after_action(self, action_index: int):
        """动作执行后进行关键点跟踪"""
        try:
            print(f"\033[94m正在跟踪第 {action_index} 个动作后的关键点...\033[0m")
            
            # 使用环境中的 track_keypoints 方法进行关键点跟踪
            if hasattr(self.action_processor, 'env') and hasattr(self.action_processor.env, 'track_keypoints'):
                # 使用 ReKepEnv 的 track_keypoints 方法
                success, tracked_keypoints = self.action_processor.env.track_keypoints()
                if success and len(tracked_keypoints) > 0:
                    print(f"\033[92m关键点跟踪成功: {tracked_keypoints}\033[0m")
                    
                    # 更新环境中的关键点
                    self.action_processor.env.update_keypoints(tracked_keypoints)
                    print(f"\033[92m已更新环境中的关键点\033[0m")
                else:
                    print("\033[93m未检测到关键点\033[0m")
            else:
                # 回退到使用 CoTrackerClient 直接跟踪
                print("\033[93m环境不支持关键点跟踪，使用 CoTrackerClient 直接跟踪\033[0m")
                
                # 使用已初始化的相机捕获当前帧
                color_image, depth_image = self.camera.get_frames()
                if color_image is None:
                    print("\033[91m无法捕获当前帧\033[0m")
                    return
                
                # 跟踪关键点
                status_code, response = self.cotracker_client.track_frame(color_image)
                if status_code == 200:
                    tracked_keypoints = np.array(response.get('keypoints', []))
                    if len(tracked_keypoints) > 0:
                        print(f"\033[92m关键点跟踪成功: {tracked_keypoints.tolist()}\033[0m")
                        
                        # 更新环境中的关键点
                        if hasattr(self.action_processor, 'env'):
                            self.action_processor.env.update_keypoints(tracked_keypoints.tolist())
                            print(f"\033[92m已更新环境中的关键点\033[0m")
                        
                        # 保存带关键点的图像
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(self.data_path, f'action_{action_index}_{timestamp}.png')
                        self.cotracker_client.save_frame_with_keypoints(
                            color_image, tracked_keypoints, output_path, depth_image
                        )
                    else:
                        print("\033[93m未检测到关键点\033[0m")
                else:
                    print(f"\033[91m关键点跟踪失败: {status_code}, {response}\033[0m")
                
        except Exception as e:
            print(f"\033[91m关键点跟踪出错: {e}\033[0m")
    
    def run_complete_task(self) -> bool:
        """运行完整的任务流程"""
        self.logger.info("开始ReKep机器人操作任务")
        print("\033[92m=== 开始ReKep机器人操作任务 ===\033[0m")
        
        try:
            # 1. 初始化所有组件
            self.logger.info("步骤1: 初始化所有组件")
            if not self.initialize_components():
                self.logger.error("组件初始化失败")
                return False
            
            # 2. 捕获初始场景
            self.logger.info("步骤2: 捕获初始场景")
            if not self.capture_initial_scene():
                self.logger.error("初始场景捕获失败")
                return False
            
            # 3. 处理视觉任务
            self.logger.info("步骤3: 处理视觉任务")
            if not self.process_vision_task():
                self.logger.error("视觉任务处理失败")
                return False
            
            # 4. 初始化关键点跟踪
            self.logger.info("步骤4: 初始化关键点跟踪")
            if not self.initialize_keypoint_tracking():
                self.logger.error("关键点跟踪初始化失败")
                return False
            
            # 5. 循环执行所有阶段
            self.logger.info(f"步骤5: 开始执行 {self.total_stages} 个阶段")
            for stage in range(1, self.total_stages + 1):
                self.logger.info(f"执行第 {stage}/{self.total_stages} 阶段")
                print(f"\n\033[95m=== 执行第 {stage}/{self.total_stages} 阶段 ===\033[0m")
                
                # 更新当前阶段
                self.current_stage = stage
                
                # 求解当前阶段动作
                action_file_path = self.solve_stage_actions(stage)
                if action_file_path is None:
                    self.logger.info(f"第 {stage} 阶段无需执行动作，跳过")
                    print(f"\033[93m第 {stage} 阶段无需执行动作，跳过\033[0m")
                    continue
                
                # 转换为关节动作
                if not self.convert_to_joint_actions(action_file_path):
                    self.logger.error(f"第 {stage} 阶段关节动作转换失败")
                    print(f"\033[91m第 {stage} 阶段关节动作转换失败\033[0m")
                    return False
                
                # 执行关节动作
                if not self.execute_joint_actions():
                    self.logger.error(f"第 {stage} 阶段动作执行失败")
                    print(f"\033[91m第 {stage} 阶段动作执行失败\033[0m")
                    return False
                
                self.logger.info(f"第 {stage} 阶段执行完成")
                print(f"\033[92m第 {stage} 阶段执行完成\033[0m")
            
            self.logger.info("所有阶段执行完成")
            print("\n\033[92m=== 所有阶段执行完成！ ===\033[0m")
            return True
            
        except KeyboardInterrupt:
            self.logger.warning("用户中断任务")
            print("\n\033[93m用户中断任务\033[0m")
            return False
        except Exception as e:
            self._handle_error(e, "任务执行失败")
            return False
        finally:
            self.cleanup()
    
    def _update_robot_state(self):
        """更新机器人状态信息到robot_state.json和robot_state_{stage}.json文件"""
        try:
            print("\033[94m正在更新机器人状态信息...\033[0m")
            
            # 使用environment_new.py中的方法获取和保存机器人状态，确保单位统一
            if hasattr(self.action_processor, 'env') and hasattr(self.action_processor.env, 'robot') and \
               hasattr(self.action_processor.env.robot, 'save_robot_state_to_file'):
                # 使用封装好的方法保存机器人状态
                success = self.action_processor.env.robot.save_robot_state_to_file(
                    './robot_state.json', self.current_stage)
                
                # 保存到robot_state_{stage}.json
                if success and self.rekep_program_dir:
                    stage_state_path = os.path.join(self.rekep_program_dir, f'robot_state_{self.current_stage}.json')
                    self.action_processor.env.robot.save_robot_state_to_file(stage_state_path, self.current_stage)
                
                return success
            else:
                # 备用方案：使用原有逻辑
                # 获取当前关节位置
                joint_msgs = self.piper_robot.GetArmJointMsgs()
                factor_to_rad = np.pi / (180 * 1000)
                current_joints = [
                    joint_msgs.joint_state.joint_1 * factor_to_rad,
                    joint_msgs.joint_state.joint_2 * factor_to_rad,
                    joint_msgs.joint_state.joint_3 * factor_to_rad,
                    joint_msgs.joint_state.joint_4 * factor_to_rad,
                    joint_msgs.joint_state.joint_5 * factor_to_rad,
                    joint_msgs.joint_state.joint_6 * factor_to_rad,
                ]
                
                # 获取末端位置和姿态
                ee_location = self.action_processor.get_ee_location().tolist()
                ee_euler = self.action_processor.env.get_tcp_pose()[3:6]
                ee_quat = R.from_euler('xyz', ee_euler).as_quat().tolist()  # 四元数 [x,y,z,w]
                
                # 获取机械臂原始数据
                joint_msgs = self.piper_robot.GetArmJointMsgs()
                arm_end_pose_msgs = self.piper_robot.GetArmEndPoseMsgs()
                
                # 构建机器人状态信息
                robot_state = {
                    "timestamp": time.time(),
                    "rekep_stage": self.current_stage,
                    "joint_info": {
                        "joint_positions": current_joints,
                        "reset_joint_pos": current_joints
                    },
                    "ee_info": {
                        "position": ee_location,
                        "orientation": {
                            "quaternion": ee_quat,
                            "euler": ee_euler.tolist(),
                            "unit": "radians",
                            "description": "End effector orientation in radians [rx, ry, rz]"
                        }
                    },
                    "gripper_info": {
                        "state": 0.0  # 假设夹爪状态为0（张开）
                    },
                    "safety_info": {
                        "collision_status": "false",
                        "safety_status": "normal",
                        "errors": []
                    },
                    "control_info": {
                        "control_mode": "position",
                        "operation_mode": "auto"
                    },
                    "misc": {
                        "world2robot_homo": [[0.71659986, -0.6894734, 0.10540908, 0.29309614],
                                            [-0.66892568, -0.72216724, -0.17610487, 0.49516889],
                                            [0.19754261, 0.05568588, -0.9787114, 0.42916608],
                                            [0.0, 0.0, 0.0, 1.0]]
                    },
                    "raw_data": {
                        "joint_angles_raw": {
                            "joint_1": joint_msgs.joint_state.joint_1,
                            "joint_2": joint_msgs.joint_state.joint_2,
                            "joint_3": joint_msgs.joint_state.joint_3,
                            "joint_4": joint_msgs.joint_state.joint_4,
                            "joint_5": joint_msgs.joint_state.joint_5,
                            "joint_6": joint_msgs.joint_state.joint_6,
                            "unit": "0.001 degrees"
                        },
                        "end_pose_raw": {
                            "X_axis": arm_end_pose_msgs.end_pose.X_axis,
                            "Y_axis": arm_end_pose_msgs.end_pose.Y_axis,
                            "Z_axis": arm_end_pose_msgs.end_pose.Z_axis,
                            "RX_axis": arm_end_pose_msgs.end_pose.RX_axis,
                            "RY_axis": arm_end_pose_msgs.end_pose.RY_axis,
                            "RZ_axis": arm_end_pose_msgs.end_pose.RZ_axis,
                            "position_unit": "0.001 mm",
                            "orientation_unit": "0.001 degrees"
                        }
                    }
                }
                
                # 保存到robot_state.json，保留initial_position字段
                try:
                    # 如果文件存在，读取现有的initial_position字段
                    if os.path.exists('./robot_state.json'):
                        with open('./robot_state.json', 'r') as f:
                            existing_state = json.load(f)
                            if 'initial_position' in existing_state:
                                robot_state['initial_position'] = existing_state['initial_position']
                except Exception as e:
                    print(f"\033[93m读取现有robot_state.json失败: {e}\033[0m")
                
                # 如果没有initial_position字段，使用当前状态创建
                if 'initial_position' not in robot_state:
                    robot_state['initial_position'] = {
                        "joint_positions": current_joints,
                        "ee_position": ee_location,
                        "ee_orientation": ee_euler.tolist()
                    }
                
                with open('./robot_state.json', 'w') as f:
                    json.dump(robot_state, f, indent=4)
                print("\033[92m机器人状态已保存到 robot_state.json\033[0m")
                
                # 保存到robot_state_{stage}.json
                if self.rekep_program_dir:
                    stage_state_path = os.path.join(self.rekep_program_dir, f'robot_state_{self.current_stage}.json')
                    with open(stage_state_path, 'w') as f:
                        json.dump(robot_state, f, indent=4)
                    print(f"\033[92m机器人状态已保存到 {stage_state_path}\033[0m")
            
        except Exception as e:
            print(f"\033[91m更新机器人状态失败: {e}\033[0m")
    
    def cleanup(self):
        """清理资源"""
        print("\033[94m正在清理资源...\033[0m")
        
        try:
            # 关闭ReKepEnv实例
            if hasattr(self, 'action_processor') and self.action_processor:
                if hasattr(self.action_processor, 'env') and self.action_processor.env:
                    self.action_processor.env.close()
                if hasattr(self.action_processor, 'robot_env') and self.action_processor.robot_env:
                    self.action_processor.robot_env.close()
            
            # 关闭相机
            if self.camera:
                self.camera.stop_streaming()
            
            # 关闭机器人
            if self.piper_robot:
                # 可以添加机器人复位操作
                pass
                
        except Exception as e:
            print(f"\033[91m清理资源时出错: {e}\033[0m")
        
        print("\033[92m资源清理完成\033[0m")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ReKep机器人操作系统")
    parser.add_argument('--instruction', type=str, 
                       default="Put the corn in the frying pan.",
                       help='任务指令')
    parser.add_argument('--config', type=str, 
                       default="./configs/config.yaml",
                       help='配置文件路径')
    parser.add_argument('--data_path', type=str, 
                       default="./data/realsense_captures",
                       help='数据保存路径')
    parser.add_argument('--output_dir', type=str, 
                       default="./test_outputs",
                       help='输出目录')
    parser.add_argument('--ik_api_url', type=str, 
                       default="http://localhost:8000/solve-ik/",
                       help='IK求解器API地址')
    parser.add_argument('--cotracker_api_url', type=str, 
                       default="http://localhost:5000",
                       help='CoTracker API地址')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式（显示详细错误信息和堆栈跟踪）')
    parser.add_argument('--log_level', type=str, 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='日志级别')
    parser.add_argument('--use_cache', type=bool,
                       default=True,
                       help='是否使用缓存的vlm_query，默认为True使用历史文件夹')
    parser.add_argument('--use_existing_vlm_query', type=str,
                       help='指定使用的vlm_query目录路径，如果不指定则自动检测最新的')
    
    args = parser.parse_args()
    
    # 创建并运行系统
    try:
        system = ReKepRobotSystem(
            instruction=args.instruction,
            config_path=args.config,
            data_path=args.data_path,
            output_dir=args.output_dir,
            ik_api_url=args.ik_api_url,
            cotracker_api_url=args.cotracker_api_url,
            debug_mode=args.debug,
            log_level=args.log_level,
            use_cache=args.use_cache,
            use_existing_vlm_query=args.use_existing_vlm_query
        )
        
        success = system.run_complete_task()
        
        if success:
            print("\033[92m任务成功完成！\033[0m")
            sys.exit(0)
        else:
            print("\033[91m任务执行失败！\033[0m")
            sys.exit(1)
            
    except Exception as e:
        print(f"\033[91m系统启动失败: {e}\033[0m")
        if args.debug:
            print(f"\033[93m详细错误信息:\033[0m")
            print(f"\033[93m错误类型: {type(e).__name__}\033[0m")
            print(f"\033[93m错误位置: {traceback.format_exc()}\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()