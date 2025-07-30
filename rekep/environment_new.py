#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
ReKep环境模块 - 真实机器人控制版本
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
import yaml
import cv2
import pyrealsense2 as rs
from typing import Optional, List, Dict, Any, Tuple, Union
from scipy.spatial.transform import Rotation as R
from piper_sdk import C_PiperInterface_V2


class RobotController:
    """Piper机器人控制器，实现与真实Piper机器人的交互"""
    
    def __init__(self, interface: str = "can0", test_mode: bool = False):
        """初始化Piper机器人控制器
        
        Args:
            interface: CAN 接口名称 (默认 "can0")
            test_mode: 是否启用测试模式（使用 Mock）
        """
        self.interface = interface
        
        # TCP偏移距离：夹爪底座到夹爪顶端的距离（米）
        # self.TCP_OFFSET_DISTANCE = 0.135283
        self.TCP_OFFSET_DISTANCE = 0.13
        
        # 初始化机器人状态
        self.joint_limits = {
            'lower': np.array([-2.6179, 0, -2.967, -1.745, -1.22, -2.09439]),
            'upper': np.array([2.6179, 3.14, 0, 1.745, 1.22, 2.09439])
        }
        self.ee_pose = np.zeros(7)  # [x, y, z, qx, qy, qz, qw]
        self.joint_positions = np.zeros(6)
        
        # 直接初始化 piper 对象，不再使用 piper_controller
        from env.piper_mock_interface import Mock_C_PiperInterface_V2
        if test_mode:
            self.piper = Mock_C_PiperInterface_V2(interface)
        else:
            self.piper = C_PiperInterface_V2(interface)
        self.piper.ConnectPort()  # 连接到机械臂接口
        
        self.enable_robot()
        self._update_robot_state()
        
        # 默认：设置关节控制模式
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)  # 切换到关节控制模式
        
        # 关节控制相关参数
        self.joint_factor = 180 * 1000 / np.pi  # 弧度转换为0.001度的因子
        
    def get_robot_pose_and_joints(self):
        """获取机械臂的末端位姿和关节角度
        
        Returns:
            Tuple[List[float], List[float]]: 返回末端位姿和关节角度的元组 (end_pose, joint_positions)
        """
        try:
            # 获取当前末端位姿（夹爪底座位置）
            gripper_base_pose = self._get_raw_gripper_base_pose()
            
            # 应用TCP偏移修正：沿着法向量延伸
            tcp_pose = self._apply_tcp_offset(gripper_base_pose)
            
            # 获取当前关节角度
            joint_msgs = self.piper.GetArmJointMsgs()
            joint_positions = [
                joint_msgs.joint_state.joint_1 * np.pi / (180 * 1000),  # 转换为弧度
                joint_msgs.joint_state.joint_2 * np.pi / (180 * 1000),
                joint_msgs.joint_state.joint_3 * np.pi / (180 * 1000),
                joint_msgs.joint_state.joint_4 * np.pi / (180 * 1000),
                joint_msgs.joint_state.joint_5 * np.pi / (180 * 1000),
                joint_msgs.joint_state.joint_6 * np.pi / (180 * 1000),
            ]

            return tcp_pose, joint_positions

        except Exception as e:
            print(f"获取机械臂数据时发生错误: {e}")
            return None, None
    
    def _apply_tcp_offset(self, gripper_base_pose: List[float]) -> List[float]:
        """应用TCP偏移修正：沿着法向量延伸TCP偏移距离
        
        Args:
            gripper_base_pose: 夹爪底座位姿 [x, y, z, rx, ry, rz] (位置单位：米，角度单位：度)
            
        Returns:
            List[float]: 修正后的TCP位姿 [x, y, z, rx, ry, rz]
        """
        try:
            # 提取位置和姿态
            position = np.array(gripper_base_pose[:3])
            rotation_deg = np.array(gripper_base_pose[3:6])
            
            # 将角度转换为弧度
            rotation_rad = rotation_deg * np.pi / 180.0
            
            # 使用欧拉角创建旋转矩阵（XYZ顺序）
            from scipy.spatial.transform import Rotation as R
            rotation_matrix = R.from_euler('xyz', rotation_rad).as_matrix()
            
            # 计算法向量（Z轴方向）
            # 在机器人坐标系中，末端执行器的法向量通常是Z轴方向
            normal_vector = rotation_matrix[:, 2]  # 取旋转矩阵的第三列（Z轴）
            
            # 沿着法向量延伸TCP偏移距离
            tcp_position = position + normal_vector * (self.TCP_OFFSET_DISTANCE)
            
            # 返回修正后的位姿（姿态保持不变）
            tcp_pose = [tcp_position[0], tcp_position[1], tcp_position[2], 
                       rotation_deg[0], rotation_deg[1], rotation_deg[2]]
            
            return tcp_pose
            
        except Exception as e:
            print(f"应用TCP偏移修正时发生错误: {e}")
            # 如果修正失败，返回原始位姿
            return gripper_base_pose
    
    def _remove_tcp_offset(self, tcp_pose: List[float]) -> List[float]:
        """移除TCP偏移修正：将TCP位姿转换回夹爪底座位姿
        
        Args:
            tcp_pose: TCP位姿 [x, y, z, rx, ry, rz] (位置单位：米，角度单位：度)
            
        Returns:
            List[float]: 夹爪底座位姿 [x, y, z, rx, ry, rz]
        """
        try:
            # 提取位置和姿态
            position = np.array(tcp_pose[:3])
            rotation_deg = np.array(tcp_pose[3:6])
            
            # 将角度转换为弧度
            rotation_rad = rotation_deg * np.pi / 180.0
            
            # 使用欧拉角创建旋转矩阵（XYZ顺序）
            from scipy.spatial.transform import Rotation as R
            rotation_matrix = R.from_euler('xyz', rotation_rad).as_matrix()
            
            # 计算法向量（Z轴方向）
            normal_vector = rotation_matrix[:, 2]  # 取旋转矩阵的第三列（Z轴）
            
            # 沿着法向量反向移动TCP偏移距离
            gripper_base_position = position - normal_vector * self.TCP_OFFSET_DISTANCE
            
            # 返回夹爪底座位姿（姿态保持不变）
            gripper_base_pose = [gripper_base_position[0], gripper_base_position[1], gripper_base_position[2], 
                               rotation_deg[0], rotation_deg[1], rotation_deg[2]]
            
            return gripper_base_pose
            
        except Exception as e:
            print(f"移除TCP偏移修正时发生错误: {e}")
            # 如果修正失败，返回原始位姿
            return tcp_pose
    
    def action(self, end_pose: List[float], grip_angle: int, grip_force: int = 1000):
        """移动机械臂到指定的末端位姿，注意action的end_pose需要同时提供夹爪控制
        
        Args:
            end_pose: 包含TCP末端位姿的列表，格式为 [X, Y, Z, RX, RY, RZ]（夹爪顶端位置）
            grip_angle: 夹爪角度
            grip_force: 手爪控制力（默认 1000）
        """
        if len(end_pose) != 6:
            raise ValueError("end_pose 必须包含 6 个元素：[X, Y, Z, RX, RY, RZ]")
        
        # 将TCP位姿转换为夹爪底座位姿
        gripper_base_pose = self._remove_tcp_offset(end_pose)
        
        factor = 1000  # 缩放因子
        X, Y, Z, RX, RY, RZ = [round(val * factor) for val in gripper_base_pose]

        print(f"TCP位姿: {end_pose}")
        print(f"转换为夹爪底座位姿: {gripper_base_pose}")
        print(f"移动机械臂到位置: X={X}, Y={Y}, Z={Z}, RX={RX}, RY={RY}, RZ={RZ}")
        print(f"夹爪角度和力矩: 角度={grip_angle}，力矩={grip_force}")
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        self.piper.GripperCtrl(abs(grip_angle), grip_force, 0x01, 0)
        print("机械臂已到达目标位置")
    
    def send_pose_to_robot(self, end_pose):
        """发送位姿到机器人控制器执行
        暂未使用，使用pose控制PiPER时有一些问题。
        Args:
            end_pose: 要执行的TCP位姿 (x, y, z, Rx, Ry, Rz)（夹爪顶端位置）
        """
        try:
            if len(end_pose) != 6:
                raise ValueError("end_pose 必须包含 6 个元素：[X, Y, Z, RX, RY, RZ]")
            
            # 将TCP位姿转换为夹爪底座位姿
            gripper_base_pose = self._remove_tcp_offset(end_pose)
            
            factor = 1000  # 缩放因子
            X, Y, Z, RX, RY, RZ = [round(val * factor) for val in gripper_base_pose]
            
            print(f"TCP位姿: {end_pose}")
            print(f"转换为夹爪底座位姿: {gripper_base_pose}")
            print(f"移动机械臂到位置: X={X}, Y={Y}, Z={Z}, RX={RX}, RY={RY}, RZ={RZ}")
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            print("机械臂已到达目标位置")
        except Exception as e:
            print("Error while sending pose to robot:", e)
    
    def get_tcp_pose(self):
        """获取当前末端执行器位姿（已应用TCP偏移修正）
        
        Returns:
            List[float]: 当前TCP位姿 [x, y, z, rx, ry, rz]，已修正为夹爪顶端位置
        """
        pose, _ = self.get_robot_pose_and_joints()
        return pose
    
    def _get_raw_gripper_base_pose(self):
        """获取原始夹爪底座位姿数据
        
        Returns:
            List[float]: 夹爪底座位姿 [x, y, z, rx, ry, rz]，单位已转换
        """
        arm_end_pose_msgs = self.piper.GetArmEndPoseMsgs()
        return [
            arm_end_pose_msgs.end_pose.X_axis / 1000000.0,  # Convert from 0.001mm to m
            arm_end_pose_msgs.end_pose.Y_axis / 1000000.0,
            arm_end_pose_msgs.end_pose.Z_axis / 1000000.0,
            arm_end_pose_msgs.end_pose.RX_axis / 1000.0,  # Convert from 0.001deg to deg
            arm_end_pose_msgs.end_pose.RY_axis / 1000.0,
            arm_end_pose_msgs.end_pose.RZ_axis / 1000.0
        ]
    
    def get_gripper_base_pose(self):
        """获取当前夹爪底座位姿（未应用TCP偏移修正）
        
        Returns:
            List[float]: 当前夹爪底座位姿 [x, y, z, rx, ry, rz]，机械臂法兰盘位置
        """
        try:
            return self._get_raw_gripper_base_pose()
        except Exception as e:
            print(f"获取夹爪底座位姿时发生错误: {e}")
            return None
    
    def get_joint_positions(self):
        """获取当前关节位置
        
        Returns:
            List[float]: 当前关节位置
        """
        _, joint_positions = self.get_robot_pose_and_joints()
        return joint_positions
    
    def control_gripper(self, width=0, effort=100, close=True):
        """控制Piper双指平行夹爪
        
        Args:
            width: 夹爪目标宽度（0为完全关闭，255为完全打开）
            effort: 夹爪施加的力（默认：100）
            close: 是否关闭（True）或打开（False）夹爪
        """
        try:
            if close:
                success = self.close_gripper()
                if success:
                    print("Gripper closed.")
            else:
                success = self.open_gripper()
                if success:
                    print("Gripper opened.")
        except Exception as e:
            print("Error controlling gripper:", e)
    
    def enable_arm(self, timeout: int = 5):
        """使能机械臂并检测使能状态（兼容原piper_controller接口）
        
        Args:
            timeout: 超时时间（秒）
        """
        self.enable_robot(timeout)
    
    def enable_robot(self, timeout: int = 10):
        """使能机器人"""
        print("正在使能机械臂...")
        enable_flag = False
        start_time = time.time()
        
        while not enable_flag:
            elapsed_time = time.time() - start_time
            arm_info = self.piper.GetArmLowSpdInfoMsgs()
            enable_flag = (
                arm_info.motor_1.foc_status.driver_enable_status and
                arm_info.motor_2.foc_status.driver_enable_status and
                arm_info.motor_3.foc_status.driver_enable_status and
                arm_info.motor_4.foc_status.driver_enable_status and
                arm_info.motor_5.foc_status.driver_enable_status and
                arm_info.motor_6.foc_status.driver_enable_status
            )
            
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            
            if elapsed_time > timeout:
                raise RuntimeError("机械臂使能超时")
            
            if not enable_flag:
                time.sleep(1)
        
        print("机械臂使能成功")
    
    def _update_robot_state(self):
        """更新机器人状态（应用TCP偏移修正）"""
        try:
            # 获取当前末端位姿（夹爪底座位置）
            gripper_base_pose = self._get_raw_gripper_base_pose()
            
            # 应用TCP偏移修正
            tcp_pose = self._apply_tcp_offset(gripper_base_pose)
            
            # 获取当前关节角度
            joint_msgs = self.piper.GetArmJointMsgs()
            joint_positions = [
                joint_msgs.joint_state.joint_1,
                joint_msgs.joint_state.joint_2,
                joint_msgs.joint_state.joint_3,
                joint_msgs.joint_state.joint_4,
                joint_msgs.joint_state.joint_5,
                joint_msgs.joint_state.joint_6,
            ]
            
            # 转换为米单位和四元数表示（使用修正后的TCP位姿）
            position = np.array(tcp_pose[:3])  # 已经是米单位
            rotation = R.from_euler('xyz', np.array(tcp_pose[3:6]), degrees=True)
            quaternion = rotation.as_quat()  # [x, y, z, w]格式
            
            # 更新末端位姿
            self.ee_pose = np.concatenate([position, quaternion])
            
            # 转换为弧度
            self.joint_positions = np.array(joint_positions) * np.pi / (180 * 1000)  # 转换为弧度
            
        except Exception as e:
            print(f"获取机械臂数据时发生错误: {e}")
    
    def create_robot_state_dict(self, current_stage: int = 1) -> dict:
        """创建机器人状态字典，统一单位和格式
        
        Args:
            current_stage: 当前执行阶段
            
        Returns:
            dict: 包含完整机器人状态信息的字典
        """
        try:
            # 获取当前机器人状态
            end_pose, current_joints = self.get_robot_pose_and_joints()
            if current_joints is None or end_pose is None:
                raise RuntimeError("无法获取机器人状态")
            
            # 根据stage决定读取哪个robot_state.json文件
            robot_state_path = './robot_state.json'
            previous_state = None
            
            # 尝试读取现有的robot_state.json或robot_state_{stage-1}.json
            try:
                if current_stage == 1:
                    # 第1阶段，读取robot_state.json
                    if os.path.exists(robot_state_path):
                        with open(robot_state_path, 'r') as f:
                            previous_state = json.load(f)
                            print(f"\033[92m读取初始状态文件: {robot_state_path}\033[0m")
                else:
                    # 第2+阶段，尝试读取上一阶段的状态文件
                    
                    # 获取rekep_program_dir
                    rekep_program_dir = None
                    if hasattr(self, 'rekep_program_dir') and self.rekep_program_dir:
                        rekep_program_dir = self.rekep_program_dir
                    elif hasattr(self, 'env') and hasattr(self.env, 'rekep_program_dir') and self.env.rekep_program_dir:
                        rekep_program_dir = self.env.rekep_program_dir
                    
                    if rekep_program_dir:
                        previous_stage_path = os.path.join(rekep_program_dir, f'robot_state_{current_stage-1}.json')
                        if os.path.exists(previous_stage_path):
                            with open(previous_stage_path, 'r') as f:
                                previous_state = json.load(f)
                                print(f"\033[92m读取上一阶段状态文件: {previous_stage_path}\033[0m")
                        else:
                            # 如果找不到上一阶段文件，尝试读取robot_state.json
                            if os.path.exists(robot_state_path):
                                with open(robot_state_path, 'r') as f:
                                    previous_state = json.load(f)
                                    print(f"\033[93m未找到上一阶段状态文件，读取初始状态: {robot_state_path}\033[0m")
            except Exception as e:
                print(f"\033[93m读取状态文件失败: {e}，将使用当前状态\033[0m")
            
            # 构建机器人状态信息
            robot_state = {
                "timestamp": time.time(),
                "rekep_state": current_stage,
                "rekep_stage": current_stage,
                "joint_info": {
                    "joint_positions": current_joints,  # 弧度单位
                    "reset_joint_pos": current_joints
                },
                "ee_info": {
                    "position": end_pose[:3],  # 米单位 [x, y, z]
                    "orientation": end_pose[3:6]  # 旋转向量 [rx, ry, rz] (弧度单位)
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
                "misc": {}
            }
            
            # 从previous_state中保留必要信息
            if previous_state:
                # 保留initial_position
                if 'initial_position' in previous_state:
                    robot_state['initial_position'] = previous_state['initial_position']
                else:
                    # 如果没有initial_position，使用当前状态创建
                    robot_state['initial_position'] = {
                        "joint_positions": current_joints,
                        "ee_position": end_pose[:3],
                        "ee_orientation": end_pose[3:6]
                    }
                
                # 保留world2robot_homo矩阵
                if 'misc' in previous_state and 'world2robot_homo' in previous_state['misc']:
                    robot_state['misc']['world2robot_homo'] = previous_state['misc']['world2robot_homo']
                
                # 保留raw_data
                if 'raw_data' in previous_state:
                    robot_state['raw_data'] = previous_state['raw_data']
            
            return robot_state
            
        except Exception as e:
            print(f"创建机器人状态字典时发生错误: {e}")
            return None
    
    def save_robot_state_to_file(self, file_path: str, current_stage: int = 1) -> bool:
        """保存机器人状态到指定文件
        
        Args:
            file_path: 保存文件路径
            current_stage: 当前执行阶段
            
        Returns:
            bool: 是否保存成功
        """
        try:
            robot_state = self.create_robot_state_dict(current_stage)
            if robot_state is None:
                return False
            
            import json
            with open(file_path, 'w') as f:
                json.dump(robot_state, f, indent=4)
            
            print(f"\033[92m机器人状态已保存到 {file_path}\033[0m")
            return True
            
        except Exception as e:
            print(f"保存机器人状态到文件时发生错误: {e}")
            return False
            
    def move_to_joint_positions(self, joint_angles_rad: np.ndarray, gripper_val: int = 0, tolerance: float = 0.1) -> bool:
        """移动到目标关节角度
        
        Args:
            joint_angles_rad: 目标关节角度（弧度）
            gripper_val: 夹爪控制值（0表示打开，1表示关闭）
            tolerance: 关节角度误差容忍度（弧度）
            
        Returns:
            bool: 是否成功移动到目标关节角度
        """
        try:
            self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)  # 切换到关节控制模式
            print(f"移动到关节角度: {joint_angles_rad}")
            
            # 转换为0.001度
            joints_mdeg = [round(angle * self.joint_factor) for angle in joint_angles_rad]
            print(f"发送给控制器的关节角度: {joints_mdeg}")
            
            # 等待机械臂到达目标关节角度
            self._wait_for_joints(joint_angles_rad, joints_mdeg, gripper_val, tolerance)
            
            # 更新机器人状态
            self._update_robot_state()
            
            return True
            
        except Exception as e:
            print(f"执行关节动作时出错: {e}")
            return False
            
    def _wait_for_joints(self, target_joints_rad: np.ndarray, joints_mdeg: List[int], gripper_val: int, tolerance: float = 0.1):
        """等待机械臂到达指定关节角度，期间会持续发送目标指令
        
        Args:
            target_joints_rad: 目标关节角度（弧度）
            joints_mdeg: 目标关节角度（0.001度）
            gripper_val: 夹爪控制值（0表示打开，1表示关闭）
            tolerance: 关节角度误差容忍度（弧度）
        """
        # 转换因子：0.001度 -> 弧度
        factor_to_rad = np.pi / (180 * 1000)
        
        while True:
            
            # 发送关节控制指令
            self.piper.JointCtrl(
                joints_mdeg[0], joints_mdeg[1], joints_mdeg[2], 
                joints_mdeg[3], joints_mdeg[4], joints_mdeg[5]
            )
            
            # 获取当前关节角度
            joint_msgs = self.piper.GetArmJointMsgs()
            current_joints_rad = np.array([
                joint_msgs.joint_state.joint_1 * factor_to_rad,
                joint_msgs.joint_state.joint_2 * factor_to_rad,
                joint_msgs.joint_state.joint_3 * factor_to_rad,
                joint_msgs.joint_state.joint_4 * factor_to_rad,
                joint_msgs.joint_state.joint_5 * factor_to_rad,
                joint_msgs.joint_state.joint_6 * factor_to_rad,
            ])
            
            # 计算关节角度误差
            joint_error = np.linalg.norm(current_joints_rad - target_joints_rad)
            print(f"当前关节角度 (rad): {current_joints_rad}")
            print(f"目标关节角度 (rad): {target_joints_rad}")
            print(f"关节角度误差: {joint_error:.4f}")
            
            # 检查是否达到目标
            if joint_error < tolerance:
                print("已到达目标关节角度。")
                # 控制夹爪
                time.sleep(2)
                if gripper_val == 0:
                    self.piper.GripperCtrl(100*1000, 1000, 0x01, 0)  # 张开 100mm
                else:
                    self.piper.GripperCtrl(0*1000, 1000, 0x01, 0)  # 闭合
                break
                
            time.sleep(0.1)
    
    def move_to_pose(self, target_pose: np.ndarray, speed: float = 0.1, acceleration: float = 0.1, precise: bool = False, wait_for_arrival: bool = True, tolerance: float = 0.01) -> bool:
        """移动到目标位姿
        
        Args:
            target_pose: 目标TCP位姿 [x, y, z, qx, qy, qz, qw]（夹爪顶端位置）
            speed: 移动速度 (m/s)
            acceleration: 加速度 (m/s^2)
            precise: 是否精确移动
            wait_for_arrival: 是否等待到达目标位置
            tolerance: 位置误差容忍度 (m)
            
        Returns:
            bool: 是否成功移动到目标位姿
        """
        try:
            # 设置为末端控制模式
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            # 使用减小的速度和加速度进行精确移动
            if precise:
                speed = speed / 2
                acceleration = acceleration / 2
                print("执行精确移动")
            
            # 将四元数转换为欧拉角
            position = target_pose[:3]
            quaternion = target_pose[3:7]  # [qx, qy, qz, qw]
            rotation = R.from_quat(quaternion)
            euler_angles = rotation.as_euler('xyz', degrees=True)
            
            # 构建TCP位姿（欧拉角格式）
            tcp_pose = [position[0], position[1], position[2], 
                       euler_angles[0], euler_angles[1], euler_angles[2]]
            
            # 将TCP位姿转换为夹爪底座位姿
            gripper_base_pose = self._remove_tcp_offset(tcp_pose)
            
            # 转换为Piper控制器可接受的格式（0.001mm和0.001度）
            pos_factor = 1000000  # m -> 0.001mm conversion factor
            rot_factor = 1000     # deg -> 0.001deg conversion factor
            X = round(gripper_base_pose[0] * pos_factor)
            Y = round(gripper_base_pose[1] * pos_factor)
            Z = round(gripper_base_pose[2] * pos_factor)
            RX = round(gripper_base_pose[3] * rot_factor)
            RY = round(gripper_base_pose[4] * rot_factor)
            RZ = round(gripper_base_pose[5] * rot_factor)
            
            print(f"TCP位姿: {tcp_pose}")
            print(f"转换为夹爪底座位姿: {gripper_base_pose}")
            print(f"移动到位姿: X={X}, Y={Y}, Z={Z}, RX={RX}, RY={RY}, RZ={RZ}")
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            
            # 等待移动完成
            if wait_for_arrival:
                self._wait_for_pose(target_pose, tolerance)
            else:
                time.sleep(0.5)  # 简单等待一段时间
            
            # 更新机器人状态
            self._update_robot_state()
            
            return True
            
        except Exception as e:
            print(f"执行动作时出错: {e}")
            return False
            
    def _wait_for_pose(self, target_pose: np.ndarray, tolerance: float = 0.01, max_attempts: int = 50):
        """等待机械臂到达目标位姿
        
        Args:
            target_pose: 目标位姿 [x, y, z, qx, qy, qz, qw]
            tolerance: 位置误差容忍度 (m)
            max_attempts: 最大尝试次数
        """
        target_pos = target_pose[:3]
        target_quat = target_pose[3:7]
        
        for attempt in range(max_attempts):
            # 更新机器人状态
            self._update_robot_state()
            
            # 获取当前位姿
            current_pos = self.ee_pose[:3]
            current_quat = self.ee_pose[3:7]
            
            # 计算位置误差
            pos_error = np.linalg.norm(current_pos - target_pos)
            
            # 计算旋转误差
            rot_error = 2 * np.arccos(np.abs(np.sum(current_quat * target_quat)))
            
            print(f"等待到达目标位置: 尝试 {attempt+1}/{max_attempts}, 位置误差: {pos_error:.4f}m, 旋转误差: {rot_error:.4f}rad")
            
            # 检查是否达到目标
            if pos_error < tolerance:
                print("已到达目标位置")
                return
            
            # 等待一段时间
            time.sleep(0.1)
        
        print(f"警告: 在 {max_attempts} 次尝试后仍未达到目标位置，继续执行")

    
    def open_gripper(self) -> bool:
        """打开夹爪"""
        try:
            print("执行打开夹爪动作")
            # 打开夹爪，使用同步调用
            self.piper.GripperCtrl(50*1000, 1000, 0x01, 0)  # 张开 50mm
            time.sleep(1.5)  # 等待夹爪打开
            print("夹爪已打开")
            return True
        except Exception as e:
            print(f"打开夹爪时出错: {e}")
            return False
    
    def close_gripper(self) -> bool:
        """关闭夹爪"""
        try:
            print("执行关闭夹爪动作")
            # 关闭夹爪，使用同步调用
            self.piper.GripperCtrl(0*1000, 1000, 0x01, 0)  # 闭合
            time.sleep(1.5)  # 等待夹爪关闭
            print("夹爪已关闭")
            return True
        except Exception as e:
            print(f"关闭夹爪时出错: {e}")
            return False
    
    def is_grasping(self, object_name: str) -> bool:
        """检查是否抓取到物体
        注意：这是一个简化实现，实际应根据夹爪传感器反馈判断
        """
        # 在实际实现中，应该检查夹爪电流或力传感器数据
        # 如果object_name为None，表示没有关联物体，返回False
        if object_name is None:
            return False
        # 这里简单返回False，避免影响正常的抓取逻辑
        # 实际使用时应替换为真实的传感器反馈逻辑
        return True
    
    def get_ee_pose(self) -> np.ndarray:
        """获取末端执行器的位姿（已应用TCP偏移修正）
        
        Returns:
            np.ndarray: 修正后的TCP位姿，已修正为夹爪顶端位置
        """
        return self.get_robot_pose_and_joints()[0]


class ReKepEnv:
    """ReKep环境类，用于机器人操作和关键点跟踪"""
    
    def __init__(self, interface: str = "can0", test_mode: bool = False, verbose: bool = True, camera_instance=None, cotracker_client=None, rekep_program_dir=None, camera_config_path="./configs/camera_config.yaml"):
        """初始化ReKep环境
        
        Args:
            interface: CAN 接口名称 (默认 "can0")
            test_mode: 是否启用测试模式
            verbose: 是否启用详细输出
            camera_instance: 外部传入的相机实例，用于避免重复创建相机pipeline
            cotracker_client: 外部传入的CoTracker客户端实例
            rekep_program_dir: ReKep程序目录路径，用于读取和保存状态文件
            camera_config_path: 相机配置文件路径
        """
        self.verbose = verbose
        self.rekep_program_dir = rekep_program_dir
        self.camera_config_path = camera_config_path
        
        # 初始化机器人控制器
        self.robot = RobotController(interface, test_mode)
        # 将rekep_program_dir传递给robot对象
        self.robot.rekep_program_dir = rekep_program_dir
        
        # 关键点相关变量
        self.keypoints = None
        self.keypoints_2d = None
        self.cotracker_client = None
        self._keypoint_object_cache = {}  # 缓存关键点对应的物体，避免重复查询
        
        # 相机相关
        self.external_camera = camera_instance  # 外部传入的相机实例
        self.camera_pipeline = None
        self.camera_config = None
        
        # 加载相机配置并设置内参
        self.camera_config_data = self._load_camera_config()
        self.intrinsics = self.get_intrinsics()
        
        # 初始化关键点跟踪器
        self.initialize_keypoint_tracker(cotracker_client)
        
        # 从配置文件加载工作空间边界和插值参数
        self._load_workspace_config()
        
        
    def _load_workspace_config(self, config_path: str = "./configs/config.yaml"):
        """从配置文件加载工作空间边界和插值参数
        
        Args:
            config_path: 配置文件路径
        """
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 从main配置中读取边界
            if 'main' in config:
                main_config = config['main']
                self.bounds_min = np.array(main_config.get('bounds_min', [0.1, -0.5, 0.02]))
                self.bounds_max = np.array(main_config.get('bounds_max', [0.8, 0.5, 0.8]))
                self.interpolate_pos_step_size = main_config.get('interpolate_pos_step_size', 0.02)
                self.interpolate_rot_step_size = main_config.get('interpolate_rot_step_size', 0.1)
            else:
                # 如果没有main配置，使用默认值
                self.bounds_min = np.array([0.1, -0.5, 0.02])
                self.bounds_max = np.array([0.8, 0.5, 0.8])
                self.interpolate_pos_step_size = 0.02
                self.interpolate_rot_step_size = 0.1
                print("\033[93m警告: 配置文件中未找到main配置，使用默认工作空间边界\033[0m")
            
            if self.verbose:
                print(f"从配置文件加载工作空间边界:")
                print(f"  bounds_min: {self.bounds_min}")
                print(f"  bounds_max: {self.bounds_max}")
                print(f"  插值步长: pos={self.interpolate_pos_step_size}, rot={self.interpolate_rot_step_size}")
                
        except Exception as e:
            print(f"\033[91m加载工作空间配置失败: {e}\033[0m")
            print("\033[93m使用默认工作空间边界\033[0m")
            # 使用默认值
            self.bounds_min = np.array([0.1, -0.5, 0.02])
            self.bounds_max = np.array([0.8, 0.5, 0.8])
            self.interpolate_pos_step_size = 0.02
            self.interpolate_rot_step_size = 0.1
    
    def initialize_keypoint_tracker(self, cotracker_client=None):
        """初始化关键点跟踪器
        
        Args:
            cotracker_client: 外部传入的CoTracker客户端实例
        """
        # 优先使用外部传入的CoTracker客户端
        if cotracker_client is not None:
            self.cotracker_client = cotracker_client
            print("使用外部传入的关键点跟踪器")
            return
            
        try:
            # 尝试导入并初始化CoTracker客户端
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from keypoint_cotracker import CoTrackerClient
            
            self.cotracker_client = CoTrackerClient()
            print("关键点跟踪器初始化成功")
        except Exception as e:
            print(f"关键点跟踪器初始化失败: {e}")
            self.cotracker_client = None
    
    def get_camera_view(self, camera_name: str = "realsense") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取相机视图
        
        Args:
            camera_name: 相机名称
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: RGB图像和深度图像
        """
        try:
            # 优先使用外部传入的相机实例
            if self.external_camera is not None:
                if hasattr(self.external_camera, 'get_frames'):
                    return self.external_camera.get_frames()
                elif hasattr(self.external_camera, 'get_camera_view'):
                    return self.external_camera.get_camera_view()
            
            # 如果没有外部相机，则使用自己的pipeline
            if self.camera_pipeline is None:
                # 加载相机配置
                config = self._load_camera_config()
                resolution = config['resolution']
                width = resolution['width']
                height = resolution['height']
                fps = resolution['fps']
                
                # 初始化相机
                self.camera_pipeline = rs.pipeline()
                self.camera_config = rs.config()
                
                # 检查是否有指定的序列号
                if 'realsense' in config and config['realsense']['serial_number']:
                    self.camera_config.enable_device(config['realsense']['serial_number'])
                
                self.camera_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                self.camera_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                self.camera_pipeline.start(self.camera_config)
                
                # 等待相机稳定
                for _ in range(30):
                    self.camera_pipeline.wait_for_frames()
            
            frames = self.camera_pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # BGR转RGB
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"获取相机视图时出错: {e}")
            return None, None
    
    def _load_camera_config(self) -> Dict[str, Any]:
        """加载相机配置文件
        
        Returns:
            Dict[str, Any]: 相机配置
        """
        try:
            with open(self.camera_config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载相机配置文件失败: {e}")
            # 返回默认配置，使用新的结构（depth和color分开）
            return {
                'resolution': {'width': 640, 'height': 480, 'fps': 30},
                'intrinsics': {
                    'depth': {
                        'fx': 382.06, 'fy': 382.06,
                        'ppx': 321.79, 'ppy': 235.10
                    },
                    'color': {
                        'fx': 606.60, 'fy': 605.47,
                        'ppx': 323.69, 'ppy': 247.12
                    },
                    'depth_scale': 0.001
                },
                'processing': {'resize_width': 640, 'resize_height': 480}
            }
    
    def get_intrinsics(self) -> Optional[Dict[str, Any]]:
        """获取相机内参
        
        Returns:
            Optional[Dict[str, Any]]: 相机内参字典
        """
        try:
            # 使用已加载的配置数据
            if hasattr(self, 'camera_config_data') and self.camera_config_data:
                config = self.camera_config_data
            else:
                # 如果尚未加载，则加载配置
                config = self._load_camera_config()
                
            intrinsics = config['intrinsics']
            resolution = config['resolution']
            
            # 检查intrinsics结构
            if 'depth' in intrinsics and 'color' in intrinsics:
                # 新结构：intrinsics下有depth和color子字典
                return {
                    'depth': {
                        'fx': intrinsics['depth']['fx'],
                        'fy': intrinsics['depth']['fy'],
                        'ppx': intrinsics['depth']['ppx'],
                        'ppy': intrinsics['depth']['ppy'],
                        'width': resolution['width'],
                        'height': resolution['height']
                    },
                    'color': {
                        'fx': intrinsics['color']['fx'],
                        'fy': intrinsics['color']['fy'],
                        'ppx': intrinsics['color']['ppx'],
                        'ppy': intrinsics['color']['ppy'],
                        'width': resolution['width'],
                        'height': resolution['height']
                    }
                }
            else:
                # 旧结构：intrinsics直接包含fx、fy等键
                return {
                    'depth': {
                        'fx': intrinsics['fx'],
                        'fy': intrinsics['fy'],
                        'ppx': intrinsics['ppx'],
                        'ppy': intrinsics['ppy'],
                        'width': resolution['width'],
                        'height': resolution['height']
                    },
                    'color': {
                        'fx': intrinsics['fx'],
                        'fy': intrinsics['fy'],
                        'ppx': intrinsics['ppx'],
                        'ppy': intrinsics['ppy'],
                        'width': resolution['width'],
                        'height': resolution['height']
                    }
                }
            
        except Exception as e:
            print(f"获取相机内参时出错: {e}")
            return None
    
    def get_robot_pose_and_joints(self):
        """获取机器人位姿和关节角度"""
        return self.robot.get_robot_pose_and_joints()
    
    def create_robot_state_dict(self, current_stage: int = 1) -> dict:
        """创建机器人状态字典"""
        return self.robot.create_robot_state_dict(current_stage)
    
    def save_robot_state_to_file(self, file_path: str, current_stage: int = 1) -> bool:
        """保存机器人状态到文件"""
        return self.robot.save_robot_state_to_file(file_path, current_stage)
    
    def action(self, end_pose: List[float], grip_angle: int, grip_force: int = 1000):
        """执行机器人动作"""
        # 验证并修正位置，确保在安全边界内
        validated_position = self.validate_position(np.array(end_pose[:3]))
        validated_end_pose = validated_position.tolist() + end_pose[3:]
        
        return self.robot.action(validated_end_pose, grip_angle, grip_force)
    
    def send_pose_to_robot(self, end_pose):
        """发送位姿到机器人"""
        return self.robot.send_pose_to_robot(end_pose)
    
    def get_tcp_pose(self):
        """获取TCP位姿"""
        return self.robot.get_tcp_pose()
    
    def control_gripper(self, width=0, effort=100, close=True):
        """控制夹爪"""
        return self.robot.control_gripper(width, effort, close)
    
    def get_ee_pose(self) -> np.ndarray:
        """获取末端执行器位姿"""
        return self.robot.get_ee_pose()
    
    def execute_joint_action(self, joint_action: np.ndarray, tolerance: float = 0.1) -> bool:
        """执行关节动作"""
        joint_action = np.array(joint_action).copy()
        assert joint_action.shape == (7,), f"关节动作维度错误: {joint_action.shape} != (7,)"
        
        joint_angles = joint_action[:6]
        gripper_action = joint_action[6]
        
        # 确定夹爪控制值
        gripper_val = 0  # 默认打开
        if gripper_action > 0.5:  # 关闭夹爪
            gripper_val = 1
        
        # 执行关节动作
        success = self.robot.move_to_joint_positions(joint_angles, gripper_val, tolerance)
        
        # 如果需要单独控制夹爪
        if success and gripper_action > 0.5:  # 关闭夹爪
            self.close_gripper()
        elif success and gripper_action < -0.5:  # 打开夹爪
            self.open_gripper()
        
        return success
    
    def execute_action(self, action: np.ndarray, precise: bool = False) -> Tuple[float, float]:
        """执行动作"""
        action = np.array(action).copy()
        assert action.shape == (8,), f"动作维度错误: {action.shape} != (8,)"
        
        target_pose = action[:7]
        gripper_action = action[7]
        
        # 安全检查：确保目标位置在工作空间内
        target_pose[:3] = self.validate_position(target_pose[:3])
        
        # 插值移动
        current_pose = self.get_ee_pose()
        pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
        rot_diff = self._angle_between_quats(current_pose[3:7], target_pose[3:7])
        
        pos_is_close = pos_diff < self.interpolate_pos_step_size
        rot_is_close = rot_diff < self.interpolate_rot_step_size
        
        if pos_is_close and rot_is_close:
            self.verbose and print("跳过插值")
            waypoints = [target_pose]
        else:
            self.verbose and print(f"生成插值路径: 位置差异={pos_diff:.4f}m, 旋转差异={rot_diff:.4f}rad")
            waypoints = self._interpolate_path(current_pose, target_pose)
        
        # 执行路径
        for i, waypoint in enumerate(waypoints):
            is_last = i == len(waypoints) - 1
            self.verbose and print(f"执行路径点 {i+1}/{len(waypoints)}")
            
            # 移动到路径点，只在最后一个路径点等待到达
            wait_for_arrival = is_last  # 只在最后一个路径点等待到达
            success = self.robot.move_to_pose(waypoint, precise=precise and is_last, wait_for_arrival=wait_for_arrival)
            if not success:
                print(f"移动到路径点 {i+1} 失败")
                break
        
        # 执行夹爪动作
        if gripper_action > 0.5:  # 关闭夹爪
            self.close_gripper()
        elif gripper_action < -0.5:  # 打开夹爪
            self.open_gripper()
        
        # 计算最终误差
        final_pos_diff, final_rot_diff = self.compute_target_delta_ee(target_pose)
        self.verbose and print(f"最终误差: 位置={final_pos_diff:.4f}m, 旋转={final_rot_diff:.4f}rad")
        
        return final_pos_diff, final_rot_diff
    
    def _interpolate_path(self, start_pose: np.ndarray, end_pose: np.ndarray) -> List[np.ndarray]:
        """在两个位姿之间插值生成路径"""
        # 计算位置差异
        pos_diff = np.linalg.norm(end_pose[:3] - start_pose[:3])
        num_steps = max(2, int(np.ceil(pos_diff / self.interpolate_pos_step_size)))
        
        waypoints = []
        for i in range(num_steps + 1):
            t = i / num_steps
            
            # 线性插值位置
            pos = start_pose[:3] * (1 - t) + end_pose[:3] * t
            
            # 球面线性插值旋转
            start_quat = start_pose[3:7]
            end_quat = end_pose[3:7]
            
            # 确保四元数是单位四元数
            start_quat = start_quat / np.linalg.norm(start_quat)
            end_quat = end_quat / np.linalg.norm(end_quat)
            
            # 计算四元数点积
            dot = np.sum(start_quat * end_quat)
            
            # 如果点积为负，反转一个四元数以获得较短的路径
            if dot < 0.0:
                end_quat = -end_quat
                dot = -dot
            
            # 如果四元数几乎相同，直接线性插值
            if dot > 0.9995:
                quat = start_quat * (1 - t) + end_quat * t
                quat = quat / np.linalg.norm(quat)
            else:
                # 球面线性插值
                theta_0 = np.arccos(dot)
                sin_theta_0 = np.sin(theta_0)
                
                theta = theta_0 * t
                sin_theta = np.sin(theta)
                
                s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
                s1 = sin_theta / sin_theta_0
                
                quat = start_quat * s0 + end_quat * s1
            
            # 创建路径点
            waypoint = np.concatenate([pos, quat])
            waypoints.append(waypoint)
        
        return waypoints
    
    def close_gripper(self) -> bool:
        """关闭夹爪"""
        return self.robot.close_gripper()
    
    def open_gripper(self) -> bool:
        """打开夹爪"""
        return self.robot.open_gripper()
    
    def is_grasping(self,object_name) -> bool:
        """检查是否抓取到物体"""
        return self.robot.is_grasping(object_name)
    
    def get_gripper_open_action(self) -> float:
        """获取打开夹爪的动作值"""
        return -1.0
    
    def get_gripper_close_action(self) -> float:
        """获取关闭夹爪的动作值"""
        return 1.0
    
    def get_gripper_null_action(self) -> float:
        """获取不改变夹爪状态的动作值"""
        return 0.0
    
    def get_keypoint_positions(self):
        """获取关键点位置"""
        return self.keypoints
    
    def get_joint_pos(self) -> np.ndarray:
        """获取当前关节位置"""
        return np.array(self.robot.get_joint_positions())
    
    def get_ee_location(self) -> np.ndarray:
        """获取当前末端执行器位置"""
        tcp_pose = self.robot.get_tcp_pose()
        return np.array(tcp_pose[:3])
    
    def validate_position(self, position: np.ndarray) -> np.ndarray:
        """验证并修正位置，确保在安全边界内
        
        Args:
            position: 要验证的位置 [x, y, z]
            
        Returns:
            np.ndarray: 修正后的安全位置
        """
        position = np.array(position)
        original_position = position.copy()
        
        # 检查并修正每个轴的位置
        for i in range(3):
            if position[i] < self.bounds_min[i]:
                axis_name = ['X', 'Y', 'Z'][i]
                print(f"\033[91m警告: {axis_name}坐标 {position[i]:.4f}m 小于安全限制 {self.bounds_min[i]:.4f}m，调整为 {self.bounds_min[i]:.4f}m\033[0m")
                position[i] = self.bounds_min[i]
            elif position[i] > self.bounds_max[i]:
                axis_name = ['X', 'Y', 'Z'][i]
                print(f"\033[91m警告: {axis_name}坐标 {position[i]:.4f}m 大于安全限制 {self.bounds_max[i]:.4f}m，调整为 {self.bounds_max[i]:.4f}m\033[0m")
                position[i] = self.bounds_max[i]
        
        # 如果位置被修正，输出信息
        if not np.allclose(original_position, position):
            print(f"位置已从 {original_position} 修正为 {position}")
        
        return position
    
    def get_ee_pose(self) -> np.ndarray:
        """获取当前末端执行器位姿"""
        return self.robot.get_ee_pose()
    
    def control_gripper(self, close: bool = True):
        """控制夹爪
        
        Args:
            close: 是否关闭夹爪
        """
        self.robot.control_gripper(close=close)
    
    def register_keypoints(self, keypoints: List[List[float]], keypoints_2d: List[List[float]] = None):
        """注册关键点"""
        # self.update_keypoints(keypoints)
        # 这里的keypoint是Camera坐标系下的。
        self.keypoints = np.array(keypoints)
        
        # 清除关键点对象缓存，因为关键点已经改变
        self._keypoint_object_cache = {}
        
        # 如果提供了2D关键点并且CoTrackerClient可用，则进行关键点注册
        if keypoints_2d is not None and self.cotracker_client is not None:
            self.keypoints_2d = keypoints_2d
            # Keypoint_2d [y,x] 转换成 [x,y]
            self.keypoints_2d = [[kp[1], kp[0]] for kp in self.keypoints_2d]
            self.register_keypoints_for_tracking()
            print(f"已注册 {len(keypoints)} 个关键点")
            return True
        else:
            print("未提供2D关键点或CoTracker客户端未初始化，无法注册关键点")
            return False
    
    def update_keypoints(self, keypoints: List[List[float]]):
        """更新关键点（相机坐标系）"""
        # self.keypoints = np.array(keypoints)
        # 清除关键点对象缓存，因为关键点已经改变
        self._keypoint_object_cache = {}
        # TODO
        print(f"MOCK====已更新 {len(keypoints)} 个关键点====MOCK ")
    
    def register_keypoints_for_tracking(self):
        """为跟踪注册关键点"""
        if self.cotracker_client is None:
            print("CoTracker客户端未初始化，无法注册关键点")
            return
        
        try:
            # 获取当前相机图像
            rgb_image, _ = self.get_camera_view()
            if rgb_image is None:
                print("无法获取相机图像，关键点注册失败")
                return
            
            # 清除关键点对象缓存，因为关键点可能已经改变
            self._keypoint_object_cache = {}
            
            # 注册关键点
            keypoints_array = np.array(self.keypoints_2d, dtype=np.float32)
            status_code, response = self.cotracker_client.register_frame(rgb_image, keypoints_array)
            if status_code == 200:
                print(f"成功注册 {len(self.keypoints_2d)} 个关键点进行跟踪")
                return True
            else:
                print(f"关键点注册失败: {status_code}, {response}")
                return False
                
        except Exception as e:
            print(f"注册关键点时出错: {e}")
    
    def track_keypoints(self) -> Tuple[bool, List[List[float]]]:
        """跟踪关键点并更新3D位置
        
        Returns:
            Tuple[bool, List[List[float]]]: (成功标志, 跟踪到的3D关键点列表)
        """
        if self.cotracker_client is None:
            print("CoTracker客户端未初始化，无法跟踪关键点")
            return False, []
        
        try:
            # 获取当前相机图像
            rgb_image, depth_image = self.get_camera_view()
            if rgb_image is None or depth_image is None:
                print("无法获取相机图像，关键点跟踪失败")
                return False, []
            
            # 跟踪关键点
            status_code, response = self.cotracker_client.track_frame(rgb_image)
            if status_code == 200:
                tracked_keypoints_2d = response.get('keypoints', [])
                if len(tracked_keypoints_2d) > 0:
                    print(f"成功跟踪到 {len(tracked_keypoints_2d)} 个关键点(2D)")
                    
                    # 更新2D关键点属性，用于可视化
                    self.keypoints_2d = tracked_keypoints_2d
                    
                    # 获取相机内参
                    intrinsics_dict = self.get_intrinsics()
                    if not intrinsics_dict or 'depth' not in intrinsics_dict:
                        print("无法获取相机内参，保持原有关键点位置")
                        return False, []
                    
                    # 使用CoTrackerClient将2D关键点转换为3D关键点，相机坐标系
                    keypoints_3d = self.cotracker_client.convert_2d_to_3d(
                        tracked_keypoints_2d, depth_image, intrinsics_dict['depth']
                    )
                    
                    # 更新关键点
                    self.update_keypoints(keypoints_3d)

                    # 返回更新后的3D关键点
                    return True, self.keypoints.tolist() if self.keypoints is not None else []
                else:
                    print("未检测到关键点")
                    return False, []
            else:
                print(f"关键点跟踪失败: {status_code}, {response}")
                return False, []
                
        except Exception as e:
            print(f"跟踪关键点时出错: {e}")
            return False, []
    
    def compute_target_delta_ee(self, target_pose: np.ndarray) -> Tuple[float, float]:
        """计算当前末端位姿与目标位姿之间的差异"""
        target_pos, target_quat = target_pose[:3], target_pose[3:7]
        ee_pose = self.get_ee_pose()
        ee_pos, ee_quat = ee_pose[:3], ee_pose[3:7]
        
        # 计算位置误差
        pos_diff = np.linalg.norm(ee_pos - target_pos)
        
        # 计算旋转误差
        rot_diff = self._angle_between_quats(ee_quat, target_quat)
        
        return pos_diff, rot_diff
    
    def _angle_between_quats(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """计算两个四元数之间的角度差异"""
        # 确保四元数是单位四元数
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # 计算四元数点积
        dot_product = np.abs(np.sum(q1 * q2))
        dot_product = min(dot_product, 1.0)
        
        # 计算角度
        angle = 2 * np.arccos(dot_product)
        
        return angle
    
    def reset(self):
        """重置环境"""
        print("重置机器人环境")
        # 可以在这里添加重置逻辑，比如回到初始位置
        time.sleep(0.5)
        
    def close(self):
        """关闭环境并释放资源"""
        print("关闭ReKep环境并释放资源")
        
        # 只有在自己创建了相机pipeline时才关闭它
        if self.camera_pipeline is not None and self.external_camera is None:
            try:
                self.camera_pipeline.stop()
                print("相机pipeline已停止")
            except Exception as e:
                print(f"停止相机pipeline时出错: {e}")
            finally:
                self.camera_pipeline = None
        
        # 不关闭外部传入的相机和CoTracker客户端，由创建它们的地方负责关闭
    
    def get_sdf_voxels(self, resolution: float = 0.1, exclude_robot: bool = True, exclude_obj_in_hand: bool = True) -> np.ndarray:
        """获取SDF体素网格 (mock实现)
        
        Args:
            resolution: 体素分辨率
            exclude_robot: 是否排除机器人
            exclude_obj_in_hand: 是否排除手中的物体
            
        Returns:
            np.ndarray: SDF体素网格 (mock数据)
        """
        if self.verbose:
            print(f"获取SDF体素 (mock数据) - 分辨率: {resolution}")
        # 返回合理大小的mock SDF网格数据，避免内存过大
        grid_size = min(int(1.0 / resolution), 50)  # 限制最大网格大小为50
        return np.ones((grid_size, grid_size, grid_size)) * 0.1  # 返回正值SDF，表示空闲空间
    
    def get_collision_points(self) -> np.ndarray:
        """获取碰撞点 (mock实现)
        
        Returns:
            np.ndarray: 碰撞点数组 (mock数据)
        """
        if self.verbose:
            print("获取碰撞点 (mock数据)")
        # 返回 None 表示没有碰撞点，这样在后续处理中会被正确处理
        # 如果需要模拟碰撞，可以根据实际场景添加合理的碰撞点
        return None
    
    def get_object_by_keypoint(self, keypoint_index):
        """Get object identifier by keypoint index.
        
        Args:
            keypoint_index (int): Index of the keypoint
            
        Returns:
            str: Object identifier for the keypoint
        """
        # 如果没有_keypoint_object_cache属性，则初始化它
        if not hasattr(self, '_keypoint_object_cache'):
            self._keypoint_object_cache = {}
            
        # 检查缓存中是否已有该关键点的对象映射
        if keypoint_index in self._keypoint_object_cache:
            return self._keypoint_object_cache[keypoint_index]
            
        # Mock object mapping - 返回None表示没有关联的物体
        # 这样不会影响is_grasping等方法的判断逻辑
        if self.verbose:
            print(f"获取关键点 {keypoint_index} 对应的物体 (mock数据)")
            
        # 将结果存入缓存
        self._keypoint_object_cache[keypoint_index] = None
        return None
    
    def draw_keypoints(self, img, keypoints, color=(0, 255, 0), radius=5):
        """在图像上绘制关键点
        
        Args:
            img (numpy.ndarray): 输入图像
            keypoints (numpy.ndarray): 关键点坐标
            color (tuple): 关键点颜色 (B, G, R)
            radius (int): 关键点半径
            
        Returns:
            numpy.ndarray: 绘制了关键点的图像
        """
        img_with_keypoints = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for point in keypoints:
            x, y = int(point[0]), int(point[1])
            cv2.circle(img_with_keypoints, (x, y), radius, color, -1)
        return img_with_keypoints
    
    def save_frame_with_keypoints(self, img, keypoints, output_path, depth_image=None, save_dir=None):
        """保存带有关键点的图像
        
        Args:
            img (numpy.ndarray): 输入图像
            keypoints (numpy.ndarray): 关键点坐标
            output_path (str): 输出文件路径
            depth_image (numpy.ndarray): 深度图像（可选）
            save_dir (str): 保存目录（可选），如果提供，将覆盖output_path中的目录部分
        """
        # 如果提供了保存目录，则使用该目录和output_path的文件名部分
        if save_dir is not None:
            # 确保目录存在
            os.makedirs(save_dir, exist_ok=True)
            # 获取output_path的文件名部分
            filename = os.path.basename(output_path)
            # 组合新的输出路径
            output_path = os.path.join(save_dir, filename)
        
        img_with_keypoints = self.draw_keypoints(img, keypoints)
        cv2.imwrite(output_path, img_with_keypoints)
        
        # 如果提供了深度图像，也保存深度数据
        if depth_image is not None:
            depth_path = output_path.replace('.png', '_depth.npy').replace('.jpg', '_depth.npy')
            np.save(depth_path, depth_image)
        
        print(f'Frame saved to {output_path}')


# PiperController 别名，直接使用 RobotController
PiperController = RobotController