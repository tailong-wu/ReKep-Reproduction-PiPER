#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机标定结果转换为外参矩阵工具

将标定结果文件中的位置和四元数转换为外参矩阵，并更新到相机配置文件中。
支持生成world2robot_homo变换矩阵。
"""

import json
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import os
from typing import Dict, Any, Tuple


def load_calibration_result(calibration_file: str) -> Dict[str, Any]:
    """加载标定结果文件
    
    Args:
        calibration_file: 标定结果JSON文件路径
        
    Returns:
        标定结果字典
    """
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)
    return calibration_data


def validate_calibration_data(calibration_data: Dict[str, Any]) -> None:
    """验证标定数据的有效性
    
    Args:
        calibration_data: 标定结果数据
        
    Raises:
        ValueError: 当数据格式不正确时
    """
    required_fields = ['position', 'orientation']
    for field in required_fields:
        if field not in calibration_data:
            raise ValueError(f"标定数据缺少必需字段: {field}")
    
    position = calibration_data['position']
    orientation = calibration_data['orientation']
    
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError(f"位置向量格式错误，期望长度为3的列表，实际: {position}")
    
    if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
        raise ValueError(f"四元数格式错误，期望长度为4的列表，实际: {orientation}")
    
    # 验证数值类型
    try:
        position = [float(x) for x in position]
        orientation = [float(x) for x in orientation]
    except (ValueError, TypeError) as e:
        raise ValueError(f"位置或四元数包含非数值类型: {e}")
    
    # 验证四元数归一化
    quat_norm = np.linalg.norm(orientation)
    if abs(quat_norm - 1.0) > 1e-3:
        print(f"警告: 四元数未归一化，模长为{quat_norm:.6f}，将自动归一化")
        # 自动归一化
        orientation_normalized = np.array(orientation) / quat_norm
        calibration_data['orientation'] = orientation_normalized.tolist()
    
    # 更新数据为浮点数
    calibration_data['position'] = position
    calibration_data['orientation'] = orientation if abs(quat_norm - 1.0) <= 1e-3 else orientation_normalized.tolist()


def quaternion_to_rotation_matrix(quaternion: list) -> np.ndarray:
    """将四元数转换为旋转矩阵
    
    Args:
        quaternion: 四元数 [qx, qy, qz, qw]
        
    Returns:
        3x3旋转矩阵
    """
    # 输入格式为[qx, qy, qz, qw]，scipy需要[x, y, z, w]格式
    qx, qy, qz, qw = quaternion
    rotation = R.from_quat([qx, qy, qz, qw])
    return rotation.as_matrix()


def create_extrinsic_matrix(position: list, quaternion: list) -> np.ndarray:
    """创建4x4外参矩阵 (Camera to Base)
    
    Args:
        position: 位置向量 [x, y, z]
        quaternion: 四元数 [qx, qy, qz, qw]
        
    Returns:
        4x4外参变换矩阵 (Camera to Base)
    """
    # 获取旋转矩阵
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    
    # 创建4x4变换矩阵
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = position
    
    return extrinsic_matrix


def create_world2robot_homo(position: list, quaternion: list) -> np.ndarray:
    """创建world2robot_homo变换矩阵
    
    Args:
        position: 相机在机器人基座坐标系中的位置 [x, y, z]
        quaternion: 相机在机器人基座坐标系中的姿态四元数 [qx, qy, qz, qw]
        
    Returns:
        4x4 world2robot_homo变换矩阵 (World to Robot Base)
    """
    # 相机到机器人基座的变换矩阵
    camera_to_base = create_extrinsic_matrix(position, quaternion)
    
    # world2robot_homo = camera_to_base (假设世界坐标系与相机坐标系重合)
    # 在实际应用中，这个假设可能需要根据具体的坐标系定义进行调整
    world2robot_homo = camera_to_base
    
    return world2robot_homo


def quaternion_to_rpy(quaternion: list) -> Tuple[float, float, float]:
    """将四元数转换为RPY角度
    
    Args:
        quaternion: 四元数 [qx, qy, qz, qw]
        
    Returns:
        (roll, pitch, yaw) 弧度值
    """
    qx, qy, qz, qw = quaternion
    rotation = R.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
    return roll, pitch, yaw


def update_camera_config(config_file: str, extrinsic_matrix: np.ndarray, position: list, quaternion: list, rpy: Tuple[float, float, float], world2robot_homo: np.ndarray = None) -> None:
    """更新相机配置文件
    
    Args:
        config_file: 相机配置文件路径
        extrinsic_matrix: 外参矩阵
        position: 位置向量
        quaternion: 四元数
        rpy: RPY角度
        world2robot_homo: world2robot_homo变换矩阵（可选）
    """
    # 尝试加载现有配置
    config_data = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"警告: 无法读取现有配置文件 {config_file}: {e}")
            print("将创建新的配置文件")
    
    # 更新标定相关的配置
    config_data.update({
        'extrinsic_matrix': extrinsic_matrix.tolist(),
        'position': [float(x) for x in position],
        'quaternion': [float(x) for x in quaternion],
        'rpy': {
            'roll': float(rpy[0]),
            'pitch': float(rpy[1]),
            'yaw': float(rpy[2])
        }
    })
    
    # 如果提供了world2robot_homo矩阵，也添加到配置中
    if world2robot_homo is not None:
        config_data['world2robot_homo'] = world2robot_homo.tolist()
    
    # 确保目录存在
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)


def update_robot_state(robot_state_file: str, world2robot_homo: np.ndarray) -> None:
    """更新robot_state.json文件中的world2robot_homo矩阵
    
    Args:
        robot_state_file: robot_state.json文件路径
        world2robot_homo: world2robot_homo变换矩阵
    """
    # 尝试加载现有的robot_state文件
    robot_state_data = {}
    if os.path.exists(robot_state_file):
        try:
            with open(robot_state_file, 'r') as f:
                robot_state_data = json.load(f)
        except Exception as e:
            print(f"警告: 无法读取现有robot_state文件 {robot_state_file}: {e}")
            print("将创建新的robot_state文件")
    
    # 确保misc字段存在
    if 'misc' not in robot_state_data:
        robot_state_data['misc'] = {}
    
    # 更新world2robot_homo矩阵
    robot_state_data['misc']['world2robot_homo'] = world2robot_homo.tolist()
    
    # 确保目录存在
    robot_state_dir = os.path.dirname(robot_state_file)
    if robot_state_dir:  # 只有当目录不为空时才创建
        os.makedirs(robot_state_dir, exist_ok=True)
    
    # 写入文件
    with open(robot_state_file, 'w') as f:
        json.dump(robot_state_data, f, indent=4)
    
    print(f"robot_state.json文件已更新: {robot_state_file}")


def print_extrinsic_matrix(extrinsic_matrix: np.ndarray, position: list, quaternion: list, rpy: Tuple[float, float, float], world2robot_homo: np.ndarray = None) -> None:
    """打印外参矩阵和相关信息
    
    Args:
        extrinsic_matrix: 外参矩阵
        position: 位置向量
        quaternion: 四元数
        rpy: RPY角度
        world2robot_homo: world2robot_homo变换矩阵（可选）
    """
    print("\n=== 相机外参矩阵 (Camera to Base) ===")
    print(extrinsic_matrix)
    
    if world2robot_homo is not None:
        print("\n=== World2Robot_Homo变换矩阵 ===")
        print(world2robot_homo)
    
    print("\n=== 位置信息 ===")
    print(f"X: {position[0]:.6f}")
    print(f"Y: {position[1]:.6f}")
    print(f"Z: {position[2]:.6f}")
    
    print("\n=== 四元数 ===")
    print(f"qx: {quaternion[0]:.6f}")
    print(f"qy: {quaternion[1]:.6f}")
    print(f"qz: {quaternion[2]:.6f}")
    print(f"qw: {quaternion[3]:.6f}")
    
    print("\n=== RPY角度 (弧度) ===")
    print(f"Roll:  {rpy[0]:.6f}")
    print(f"Pitch: {rpy[1]:.6f}")
    print(f"Yaw:   {rpy[2]:.6f}")
    
    print("\n=== RPY角度 (度) ===")
    print(f"Roll:  {np.degrees(rpy[0]):.3f}°")
    print(f"Pitch: {np.degrees(rpy[1]):.3f}°")
    print(f"Yaw:   {np.degrees(rpy[2]):.3f}°")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将相机标定结果转换为外参矩阵')
    parser.add_argument('calibration_file', help='标定结果JSON文件路径')
    parser.add_argument('--config', '-c', 
                       default='./configs/camera_config.yaml',
                       help='相机配置文件路径')
    parser.add_argument('--update', '-u', action='store_true',
                       help='是否更新相机配置文件')
    parser.add_argument('--world2robot', '-w', action='store_true',
                       help='是否生成world2robot_homo变换矩阵')
    parser.add_argument('--robot-state', '-r',
                       help='robot_state.json文件路径，用于更新world2robot_homo矩阵')
    
    args = parser.parse_args()
    
    try:
        # 检查输入文件是否存在
        if not os.path.exists(args.calibration_file):
            raise FileNotFoundError(f"标定文件不存在: {args.calibration_file}")
        
        # 加载标定结果
        print(f"正在加载标定文件: {args.calibration_file}")
        calibration_data = load_calibration_result(args.calibration_file)
        
        # 验证标定数据
        validate_calibration_data(calibration_data)
        
        # 提取位置和四元数
        position = calibration_data['position']
        quaternion = calibration_data['orientation']
        
        # 创建外参矩阵
        extrinsic_matrix = create_extrinsic_matrix(position, quaternion)
        
        # 转换为RPY角度
        rpy = quaternion_to_rpy(quaternion)
        
        # 创建world2robot_homo矩阵（如果需要）
        world2robot_homo = None
        if args.world2robot:
            world2robot_homo = create_world2robot_homo(position, quaternion)
        
        # 打印外参矩阵和相关信息
        print_extrinsic_matrix(extrinsic_matrix, position, quaternion, rpy, world2robot_homo)
        
        # 如果指定了更新配置文件
        if args.update:
            print(f"\n正在更新配置文件: {args.config}")
            update_camera_config(args.config, extrinsic_matrix, position, quaternion, rpy, world2robot_homo)
            print(f"配置文件已更新: {args.config}")
        
        # 如果指定了更新robot_state文件且生成了world2robot_homo矩阵
        if args.robot_state and world2robot_homo is not None:
            print(f"\n正在更新robot_state文件: {args.robot_state}")
            update_robot_state(args.robot_state, world2robot_homo)
        elif args.robot_state and world2robot_homo is None:
            print("\n警告: 需要使用 --world2robot 参数生成world2robot_homo矩阵才能更新robot_state文件")
            
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        return 1
    except ValueError as e:
        print(f"数据验证错误: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return 1
    except yaml.YAMLError as e:
        print(f"YAML处理错误: {e}")
        return 1
    except Exception as e:
        print(f"未知错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    main()