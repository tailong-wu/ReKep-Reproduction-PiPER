#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据相机外参更新world2robot_homo矩阵

将相机配置文件中的位置和四元数转换为world2robot_homo矩阵，并更新到robot_state.json文件中。
"""

import json
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from typing import Dict, Any, Tuple


def load_camera_extrinsics(camera_config_path: str = './configs/camera_config.yaml') -> np.ndarray:
    """加载相机外参并转换为变换矩阵
    
    Args:
        camera_config_path: 相机配置文件路径
        
    Returns:
        4x4变换矩阵 (Camera to Base)
    """
    with open(camera_config_path, 'r') as f:
        extrinsics_data = yaml.safe_load(f)
    
    # 提取四元数和位置
    qx = extrinsics_data['transformation']['qx']
    qy = extrinsics_data['transformation']['qy']
    qz = extrinsics_data['transformation']['qz']
    qw = extrinsics_data['transformation']['qw']
    
    # 转换为旋转矩阵 (Scipy格式: [x,y,z,w])
    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    # 提取平移向量
    tx = extrinsics_data['transformation']['x']
    ty = extrinsics_data['transformation']['y']
    tz = extrinsics_data['transformation']['z']
    
    # 创建4x4变换矩阵
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rot
    extrinsics[:3, 3] = [tx, ty, tz]
    
    return extrinsics


def create_world2robot_homo(camera_to_base: np.ndarray) -> np.ndarray:
    """创建world2robot_homo变换矩阵
    
    Args:
        camera_to_base: 相机到机器人基座的变换矩阵
        
    Returns:
        4x4 world2robot_homo变换矩阵 (World to Robot Base)
    """
    # 假设世界坐标系与相机坐标系重合
    # 在实际应用中，这个假设可能需要根据具体的坐标系定义进行调整
    world2robot_homo = camera_to_base
    
    return world2robot_homo


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


def main():
    """主程序"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='根据相机外参更新world2robot_homo矩阵')
    parser.add_argument('--camera-config', type=str, default='./configs/camera_config.yaml',
                        help='相机配置文件路径 (默认: ./configs/camera_config.yaml)')
    parser.add_argument('--robot-state', type=str, default='./robot_state.json',
                        help='机器人状态文件路径 (默认: ./robot_state.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细信息')
    args = parser.parse_args()
    
    print("\033[94m根据相机外参更新world2robot_homo矩阵\033[0m")
    
    # 加载相机外参
    camera_config_path = args.camera_config
    print(f"\n加载相机配置文件: {camera_config_path}")
    
    try:
        with open(camera_config_path, 'r') as f:
            camera_config = yaml.safe_load(f)
            print("\n相机外参:")
            print(f"位置: x={camera_config['transformation']['x']}, y={camera_config['transformation']['y']}, z={camera_config['transformation']['z']}")
            print(f"四元数: qx={camera_config['transformation']['qx']}, qy={camera_config['transformation']['qy']}, qz={camera_config['transformation']['qz']}, qw={camera_config['transformation']['qw']}")
    except Exception as e:
        print(f"\033[91m无法读取相机配置文件: {e}\033[0m")
        return
    
    camera_to_base = load_camera_extrinsics(camera_config_path)
    if args.verbose:
        print("\n相机到机器人基座的变换矩阵:")
        print(np.array2string(camera_to_base, precision=6, suppress_small=True))
    
    # 创建world2robot_homo矩阵
    world2robot_homo = create_world2robot_homo(camera_to_base)
    print("\nworld2robot_homo变换矩阵:")
    print(np.array2string(world2robot_homo, precision=6, suppress_small=True))
    
    # 更新robot_state.json文件
    robot_state_file = args.robot_state
    
    # 显示更新前的矩阵
    if os.path.exists(robot_state_file):
        try:
            with open(robot_state_file, 'r') as f:
                robot_state = json.load(f)
                if 'misc' in robot_state and 'world2robot_homo' in robot_state['misc'] and args.verbose:
                    print("\n更新前的world2robot_homo矩阵:")
                    print(np.array2string(np.array(robot_state['misc']['world2robot_homo']), precision=6, suppress_small=True))
        except Exception as e:
            print(f"\033[93m无法读取现有robot_state文件: {e}\033[0m")
    
    update_robot_state(robot_state_file, world2robot_homo)
    
    print(f"\n\033[92m已成功更新 {robot_state_file} 中的world2robot_homo矩阵!\033[0m")


if __name__ == "__main__":
    main()