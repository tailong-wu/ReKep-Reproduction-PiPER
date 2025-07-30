#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机标定流水线脚本 - 整合版

依次执行以下操作：
1. 从Docker容器复制标定结果文件
2. 转换最新的标定结果为外参矩阵
3. 更新robot_state.json中的world2robot_homo矩阵

整合了原本的三个脚本功能：
- run_calibration_pipeline.py
- convert_calibration_to_extrinsics.py  
- update_world2robot_homo.py

支持的标定数据格式：
1. 四元数格式：
   {
       "position": [x, y, z],
       "orientation": [qx, qy, qz, qw]
   }

2. 旋转矩阵格式：
   {
       "rotation_matrix": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
       "translation_vector": [x, y, z] 或 [[x], [y], [z]]
   }
"""

import os
import subprocess
import glob
import sys
import json
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from typing import Dict, Any, Tuple


def run_command(command, description):
    """执行命令并处理错误
    
    Args:
        command: 要执行的命令（字符串或列表）
        description: 命令描述
    
    Returns:
        bool: 执行是否成功
    """
    print(f"\n{'='*50}")
    print(f"正在执行: {description}")
    print(f"命令: {command if isinstance(command, str) else ' '.join(command)}")
    print(f"{'='*50}")
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=True, 
                                  capture_output=True, text=True)
        
        if result.stdout:
            print("输出:")
            print(result.stdout)
        
        print(f"✅ {description} 执行成功")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 执行失败")
        print(f"错误代码: {e.returncode}")
        if e.stdout:
            print(f"标准输出: {e.stdout}")
        if e.stderr:
            print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ {description} 执行异常: {e}")
        return False


def find_latest_calibration_file(result_dir="./result"):
    """查找最新的标定文件
    
    Args:
        result_dir: 结果目录路径
    
    Returns:
        str: 最新标定文件的路径，如果没有找到则返回None
    """
    pattern = os.path.join(result_dir, "*_calibration.json")
    calibration_files = glob.glob(pattern)
    
    if not calibration_files:
        print(f"❌ 在 {result_dir} 目录中没有找到标定文件")
        return None
    
    # 按文件修改时间排序，获取最新的文件
    latest_file = max(calibration_files, key=os.path.getmtime)
    print(f"📁 找到最新的标定文件: {latest_file}")
    return latest_file


# ==================== 标定结果转换功能 ====================

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


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> list:
    """将旋转矩阵转换为四元数
    
    Args:
        rotation_matrix: 3x3旋转矩阵
        
    Returns:
        四元数 [qx, qy, qz, qw]
    """
    rotation = R.from_matrix(rotation_matrix)
    quat = rotation.as_quat()  # scipy格式: [x, y, z, w]
    return quat.tolist()  # [qx, qy, qz, qw]


def validate_calibration_data(calibration_data: Dict[str, Any]) -> Dict[str, Any]:
    """验证标定数据的有效性
    
    Args:
        calibration_data: 标定结果数据
        
    Returns:
        验证和转换后的标定数据（包含position和orientation字段）
        
    Raises:
        ValueError: 当数据格式不正确时
    """
    # 检查是否包含旋转矩阵和平移向量格式
    has_rotation_matrix = 'rotation_matrix' in calibration_data and 'translation_vector' in calibration_data
    has_quaternion = 'position' in calibration_data and 'orientation' in calibration_data
    
    if has_rotation_matrix:
        # 处理旋转矩阵和平移向量格式
        rotation_matrix = np.array(calibration_data['rotation_matrix'])
        translation_vector = np.array(calibration_data['translation_vector'])
        
        # 验证旋转矩阵格式
        if rotation_matrix.shape != (3, 3):
            raise ValueError(f"旋转矩阵格式错误，期望3x3矩阵，实际: {rotation_matrix.shape}")
        
        # 验证平移向量格式
        if translation_vector.shape not in [(3,), (3, 1)]:
            raise ValueError(f"平移向量格式错误，期望(3,)或(3,1)，实际: {translation_vector.shape}")
        
        # 将平移向量转换为一维
        if translation_vector.shape == (3, 1):
            translation_vector = translation_vector.flatten()
        
        # 转换为四元数格式
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
        position = translation_vector.tolist()
        
        # 更新标定数据为标准格式
        calibration_data['position'] = position
        calibration_data['orientation'] = quaternion
        
        print(f"已将旋转矩阵和平移向量转换为四元数格式")
        print(f"位置: {position}")
        print(f"四元数: {quaternion}")
        
    elif has_quaternion:
        # 处理原有的四元数格式
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
        
    else:
        raise ValueError("标定数据必须包含以下格式之一:\n1. 'position'和'orientation'字段（四元数格式）\n2. 'rotation_matrix'和'translation_vector'字段（旋转矩阵格式）")
    
    # 返回验证后的数据
    return {
        'position': calibration_data['position'],
        'orientation': calibration_data['orientation']
    }


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
        },
        # 添加 transformation 字段，这是代码实际使用的部分
        'transformation': {
            'qx': float(quaternion[0]),
            'qy': float(quaternion[1]),
            'qz': float(quaternion[2]),
            'qw': float(quaternion[3]),
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2])
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
    
    print("\n=== transformation 字段 (实际被代码使用) ===")
    print(f"qx: {quaternion[0]:.6f}")
    print(f"qy: {quaternion[1]:.6f}")
    print(f"qz: {quaternion[2]:.6f}")
    print(f"qw: {quaternion[3]:.6f}")
    print(f"x: {position[0]:.6f}")
    print(f"y: {position[1]:.6f}")
    print(f"z: {position[2]:.6f}")


def convert_calibration_to_extrinsics(calibration_file: str, config_file: str = './configs/camera_config.yaml', robot_state_file: str = 'robot_state.json') -> bool:
    """转换标定结果为外参矩阵
    
    Args:
        calibration_file: 标定结果JSON文件路径
        config_file: 相机配置文件路径
        robot_state_file: robot_state.json文件路径
        
    Returns:
        bool: 转换是否成功
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(calibration_file):
            raise FileNotFoundError(f"标定文件不存在: {calibration_file}")
        
        # 加载标定结果
        print(f"正在加载标定文件: {calibration_file}")
        calibration_data = load_calibration_result(calibration_file)
        
        # 验证标定数据
        validated_data = validate_calibration_data(calibration_data)
        
        # 提取位置和四元数
        position = validated_data['position']
        quaternion = validated_data['orientation']
        
        # 创建外参矩阵
        extrinsic_matrix = create_extrinsic_matrix(position, quaternion)
        
        # 转换为RPY角度
        rpy = quaternion_to_rpy(quaternion)
        
        # 创建world2robot_homo矩阵
        world2robot_homo = create_world2robot_homo(position, quaternion)
        
        # 打印外参矩阵和相关信息
        print_extrinsic_matrix(extrinsic_matrix, position, quaternion, rpy, world2robot_homo)
        
        # 更新配置文件
        print(f"\n正在更新配置文件: {config_file}")
        update_camera_config(config_file, extrinsic_matrix, position, quaternion, rpy, world2robot_homo)
        print(f"配置文件已更新: {config_file}")
        
        # 更新robot_state文件
        print(f"\n正在更新robot_state文件: {robot_state_file}")
        update_robot_state(robot_state_file, world2robot_homo)
        
        return True
            
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        return False
    except ValueError as e:
        print(f"数据验证错误: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return False
    except yaml.YAMLError as e:
        print(f"YAML处理错误: {e}")
        return False
    except Exception as e:
        print(f"未知错误: {e}")
        return False


def create_calibration_data_from_matrix(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> Dict[str, Any]:
    """从旋转矩阵和平移向量创建标定数据
    
    Args:
        rotation_matrix: 3x3旋转矩阵
        translation_vector: 3x1或1x3平移向量
        
    Returns:
        标定数据字典
    """
    return {
        'rotation_matrix': rotation_matrix.tolist(),
        'translation_vector': translation_vector.tolist()
    }


def example_usage_rotation_matrix():
    """示例：如何使用旋转矩阵格式的标定数据"""
    # 示例旋转矩阵和平移向量
    rotation_matrix = np.array([
        [0.99961831, -0.0276215,   0.00053062],
        [0.02762522,  0.99957574, -0.00923017],
        [-0.00027544,  0.00924131,  0.99995726]
    ])
    
    translation_vector = np.array([
        [0.02726196],
        [0.01957654],
        [0.0054933]
    ])
    
    # 创建标定数据
    calibration_data = create_calibration_data_from_matrix(rotation_matrix, translation_vector)
    
    print("示例旋转矩阵格式的标定数据:")
    print(json.dumps(calibration_data, indent=2))
    
    # 验证和转换数据
    validated_data = validate_calibration_data(calibration_data)
    
    print("\n转换后的位置:")
    print(validated_data['position'])
    print("\n转换后的四元数:")
    print(validated_data['orientation'])
    
    return validated_data


# ==================== world2robot_homo更新功能 ====================

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


def update_world2robot_homo_from_config(camera_config_path: str = './configs/camera_config.yaml', robot_state_file: str = './robot_state.json') -> bool:
    """根据相机外参更新world2robot_homo矩阵
    
    Args:
        camera_config_path: 相机配置文件路径
        robot_state_file: 机器人状态文件路径
        
    Returns:
        bool: 更新是否成功
    """
    try:
        print("\n根据相机外参更新world2robot_homo矩阵")
        
        # 加载相机外参
        print(f"\n加载相机配置文件: {camera_config_path}")
        
        with open(camera_config_path, 'r') as f:
            camera_config = yaml.safe_load(f)
            print("\n相机外参:")
            print(f"位置: x={camera_config['transformation']['x']}, y={camera_config['transformation']['y']}, z={camera_config['transformation']['z']}")
            print(f"四元数: qx={camera_config['transformation']['qx']}, qy={camera_config['transformation']['qy']}, qz={camera_config['transformation']['qz']}, qw={camera_config['transformation']['qw']}")
        
        camera_to_base = load_camera_extrinsics(camera_config_path)
        
        # 创建world2robot_homo矩阵
        world2robot_homo = camera_to_base  # 假设世界坐标系与相机坐标系重合
        print("\nworld2robot_homo变换矩阵:")
        print(np.array2string(world2robot_homo, precision=6, suppress_small=True))
        
        # 更新robot_state.json文件
        update_robot_state(robot_state_file, world2robot_homo)
        
        print(f"\n已成功更新 {robot_state_file} 中的world2robot_homo矩阵!")
        return True
        
    except Exception as e:
        print(f"更新world2robot_homo矩阵失败: {e}")
        return False


# ==================== 主流水线函数 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='相机标定流水线 - 整合版')
    parser.add_argument('--example', action='store_true', 
                       help='运行旋转矩阵格式的示例')
    args = parser.parse_args()
    
    if args.example:
        print("运行旋转矩阵格式示例...")
        example_usage_rotation_matrix()
        return 0
    
    print("\033[94m🚀 开始执行相机标定流水线 - 整合版\033[0m")
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success_count = 0
    total_steps = 3
    
    # 步骤1: 执行docker_cp.sh
    docker_cp_script = "./docker_cp.sh"
    if os.path.exists(docker_cp_script):
        if run_command(f"bash {docker_cp_script}", "从Docker容器复制标定结果"):
            success_count += 1
    else:
        print(f"❌ Docker复制脚本不存在: {docker_cp_script}")
        print("跳过Docker复制步骤")
        total_steps -= 1
    
    # 步骤2: 查找最新的标定文件并转换为外参矩阵
    latest_calibration = find_latest_calibration_file()
    if latest_calibration:
        if convert_calibration_to_extrinsics(
            calibration_file=latest_calibration,
            config_file='./configs/camera_config.yaml',
            robot_state_file='robot_state.json'
        ):
            success_count += 1
            print("✅ 标定结果转换为外参矩阵成功")
        else:
            print("❌ 标定结果转换失败")
    else:
        print("❌ 无法找到标定文件，跳过转换步骤")
    
    # 步骤3: 更新world2robot_homo矩阵（作为备用步骤）
    if os.path.exists('./configs/camera_config.yaml'):
        if update_world2robot_homo_from_config(
            camera_config_path='./configs/camera_config.yaml',
            robot_state_file='robot_state.json'
        ):
            success_count += 1
            print("✅ world2robot_homo矩阵更新成功")
        else:
            print("❌ world2robot_homo矩阵更新失败")
    else:
        print("❌ 相机配置文件不存在，跳过world2robot_homo更新步骤")
    
    # 总结
    print(f"\n{'='*60}")
    print(f"\033[94m📊 流水线执行完成\033[0m")
    print(f"成功步骤: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("\033[92m✅ 所有步骤执行成功！\033[0m")
        return 0
    else:
        print(f"\033[91m❌ 有 {total_steps - success_count} 个步骤执行失败\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())