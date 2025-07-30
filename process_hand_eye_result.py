#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理手眼标定结果并更新相机配置和机器人状态
"""

import json
import os
from run_calibration_pipeline import (
    validate_calibration_data,
    create_extrinsic_matrix,
    quaternion_to_rpy,
    create_world2robot_homo,
    update_camera_config,
    update_robot_state,
    print_extrinsic_matrix
)

def process_hand_eye_calibration_result(
    hand_eye_result_file: str,
    camera_config_file: str = "./configs/camera_config.yaml",
    robot_state_file: str = "./robot_state.json"
):
    """
    处理手眼标定结果并更新相机配置和机器人状态
    
    Args:
        hand_eye_result_file: 手眼标定结果文件路径
        camera_config_file: 相机配置文件路径
        robot_state_file: 机器人状态文件路径
    """
    try:
        print(f"🔄 开始处理手眼标定结果: {hand_eye_result_file}")
        
        # 1. 读取手眼标定结果
        print(f"\n📖 读取手眼标定结果文件...")
        with open(hand_eye_result_file, 'r') as f:
            calibration_data = json.load(f)
        
        print("原始标定数据:")
        print(json.dumps(calibration_data, indent=2))
        
        # 2. 验证和转换数据
        print("\n🔍 验证和转换标定数据...")
        validated_data = validate_calibration_data(calibration_data)
        
        # 3. 提取位置和四元数
        position = validated_data['position']
        quaternion = validated_data['orientation']
        
        print(f"\n📍 转换后的位置: {position}")
        print(f"🔄 转换后的四元数: {quaternion}")
        
        # 4. 创建外参矩阵
        print("\n🔧 创建外参矩阵...")
        extrinsic_matrix = create_extrinsic_matrix(position, quaternion)
        
        # 5. 转换为RPY角度
        rpy = quaternion_to_rpy(quaternion)
        
        # 6. 创建world2robot_homo矩阵
        world2robot_homo = create_world2robot_homo(position, quaternion)
        
        # 7. 打印外参矩阵和相关信息
        print("\n📊 外参矩阵和相关信息:")
        print_extrinsic_matrix(extrinsic_matrix, position, quaternion, rpy, world2robot_homo)
        
        # 8. 更新相机配置文件
        print(f"\n💾 更新相机配置文件: {camera_config_file}")
        update_camera_config(camera_config_file, extrinsic_matrix, position, quaternion, rpy, world2robot_homo)
        print(f"✅ 相机配置文件已更新")
        
        # 9. 更新机器人状态文件
        print(f"\n💾 更新机器人状态文件: {robot_state_file}")
        update_robot_state(robot_state_file, world2robot_homo)
        print(f"✅ 机器人状态文件已更新")
        
        print("\n🎉 手眼标定结果处理完成！")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='处理手眼标定结果')
    parser.add_argument('--hand_eye_file', 
                       default='./data/calibration_results/handineye/hand_eye_result.json',
                       help='手眼标定结果文件路径')
    parser.add_argument('--camera_config', 
                       default='./configs/camera_config.yaml',
                       help='相机配置文件路径')
    parser.add_argument('--robot_state', 
                       default='./robot_state.json',
                       help='机器人状态文件路径')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.hand_eye_file):
        print(f"❌ 手眼标定结果文件不存在: {args.hand_eye_file}")
        return 1
    
    # 处理手眼标定结果
    success = process_hand_eye_calibration_result(
        args.hand_eye_file,
        args.camera_config,
        args.robot_state
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())