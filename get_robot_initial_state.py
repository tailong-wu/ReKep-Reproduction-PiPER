#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
机器人初始状态获取程序
获取机器人的关节角度和末端执行器pose，保存到JSON文件中

Author: ReKep Team
Date: 2025
"""

import json
import os
import time
import numpy as np
from typing import Optional, List, Dict, Any
from piper_sdk import *

def get_robot_initial_state(piper_interface: C_PiperInterface_V2) -> Dict[str, Any]:
    """
    获取机器人的初始状态信息
    
    Args:
        piper_interface: 已连接的机械臂接口对象
        
    Returns:
        dict: 包含关节角度和末端执行器pose的字典
    """
    try:
        # 获取当前关节角度
        joint_angles_msgs = piper_interface.GetArmJointMsgs()
        
        # 转换关节角度单位：从0.001度转换为弧度
        joint_angles = [
            joint_angles_msgs.joint_state.joint_1 / 1000.0 * np.pi / 180.0,  # 0.001度 -> 弧度
            joint_angles_msgs.joint_state.joint_2 / 1000.0 * np.pi / 180.0,
            joint_angles_msgs.joint_state.joint_3 / 1000.0 * np.pi / 180.0,
            joint_angles_msgs.joint_state.joint_4 / 1000.0 * np.pi / 180.0,
            joint_angles_msgs.joint_state.joint_5 / 1000.0 * np.pi / 180.0,
            joint_angles_msgs.joint_state.joint_6 / 1000.0 * np.pi / 180.0
        ]
        
        # 获取当前末端位姿
        arm_end_pose_msgs = piper_interface.GetArmEndPoseMsgs()
        
        # 转换位置单位：从0.001mm转换为米
        ee_position = [
            arm_end_pose_msgs.end_pose.X_axis / 1000000.0,  # 0.001mm -> m
            arm_end_pose_msgs.end_pose.Y_axis / 1000000.0,
            arm_end_pose_msgs.end_pose.Z_axis / 1000000.0
        ]
        
        # 转换旋转单位：从0.001度转换为弧度
        ee_orientation = [
            arm_end_pose_msgs.end_pose.RX_axis / 1000.0 * np.pi / 180.0,  # 0.001度 -> 弧度
            arm_end_pose_msgs.end_pose.RY_axis / 1000.0 * np.pi / 180.0,
            arm_end_pose_msgs.end_pose.RZ_axis / 1000.0 * np.pi / 180.0
        ]
        
        # 构建状态字典
        robot_state = {
            "timestamp": time.time(),
            "joint_angles": {
                "values": joint_angles,
                "unit": "radians",
                "description": "Joint angles in radians [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]"
            },
            "end_effector_pose": {
                "position": {
                    "values": ee_position,
                    "unit": "meters",
                    "description": "End effector position in meters [x, y, z]"
                },
                "orientation": {
                    "values": ee_orientation,
                    "unit": "radians",
                    "description": "End effector orientation in radians [rx, ry, rz]"
                }
            },
            "raw_data": {
                "joint_angles_raw": {
                    "joint_1": joint_angles_msgs.joint_state.joint_1,
                    "joint_2": joint_angles_msgs.joint_state.joint_2,
                    "joint_3": joint_angles_msgs.joint_state.joint_3,
                    "joint_4": joint_angles_msgs.joint_state.joint_4,
                    "joint_5": joint_angles_msgs.joint_state.joint_5,
                    "joint_6": joint_angles_msgs.joint_state.joint_6,
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
        
        return robot_state
        
    except Exception as e:
        print(f"\033[91m获取机器人状态失败: {e}\033[0m")
        raise

def save_robot_initial_state(robot_state: Dict[str, Any], output_path: str = "robot_state.json") -> None:
    """
    保存机器人初始状态到JSON文件
    
    Args:
        robot_state: 机器人状态字典
        output_path: 输出文件路径
    """
    try:
        # 如果robot_state.json已存在，读取它并更新相关字段
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_state = json.load(f)
                
            # 保留现有的rekep_stage和misc字段
            if 'rekep_stage' in existing_state:
                robot_state['rekep_stage'] = existing_state['rekep_stage']
            else:
                robot_state['rekep_stage'] = 1
                
            if 'misc' in existing_state:
                robot_state['misc'] = existing_state['misc']
            
            # 添加安全和控制信息
            robot_state['safety_info'] = {
                "collision_status": "false",
                "safety_status": "normal",
                "errors": []
            }
            robot_state['control_info'] = {
                "control_mode": "position",
                "operation_mode": "auto"
            }
            robot_state['gripper_info'] = {
                "state": 0.0
            }
        
        # 转换为统一格式
        unified_state = {
            "timestamp": robot_state["timestamp"],
            "rekep_stage": robot_state.get("rekep_stage", 1),
            "joint_info": {
                "joint_positions": robot_state["joint_angles"]["values"],
                "reset_joint_pos": robot_state["joint_angles"]["values"]
            },
            "initial_position": {
                "joint_positions": robot_state["joint_angles"]["values"],
                "ee_position": robot_state["end_effector_pose"]["position"]["values"],
                "ee_orientation": robot_state["end_effector_pose"]["orientation"]["values"]
            },
            "ee_info": {
                "position": robot_state["end_effector_pose"]["position"]["values"],
                "orientation": {
                    "quaternion": [0.0, 0.0, 0.0, 1.0],  # 默认四元数
                    "euler": robot_state["end_effector_pose"]["orientation"]["values"],
                    "unit": "radians",
                    "description": "End effector orientation in radians [rx, ry, rz]"
                }
            },
            "gripper_info": robot_state.get("gripper_info", {"state": 0.0}),
            "safety_info": robot_state.get("safety_info", {
                "collision_status": "false",
                "safety_status": "normal",
                "errors": []
            }),
            "control_info": robot_state.get("control_info", {
                "control_mode": "position",
                "operation_mode": "auto"
            }),
            "misc": robot_state.get("misc", {}),
            "raw_data": robot_state["raw_data"]
        }
        
        # 如果已存在的文件中有initial_position字段，保留它
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_state = json.load(f)
                    if 'initial_position' in existing_state:
                        # 如果是初始化操作，更新initial_position
                        if output_path.endswith('robot_state.json'):
                            # 保持initial_position与当前值一致
                            pass
                        else:
                            # 对于备份文件，保留现有的initial_position
                            unified_state['initial_position'] = existing_state['initial_position']
            except Exception as e:
                print(f"\033[93m读取现有{output_path}失败: {e}\033[0m")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unified_state, f, indent=2, ensure_ascii=False)
        
        print(f"\033[92m机器人状态已保存到: {output_path}\033[0m")
        
    except Exception as e:
        print(f"\033[91m保存文件失败: {e}\033[0m")
        raise

def display_robot_state(robot_state: Dict[str, Any]) -> None:
    """
    显示机器人状态信息
    
    Args:
        robot_state: 机器人状态字典
    """
    print("\n\033[94m=== 机器人初始状态信息 ===\033[0m")
    
    # 显示关节角度
    joint_angles = robot_state["joint_angles"]["values"]
    print(f"\n\033[93m关节角度 (弧度):\033[0m")
    for i, angle in enumerate(joint_angles, 1):
        print(f"  Joint {i}: {angle:.6f} rad ({angle * 180.0 / np.pi:.3f}°)")
    
    # 显示末端执行器位置
    ee_position = robot_state["end_effector_pose"]["position"]["values"]
    print(f"\n\033[93m末端执行器位置 (米):\033[0m")
    print(f"  X: {ee_position[0]:.6f} m")
    print(f"  Y: {ee_position[1]:.6f} m")
    print(f"  Z: {ee_position[2]:.6f} m")
    
    # 显示末端执行器姿态
    ee_orientation = robot_state["end_effector_pose"]["orientation"]["values"]
    print(f"\n\033[93m末端执行器姿态 (弧度):\033[0m")
    print(f"  RX: {ee_orientation[0]:.6f} rad ({ee_orientation[0] * 180.0 / np.pi:.3f}°)")
    print(f"  RY: {ee_orientation[1]:.6f} rad ({ee_orientation[1] * 180.0 / np.pi:.3f}°)")
    print(f"  RZ: {ee_orientation[2]:.6f} rad ({ee_orientation[2] * 180.0 / np.pi:.3f}°)")
    
    print(f"\n\033[92m时间戳: {time.ctime(robot_state['timestamp'])}\033[0m")

def main():
    """
    主程序
    """
    print("\033[94m机器人初始状态获取程序\033[0m")
    print("按下Enter键获取当前机器人状态并保存到JSON文件")
    print("按下Ctrl+C退出程序")
    print("-" * 50)
    
    try:
        # 连接机械臂
        piper = C_PiperInterface_V2()
        piper.ConnectPort()
        print("\033[92m已连接到机械臂\033[0m")
        
        while True:
            try:
                # 等待用户输入
                input("\n按下Enter键获取机器人初始状态...")
                
                # 获取机器人状态
                robot_state = get_robot_initial_state(piper)
                
                # 显示状态信息
                display_robot_state(robot_state)
                
                # 保存到文件
                timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                output_filename = f"robot_state_backup_{timestamp_str}.json"
                save_robot_initial_state(robot_state, output_filename)
                
                # 同时更新robot_state.json
                save_robot_initial_state(robot_state, "robot_state.json")
                
            except KeyboardInterrupt:
                print("\n\033[93m程序被用户中断\033[0m")
                break
            except Exception as e:
                print(f"\033[91m获取状态失败: {e}\033[0m")
                continue
                
    except Exception as e:
        print(f"\033[91m连接机械臂失败: {e}\033[0m")
        print("请确保机械臂已连接并且piper_sdk已正确安装")
        return
    
    print("\033[94m程序结束\033[0m")

if __name__ == "__main__":
    main()