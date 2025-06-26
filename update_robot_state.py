#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
机械臂位姿更新程序
等待用户按下Enter键后，获取当前机械臂位姿并更新robot_state.json文件

Author: ReKep Team
Date: 2025
"""

import json
import os
import time
import numpy as np
from typing import Optional
from piper_sdk import *

def update_robot_state_json(ee_position, ee_orientation, robot_state_path="robot_state.json"):
    """更新robot_state.json文件中的ee_info部分
    
    Args:
        ee_position: 末端执行器位置 [x, y, z] (米)
        ee_orientation: 末端执行器姿态 [rx, ry, rz] (弧度)
        robot_state_path: robot_state.json文件路径
    """
    try:
        # 读取现有的robot_state.json
        if os.path.exists(robot_state_path):
            with open(robot_state_path, 'r', encoding='utf-8') as f:
                robot_state = json.load(f)
        else:
            robot_state = {}
        
        # 计算四元数
        from scipy.spatial.transform import Rotation as R
        quat = R.from_euler('xyz', ee_orientation).as_quat().tolist()
        
        # 更新ee_info部分
        robot_state["ee_info"] = {
            "position": ee_position,
            "orientation": {
                "quaternion": quat,
                "euler": ee_orientation,
                "unit": "radians",
                "description": "End effector orientation in radians [rx, ry, rz]"
            }
        }
        
        # 更新时间戳
        robot_state["timestamp"] = time.time()
        
        # 确保保留initial_position字段
        if "initial_position" not in robot_state:
            # 如果没有initial_position字段，使用当前状态创建
            robot_state["initial_position"] = {
                "joint_positions": robot_state.get("joint_info", {}).get("joint_positions", []),
                "ee_position": ee_position,
                "ee_orientation": ee_orientation
            }
            print("\033[93m创建initial_position字段\033[0m")
        
        # 保存更新后的robot_state.json
        with open(robot_state_path, 'w', encoding='utf-8') as f:
            json.dump(robot_state, f, indent=2, ensure_ascii=False)
        
        print(f"\033[92m已成功更新robot_state.json文件\033[0m")
        
    except Exception as e:
        print(f"\033[91m更新robot_state.json失败: {e}\033[0m")
        raise

def main():
    """主程序"""
    print("\033[94m机械臂位姿更新程序\033[0m")
    print("按下Enter键获取当前机械臂位姿并更新robot_state.json")
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
                input("\n按下Enter键获取当前位姿...")
                
                # 获取当前末端位姿
                arm_end_pose_msgs = piper.GetArmEndPoseMsgs()
                
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
                
                # 显示获取到的位姿信息
                print(f"\n\033[93m原始数据:\033[0m")
                print(f"X: {arm_end_pose_msgs.end_pose.X_axis} (0.001mm) -> {ee_position[0]:.6f} (m)")
                print(f"Y: {arm_end_pose_msgs.end_pose.Y_axis} (0.001mm) -> {ee_position[1]:.6f} (m)")
                print(f"Z: {arm_end_pose_msgs.end_pose.Z_axis} (0.001mm) -> {ee_position[2]:.6f} (m)")
                print(f"RX: {arm_end_pose_msgs.end_pose.RX_axis} (0.001deg) -> {ee_orientation[0]:.6f} (rad)")
                print(f"RY: {arm_end_pose_msgs.end_pose.RY_axis} (0.001deg) -> {ee_orientation[1]:.6f} (rad)")
                print(f"RZ: {arm_end_pose_msgs.end_pose.RZ_axis} (0.001deg) -> {ee_orientation[2]:.6f} (rad)")
                
                print(f"\n\033[92m转换后的位置: {ee_position}\033[0m")
                print(f"\033[92m转换后的姿态: {ee_orientation}\033[0m")
                
                # 更新robot_state.json文件
                update_robot_state_json(ee_position, ee_orientation)
                
            except KeyboardInterrupt:
                print("\n\033[93m程序被用户中断\033[0m")
                break
            except Exception as e:
                print(f"\033[91m获取位姿失败: {e}\033[0m")
                continue
                
    except Exception as e:
        print(f"\033[91m连接机械臂失败: {e}\033[0m")
        print("请确保机械臂已连接并且piper_sdk已正确安装")
        return
    
    print("\033[94m程序结束\033[0m")

if __name__ == "__main__":
    main()


