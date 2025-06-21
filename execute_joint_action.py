#!/usr/bin/env python3
# -*-coding:utf8-*-
import json
import time
import numpy as np
from piper_sdk import C_PiperInterface_V2

def enable_fun(piper: C_PiperInterface_V2):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    timeout = 5
    start_time = time.time()
    elapsed_time_flag = False
    while not enable_flag:
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = (
            piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        )
        print("使能状态:", enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        print("--------------------")
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            break
        time.sleep(1)
    if elapsed_time_flag:
        print("程序自动使能超时,退出程序")
        exit(0)

def wait_for_joints(piper: C_PiperInterface_V2, target_joints_rad, joints_mdeg, gripper_val,tolerance=0.1):
    """等待机械臂到达指定关节角度, 期间会持续发送目标指令"""
    # factor to convert from 0.001 degrees (from robot) to radians
    factor_to_rad = np.pi / (180 * 1000)
    while True:
        if gripper_val == 0:
            piper.GripperCtrl(50*1000, 1000, 0x01, 0)  # 张开 50mm
        else:
            piper.GripperCtrl(0*1000, 1000, 0x01, 0)  # 闭合
        piper.JointCtrl(joints_mdeg[0], joints_mdeg[1], joints_mdeg[2], joints_mdeg[3], joints_mdeg[4], joints_mdeg[5])
        
        joint_msgs = piper.GetArmJointMsgs()
        current_joints_rad = np.array([
            joint_msgs.joint_state.joint_1 * factor_to_rad,
            joint_msgs.joint_state.joint_2 * factor_to_rad,
            joint_msgs.joint_state.joint_3 * factor_to_rad,
            joint_msgs.joint_state.joint_4 * factor_to_rad,
            joint_msgs.joint_state.joint_5 * factor_to_rad,
            joint_msgs.joint_state.joint_6 * factor_to_rad,
        ])

        joint_error = np.linalg.norm(current_joints_rad - target_joints_rad)
        print(f"当前关节角度 (rad): {current_joints_rad}")
        print(f"目标关节角度 (rad): {target_joints_rad}")
        print(f"关节角度误差: {joint_error:.4f}")

        if joint_error < tolerance:
            print("已到达目标关节角度。")
            break
        time.sleep(0.1)

if __name__ == "__main__":
    # Initialize real robot connection
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    enable_fun(piper)

    with open('./outputs/joint_action.json', 'r') as f:
        action_data = json.load(f)

    joint_action_seq = action_data['joint_action_seq']

    # Conversion factor for JointCtrl: rad -> 0.001deg
    joint_factor = 180 * 1000 / np.pi

    piper.MotionCtrl_2(0x01, 0x01, 100, 0x00) # Switched to joint control mode

    for i, action in enumerate(joint_action_seq):
        print(f"\n执行第 {i+1} 个动作: {action}")

        joint_angles_rad = np.array(action[:6])

        # Convert to 0.001 degrees for the controller
        joints_mdeg = [round(angle * joint_factor) for angle in joint_angles_rad]
        print(f"发送给控制器的关节角度: {joints_mdeg}")

        gripper_val = 0

        print("等待机械臂到达目标关节角度...")
        wait_for_joints(piper, joint_angles_rad, joints_mdeg,gripper_val)

        input("按 Enter键 继续下一个动作...")

    print("所有动作执行完毕。")