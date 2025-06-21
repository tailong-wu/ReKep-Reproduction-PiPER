#!/usr/bin/env python3
# -*-coding:utf8-*-
import json
import time
import numpy as np
from piper_sdk import C_PiperInterface_V2
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

# Note: This snippet is adapted from pyroki/examples/pyroki_snippets/_solve_ik.py
import jax
import jax.numpy as jnp
import jaxlie
import jaxls

@jax.jit
def _solve_ik_jax(
    robot: pk.Robot,
    T_world_target: jaxlie.SE3,
    target_link_index: jax.Array,
) -> jax.Array:
    """Solves the basic IK problem. Returns joint configuration."""
    joint_var = robot.joint_var_cls(0)
    vars = [joint_var]

    # Weights and margins defined directly in factors
    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=T_world_target,
            target_link_index=target_link_index,
            pos_weight=5.0,
            ori_weight=1.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var=joint_var,
            weight=100.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(joint_var.default_factory()),
            weight=0.01,
        ),
    ]

    sol = (
        jaxls.LeastSquaresProblem(costs, vars)
        .analyze()
        .solve(verbose=False, linear_solver="dense_cholesky")
    )
    return sol[joint_var]

def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
) -> np.ndarray:
    target_link_idx = robot.links.names.index(target_link_name)

    T_world_targets = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    cfg = _solve_ik_jax(
        robot,
        T_world_targets,
        jnp.array(target_link_idx),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return np.array(cfg)

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

def wait_for_joints(piper: C_PiperInterface_V2, target_joints_rad, joints_mdeg, tolerance=0.1):
    """等待机械臂到达指定关节角度, 期间会持续发送目标指令"""
    # factor to convert from 0.001 degrees (from robot) to radians
    factor_to_rad = np.pi / (180 * 1000)
    while True:
        piper.JointCtrl(joints_mdeg[0], joints_mdeg[1], joints_mdeg[2], joints_mdeg[3], joints_mdeg[4], joints_mdeg[5])
        
        status = piper.GetArmLowSpdInfoMsgs()
        current_joints_rad = np.array([
            status.motor_1.foc_status.pos_estimate * factor_to_rad,
            status.motor_2.foc_status.pos_estimate * factor_to_rad,
            status.motor_3.foc_status.pos_estimate * factor_to_rad,
            status.motor_4.foc_status.pos_estimate * factor_to_rad,
            status.motor_5.foc_status.pos_estimate * factor_to_rad,
            status.motor_6.foc_status.pos_estimate * factor_to_rad,
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
    # Initialize robot model for IK
    print("正在加载机器人模型...")
    urdf = load_robot_description("piper_description")
    robot = pk.Robot.from_urdf(urdf)
    target_link_name = "link7" # or gripper_base, depending on URDF
    print("机器人模型加载完毕。")

    # Initialize real robot connection
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    enable_fun(piper)

    with open('./outputs/action.json', 'r') as f:
        action_data = json.load(f)

    ee_action_seq = action_data['ee_action_seq']

    # Conversion factor for JointCtrl: rad -> 0.001deg
    joint_factor = 180 * 1000 / np.pi

    piper.MotionCtrl_2(0x01, 0x01, 100, 0x00) # Switched to joint control mode

    for i, action in enumerate(ee_action_seq):
        print(f"\n执行第 {i+1} 个动作: {action}")

        position_m = np.array(action[:3])
        quaternion = action[3:7]  # qx, qy, qz, qw
        # pyroki/jaxlie expects wxyz
        target_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

        print("正在进行逆运动学求解...")
        joint_angles_rad = solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=position_m,
            target_wxyz=target_wxyz,
        )
        print(f"求解出的关节角度 (rad): {joint_angles_rad}")

        # Convert to 0.001 degrees for the controller
        joints_mdeg = [round(angle * joint_factor) for angle in joint_angles_rad]
        print(f"发送给控制器的关节角度: {joints_mdeg}")

        gripper_val = action[7]
        if gripper_val == 0:
            piper.GripperCtrl(0, 1000, 0x01, 0)  # 张开
        else:
            piper.GripperCtrl(1000, 1000, 0x01, 0)  # 闭合

        print("等待机械臂到达目标关节角度...")
        wait_for_joints(piper, joint_angles_rad, joints_mdeg)

        input("按 Enter键 继续下一个动作...")

    print("所有动作执行完毕。")