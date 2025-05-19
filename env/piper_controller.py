#!/usr/bin/env python3
# -*-coding:utf8-*-
import asyncio
from typing import List
import time
from piper_sdk import C_PiperInterface_V2
from env.piper_mock_interface import Mock_C_PiperInterface_V2

class PiperController:
    def __init__(self, interface: str = "can0", test_mode: bool = False):
        """
        初始化 PiperController 类并连接到机械臂接口
        :param interface: CAN 接口名称 (默认 "can0")
        :param test_mode: 是否启用测试模式（使用 Mock）
        """
        if test_mode:
            self.piper = Mock_C_PiperInterface_V2(interface)
        else:
            self.piper = C_PiperInterface_V2(interface)
        self.piper.ConnectPort()  # 连接到机械臂接口
        self.piper.EnableArm(7)  # 使能机械臂

    def enable_arm(self, timeout: int = 5):
        """
        使能机械臂并检测使能状态
        :param timeout: 超时时间（秒）
        """
        enable_flag = False
        start_time = time.time()

        while not enable_flag:
            elapsed_time = time.time() - start_time
            enable_flag = all([
                self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status,
                self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status,
            ])
            if enable_flag:
                print("机械臂已成功使能")
            elif elapsed_time > timeout:
                print("机械臂使能超时，退出程序")
                exit(0)
            else:
                self.piper.EnableArm(7)
                time.sleep(1)

    def action(self, end_pose: List[float],grip_angle: int, grip_force: int = 1000):
        """
        移动机械臂到指定的末端位姿，注意action的end_pose需要同时提供夹爪控制
        :param end_pose: 包含末端位姿的列表，格式为 [X, Y, Z, RX, RY, RZ, Joint_6]
        :param grip_force: 手爪控制力（默认 1000）
        """
        if len(end_pose) != 6:
            raise ValueError("end_pose 必须包含 6 个元素：[X, Y, Z, RX, RY, RZ]")
        

        factor = 1000  # 缩放因子
        X, Y, Z, RX, RY, RZ = [round(val * factor) for val in end_pose]

        print(f"移动机械臂到位置: X={X}, Y={Y}, Z={Z}, RX={RX}, RY={RY}, RZ={RZ}")
        print(f"夹爪角度和力矩: 角度={grip_angle}，力矩={grip_force}")
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        self.piper.GripperCtrl(abs(grip_angle), grip_force, 0x01, 0)
        print("机械臂已到达目标位置")

    def send_pose_to_robot(self, end_pose):
        """
        Sends a pose to the robot controller to execute using moveL.

        :param pose: The pose to be executed (x, y, z, Rx, Ry, Rz).
        """
        try:
            if len(end_pose) != 6:
                raise ValueError("end_pose 必须包含 7 个元素：[X, Y, Z, RX, RY, RZ]")
            factor = 1000  # 缩放因子
            X, Y, Z, RX, RY, RZ, joint_6 = [round(val * factor) for val in end_pose]
            print(f"移动机械臂到位置: X={X}, Y={Y}, Z={Z}, RX={RX}, RY={RY}, RZ={RZ}, Joint_6={joint_6}")
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            print("机械臂已到达目标位置")
        except Exception as e:
            print("Error while sending pose to robot:", e)
        # finally:
        #     pass
    
    def get_robot_pose_and_joints(self):
        """
        获取机械臂的末端位姿和关节角度
        :return: 返回末端位姿和关节角度的元组 (end_pose, joint_positions)
        """
        # 由于https://github.com/agilexrobotics/piper_sdk/blob/master/interface/piper_interface_v2.py#L683 中的GetArmEndPoseMsgs和GetArmJointMsgs
        # TODO 确认这个返回的数值转换是否正确，或者来说是否可能完全不需要转换，直接保证和最后action的数值相同，返回的是0.001mm 和 0.001度所以可能要进行转换，但是有可能接口都使用这个单位
        try:
            arm_end_pose_msgs = self.piper.GetArmEndPoseMsgs()
            end_pose = [
                arm_end_pose_msgs.x,
                arm_end_pose_msgs.y,
                arm_end_pose_msgs.z,
                arm_end_pose_msgs.rx,
                arm_end_pose_msgs.ry,
                arm_end_pose_msgs.rz
            ]
            
            arm_low_spd_info_msgs = self.piper.GetArmLowSpdInfoMsgs()
            joint_positions = [
                arm_low_spd_info_msgs.motor_1.act_pos,
                arm_low_spd_info_msgs.motor_2.act_pos,
                arm_low_spd_info_msgs.motor_3.act_pos,
                arm_low_spd_info_msgs.motor_4.act_pos,
                arm_low_spd_info_msgs.motor_5.act_pos,
                arm_low_spd_info_msgs.motor_6.act_pos,
            ]

            return end_pose, joint_positions

        except Exception as e:
            print("获取机械臂数据时发生错误:", e)
            return None, None
    def get_tcp_pose(self):
        """
        获取当前末端执行器位姿
        Returns:
            list: 当前TCP位姿 [x, y, z, rx, ry, rz]
        """
        pose, _ = self.get_robot_pose_and_joints()
        return pose

    def get_joint_positions(self):
        """
        获取当前关节位置
        """
        _, joint_positions = self.get_robot_pose_and_joints()
        return joint_positions

    def control_gripper(self, width=0,effort=100, close=True):
        """
        Controls the Piper two-finger parallel gripper.
        # TODO 夹爪范围测试，没有范围无法实现的width到gripper_angle的改变
        :param width: Target width of the gripper (0 for fully closed, 255 for fully open).
        :param force: Force applied by the gripper (default: 100).
        :param close: Whether to close (True) or open (False) the gripper.
        """
        # TODO width to gripper_angle 
        angle = None
        pass

        async def execute_gripper_commands():
            # self.piper.GripperCtrl(0,1000,0x01, 0)
            if close:
                # TODO 明确当前夹爪角度大于目标角度
                await self.piper.GripperCtrl(angle,effort)
                print("Gripper closed.")
            else:
                await self.piper.GripperCtrl(angle,effort)
                print("Gripper opened.")

        try:
            # Get the current running event loop or create a new one
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the asynchronous function in the loop
            if loop.is_running():
                asyncio.ensure_future(execute_gripper_commands())
            else:
                loop.run_until_complete(execute_gripper_commands())

        except Exception as e:
            print("Error controlling gripper:", e)




if __name__ == "__main__":
    # 示例代码
    controller = PiperController()
    controller.enable_arm()

    # 示例末端位姿
    target_pose = [55.0, 0.0, 206.0, 0.0, 85.0, 0.0, 0.0]
    controller.action(target_pose)