#!/usr/bin/env python3
# -*- coding:utf8 -*-

class Mock_C_PiperInterface_V2:
    """
    用于测试的 Piper SDK 接口模拟类
    """

    def __init__(self, interface: str):
        self.interface = interface
        print(f"[Mock] 初始化 Piper 接口：{interface}")
        self.arm_enabled = False
        self.current_pose = [0.054957, 0., 0.260009, 0, 0.084958, 0]  # m单位 但是piper是mm or 0.001mm 单位
        self.joint_positions = [0, 0, 0, 0, 0, 0]
        self.gripper_angle = 0
        self.gripper_force = 0

    def ConnectPort(self):
        print("[Mock] 连接机械臂端口成功")

    def EnableArm(self, value: int):
        print(f"[Mock] 正在使能机械臂，值为 {value}")
        self.arm_enabled = True

    def EndPoseCtrl(self, x: int, y: int, z: int, rx: int, ry: int, rz: int):
        print(f"[Mock] 控制末端到位置 X={x}, Y={y}, Z={z}, RX={rx}, RY={ry}, RZ={rz}")
        self.current_pose = [x, y, z, rx, ry, rz]

    def GripperCtrl(self, angle: int, force: int, mode: int = 0x01, timeout: int = 0):
        print(f"[Mock] 控制夹爪，角度={angle}, 力度={force}")
        self.gripper_angle = angle
        self.gripper_force = force

    def GetArmEndPoseMsgs(self):
        class MockEndPose:
            def __init__(self, parent):
                self.x = parent.current_pose[0]
                self.y = parent.current_pose[1]
                self.z = parent.current_pose[2]
                self.rx = parent.current_pose[3]
                self.ry = parent.current_pose[4]
                self.rz = parent.current_pose[5]

        return MockEndPose(self)

    def GetArmLowSpdInfoMsgs(self):
        class MockMotor:
            def __init__(self, pos, enable=True):
                self.act_pos = pos
                class FOCStatus:
                    def __init__(self, enable_status):
                        self.driver_enable_status = enable_status
                self.foc_status = FOCStatus(True)

        class MockLowSpdInfo:
            def __init__(self, parent):
                self.motor_1 = MockMotor(parent.joint_positions[0])
                self.motor_2 = MockMotor(parent.joint_positions[1])
                self.motor_3 = MockMotor(parent.joint_positions[2])
                self.motor_4 = MockMotor(parent.joint_positions[3])
                self.motor_5 = MockMotor(parent.joint_positions[4])
                self.motor_6 = MockMotor(parent.joint_positions[5])

        return MockLowSpdInfo(self)