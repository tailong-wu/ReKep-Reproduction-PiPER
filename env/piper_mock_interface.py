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
        # Initial pose in 0.001mm units (to match real PiPer interface)
        # Converting from meters: [0.054957, 0., 0.260009, 0, 0.084958, 0]
        self.current_pose = [54957, 0, 260009, 0, 84958, 0]  # 0.001mm units
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
    
    def MotionCtrl_2(self, param1: int, param2: int, param3: int, param4: int):
        print(f"[Mock] 运动控制，参数: {param1}, {param2}, {param3}, {param4}")
    
    def JointCtrl(self, j1: int, j2: int, j3: int, j4: int, j5: int, j6: int):
        print(f"[Mock] 关节控制: J1={j1}, J2={j2}, J3={j3}, J4={j4}, J5={j5}, J6={j6}")
        self.joint_positions = [j1, j2, j3, j4, j5, j6]

    def GetArmEndPoseMsgs(self):
        class MockEndPose:
            def __init__(self, parent):
                self.X_axis = parent.current_pose[0]
                self.Y_axis = parent.current_pose[1]
                self.Z_axis = parent.current_pose[2]
                self.RX_axis = parent.current_pose[3]
                self.RY_axis = parent.current_pose[4]
                self.RZ_axis = parent.current_pose[5]
        
        class MockEndPoseMsgs:
            def __init__(self, parent):
                self.end_pose = MockEndPose(parent)

        return MockEndPoseMsgs(self)

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
    
    def GetArmJointMsgs(self):
        """获取机械臂关节消息
        
        Returns:
            MockJointMsgs: 包含关节状态的模拟消息对象
        """
        class MockJointState:
            def __init__(self, joint_positions):
                # joint_positions are in radians, convert to 0.001 degrees for PiPer format
                import numpy as np
                self.joint_1 = int(joint_positions[0] * 180 * 1000 / np.pi)
                self.joint_2 = int(joint_positions[1] * 180 * 1000 / np.pi)
                self.joint_3 = int(joint_positions[2] * 180 * 1000 / np.pi)
                self.joint_4 = int(joint_positions[3] * 180 * 1000 / np.pi)
                self.joint_5 = int(joint_positions[4] * 180 * 1000 / np.pi)
                self.joint_6 = int(joint_positions[5] * 180 * 1000 / np.pi)
        
        class MockJointMsgs:
            def __init__(self, joint_positions):
                self.joint_state = MockJointState(joint_positions)
        
        return MockJointMsgs(self.joint_positions)