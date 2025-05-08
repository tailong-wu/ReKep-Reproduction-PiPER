
from piper_controller import PiperController
import time

class RobotEnv:
    def __init__(self, interface: str = "can0"):
        # 初始化真实机械臂环境   
        self.interface = interface

        # 初始化机械臂
        self.robot = PiperController(self.interface)
        
    def execute_action(self, action, precise=False, speed=0.05, acceleration=0.05):
        """
        Execute a single action with pose change only (without gripper control)
        
        Args:
            action: List containing position and orientation [x, y, z, rx, ry, rz]
            precise: Whether to execute precisely (usually the last action in the queue)
            speed: Movement speed (default: 0.05 m/s)
            acceleration: Movement acceleration (default: 0.05 m/s^2)
            position_only: If True, only change position and keep current orientation
        
        Returns:
            bool: Whether the execution was successful
        """
        try:            
            # Use reduced speed and acceleration for precise movements
            if precise:
                speed = speed / 2
                acceleration = acceleration / 2
                print("Performing precise action")
            
            # Execute pose change
            print(f"Moving to pose: {action}")
            self.robot.send_pose_to_robot(action, speed, acceleration)
            
            # Allow time for movement to complete
            self.sleep(0.5)  # Adjust based on movement size
            
            return True
            
        except Exception as e:
            print(f"Error performing action: {e}")
            return False
            
    def _execute_grasp_action(self):
        """
        Execute a grasp action (close gripper)
        
        Returns:
            bool: Whether the grasp was successful
        """
        try:
            print("Executing grasp action")
            self.robot.control_gripper(close=True)
            self.sleep(1.5)  # Wait for gripper to close
            print("Gripper closed")
            return True
        except Exception as e:
            print(f"Error executing grasp action: {e}")
            return False
            
    def _execute_release_action(self):
        """
        Execute a release action (open gripper)
        
        Returns:
            bool: Whether the release was successful
        """
        try:
            print("Executing release action")
            self.robot.control_gripper(close=False)
            self.sleep(1.5)  # Wait for gripper to open
            print("Gripper opened")
            return True
        except Exception as e:
            print(f"Error executing release action: {e}")
            return False
    
    def sleep(self, seconds):
        """Wait for specified duration"""
        time.sleep(seconds)

