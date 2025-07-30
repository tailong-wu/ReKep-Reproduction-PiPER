#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
移动机械臂到初始位置的独立脚本

该脚本用于将机械臂移动到预设的初始位置，便于后续操作。
初始位置从robot_state.json文件中读取。

Author: tailong-wu
Date: 2025
"""

import os
import sys
import argparse
from main import ReKepRobotSystem


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='移动机械臂到初始位置')
    parser.add_argument('--config', '-c', 
                       default='./configs/config.yaml',
                       help='配置文件路径 (默认: ./configs/config.yaml)')
    parser.add_argument('--debug', '-d', 
                       action='store_true',
                       help='启用调试模式')
    parser.add_argument('--log-level', '-l',
                       default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    
    args = parser.parse_args()
    
    print("\033[92m=== 机械臂初始位置移动脚本 ===\033[0m")
    print(f"配置文件: {args.config}")
    print(f"调试模式: {'启用' if args.debug else '禁用'}")
    print(f"日志级别: {args.log_level}")
    print()
    
    try:
        # 创建ReKep机器人系统实例
        # 这里只需要基本的初始化，不需要完整的任务指令
        robot_system = ReKepRobotSystem(
            instruction="Move to initial position",  # 临时指令
            config_path=args.config,
            debug_mode=args.debug,
            log_level=args.log_level
        )
        
        print("\033[94m正在初始化机器人系统...\033[0m")
        
        # 只初始化必要的组件（机器人接口）
        if not robot_system._initialize_robot_only():
            print("\033[91m机器人初始化失败\033[0m")
            return 1
        
        print("\033[92m机器人初始化完成\033[0m")
        
        # 移动到初始位置
        print("\033[94m开始移动到初始位置...\033[0m")
        if robot_system.move_to_initial_position():
            print("\033[92m成功移动到初始位置！\033[0m")
            return 0
        else:
            print("\033[91m移动到初始位置失败！\033[0m")
            return 1
            
    except KeyboardInterrupt:
        print("\n\033[93m用户中断操作\033[0m")
        return 1
    except Exception as e:
        print(f"\033[91m发生错误: {e}\033[0m")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
if __name__ == '__main__':
    sys.exit(main())