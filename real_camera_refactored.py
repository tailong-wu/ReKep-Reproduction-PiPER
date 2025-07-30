## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import yaml
from typing import Tuple, Optional, Dict, Any

class RealSenseCamera:
    """RealSense相机封装类"""
    
    def __init__(self, width: int = None, height: int = None, fps: int = None, save_dir: str = "./data/realsense_captures", config_path: str = "./configs/camera_config.yaml"):
        """
        初始化RealSense相机
        
        Args:
            width: 图像宽度，如果为None则从配置文件读取
            height: 图像高度，如果为None则从配置文件读取
            fps: 帧率，如果为None则从配置文件读取
            save_dir: 保存目录
            config_path: 相机配置文件路径
        """
        # 加载相机配置
        self.config_path = config_path
        self.camera_config = self._load_camera_config()
        
        # 设置相机参数，优先使用传入的参数，否则使用配置文件中的参数
        self.width = width if width is not None else self.camera_config.get('resolution', {}).get('width', 640)
        self.height = height if height is not None else self.camera_config.get('resolution', {}).get('height', 480)
        self.fps = fps if fps is not None else self.camera_config.get('resolution', {}).get('fps', 30)
        self.save_dir = save_dir
        
        # 创建保存图片的目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化相机组件
        self.pipeline = None
        self.config = None
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.is_streaming = False
    
    def _load_camera_config(self) -> Dict:
        """
        加载相机配置文件
        
        Returns:
            Dict: 相机配置字典
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"加载相机配置文件失败: {e}，将使用默认配置")
            # 返回默认配置
            return {
                'resolution': {
                    'width': 640,
                    'height': 480,
                    'fps': 30
                },
                'intrinsics': {
                    'color': {
                        'fx': 606.60,
                        'fy': 605.47,
                        'ppx': 323.69,
                        'ppy': 247.12
                    },
                    'depth_scale': 0.001
                },
                'realsense': {
                    'serial_number': '',  # 默认为空字符串，表示使用第一个可用设备
                    'resolution': {
                        'width': 640,
                        'height': 480
                    },
                    'fps': 30
                }
            }
    
    def _save_camera_config(self) -> bool:
        """
        保存相机配置到文件
        
        Returns:
            bool: 保存是否成功
        """
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.camera_config, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"保存相机配置文件失败: {e}")
            return False
        
    def initialize_camera(self) -> bool:
        """
        初始化相机配置
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # 获取相机序列号（如果在配置文件中指定）
            serial_number = self.camera_config.get('realsense', {}).get('serial_number', None)
            
            # 查找可用的相机设备
            ctx = rs.context()
            devices = ctx.query_devices()
            
            # 打印可用的相机设备信息，帮助调试
            print("可用的相机设备:")
            device_found = False
            for i, dev in enumerate(devices):
                dev_name = dev.get_info(rs.camera_info.name)
                dev_serial = dev.get_info(rs.camera_info.serial_number)
                print(f"  设备 {i+1}: {dev_name} (序列号: {dev_serial})")
                
                # 检查是否匹配配置中的序列号
                if serial_number and dev_serial == serial_number:
                    print(f"  找到匹配的设备: {dev_name} (序列号: {dev_serial})")
                    self.config.enable_device(dev_serial)
                    device_found = True
                    break
            
            # 如果没有找到匹配的设备，使用第一个可用的设备
            if not device_found:
                if len(devices) > 0:
                    first_dev = devices[0]
                    dev_name = first_dev.get_info(rs.camera_info.name)
                    dev_serial = first_dev.get_info(rs.camera_info.serial_number)
                    print(f"  未找到序列号为 {serial_number} 的设备，使用第一个可用设备: {dev_name} (序列号: {dev_serial})")
                    self.config.enable_device(dev_serial)
                else:
                    print("  未找到任何相机设备")
                    return False
            
            # 获取设备信息用于设置分辨率
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = self.config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))
            
            # 检查RGB相机
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                return False
            
            # 配置流
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            # 使用RGB格式而不是BGR格式，避免颜色通道错误
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            
            return True
            
        except Exception as e:
            print(f"相机初始化失败: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """
        开始流式传输
        
        Returns:
            bool: 启动是否成功
        """
        try:
            if self.pipeline is None:
                if not self.initialize_camera():
                    return False
            
            # 硬件重置相机以解决 "Frame didn't arrive within 5000" 错误
            print("正在重置相机硬件...")
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            print("相机硬件重置完成")
            
            # 等待一段时间让设备重新初始化
            import time
            time.sleep(2)
            
            # Start streaming
            self.pipeline.start(self.config)
            self.is_streaming = True
            
            # 获取相机内参
            profile = self.pipeline.get_active_profile()
            
            # 获取深度流内参
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            self.depth_intrinsics = depth_profile.get_intrinsics()
            
            # 获取彩色流内参
            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            self.color_intrinsics = color_profile.get_intrinsics()
            # 获取深度到彩色图的对齐器
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            # 更新配置文件中的内参
            self._update_intrinsics_in_config()
            
            return True
            
        except Exception as e:
            print(f"启动流式传输失败: {e}")
            return False
            
    def _update_intrinsics_in_config(self):
        """
        更新配置文件中的相机内参和设备信息
        """
        if not self.is_streaming or self.depth_intrinsics is None or self.color_intrinsics is None:
            print("相机未启动或内参未获取，无法更新配置文件")
            return
        
        # 确保配置文件中有 intrinsics 部分
        if 'intrinsics' not in self.camera_config:
            self.camera_config['intrinsics'] = {}
        
        # # 更新深度相机内参
        # self.camera_config['intrinsics']['depth'] = {
        #     'fx': float(self.depth_intrinsics.fx),
        #     'fy': float(self.depth_intrinsics.fy),
        #     'ppx': float(self.depth_intrinsics.ppx),
        #     'ppy': float(self.depth_intrinsics.ppy)
        # }
        
        # 更新彩色相机内参
        self.camera_config['intrinsics']['color'] = {
            'fx': float(self.color_intrinsics.fx),
            'fy': float(self.color_intrinsics.fy),
            'ppx': float(self.color_intrinsics.ppx),
            'ppy': float(self.color_intrinsics.ppy)
        }
        
        # 获取深度比例（如果可用）
        try:
            depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            self.camera_config['intrinsics']['depth_scale'] = float(depth_scale)
        except Exception as e:
            print(f"获取深度比例失败: {e}，使用默认值 0.001")
            self.camera_config['intrinsics']['depth_scale'] = 0.001
        
        # 更新相机设备信息
        try:
            # 获取当前连接的设备信息
            device = self.pipeline.get_active_profile().get_device()
            serial_number = device.get_info(rs.camera_info.serial_number)
            
            # 确保配置文件中有 realsense 部分
            if 'realsense' not in self.camera_config:
                self.camera_config['realsense'] = {}
            
            # 更新序列号
            self.camera_config['realsense']['serial_number'] = serial_number
            
            # 更新分辨率和帧率
            if 'resolution' not in self.camera_config['realsense']:
                self.camera_config['realsense']['resolution'] = {}
            
            self.camera_config['realsense']['resolution']['width'] = self.width
            self.camera_config['realsense']['resolution']['height'] = self.height
            self.camera_config['realsense']['fps'] = self.fps
            
            print(f"已更新相机设备信息，序列号: {serial_number}")
        except Exception as e:
            print(f"更新相机设备信息失败: {e}")
        
        # 保存更新后的配置
        if self._save_camera_config():
            print("相机内参和设备信息已更新到配置文件")
        else:
            print("更新相机内参和设备信息到配置文件失败")
    
    def get_frames(self, timeout_ms: int = 1000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取一帧图像
        
        Args:
            timeout_ms: 超时时间（毫秒），默认1000ms
        
        Returns:
            Tuple[color_image, depth_image]: 彩色图像和深度图像
        """
        if not self.is_streaming:
            print("相机未启动流式传输")
            return None, None
        
        try:
            # Wait for a coherent pair of frames: depth and color with timeout
            frames = self.pipeline.wait_for_frames(timeout_ms)
            # 对齐深度图到彩色图
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"获取帧失败: {e}")
            return None, None
    
    def save_images(self, color_image: np.ndarray, depth_image: np.ndarray, 
                   timestamp: Optional[str] = None, save_dir: Optional[str] = None) -> Dict[str, str]:
        """
        保存图像
        
        Args:
            color_image: 彩色图像
            depth_image: 深度图像
            timestamp: 时间戳，如果为None则自动生成
            save_dir: 保存目录，如果为None则使用默认目录
            
        Returns:
            Dict[str, str]: 保存的文件路径
        """
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 使用指定的保存目录或默认目录
        target_dir = save_dir if save_dir is not None else self.save_dir
        
        # 确保目录存在
        os.makedirs(target_dir, exist_ok=True)
        
        saved_files = {}
        
        try:
            # 生成文件名
            color_filename = f'color_{timestamp}.png'
            depth_filename = f'depth_{timestamp}.npy'
            
            # 保存RGB图像
            color_path = os.path.join(target_dir, color_filename)
            # 由于RealSense现在输出RGB格式，需要转换为BGR格式供OpenCV保存
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(color_path, color_image_bgr)
            saved_files['color'] = color_path
            
            # 保存原始深度数据为npy格式
            depth_path = os.path.join(target_dir, depth_filename)
            np.save(depth_path, depth_image)
            saved_files['depth'] = depth_path
            
            # 保存深度图的可视化结果
            depth_vis_filename = f'depth_vis_{timestamp}.png'
            depth_vis_path = os.path.join(target_dir, depth_vis_filename)
            depth_colormap = self.create_depth_colormap(depth_image)
            cv2.imwrite(depth_vis_path, depth_colormap)
            saved_files['depth_vis'] = depth_vis_path
            
            print(f'图片已保存到 {target_dir}:')
            print(f'- RGB图像: {color_path}')
            print(f'- 深度数据: {depth_path}')
            print(f'- 深度可视化: {depth_vis_path}')
        except Exception as e:
            print(f"保存图像失败: {e}")
        
        return saved_files
    
    def get_intrinsics(self) -> Dict[str, Any]:
        """
        获取相机内参
        
        Returns:
            Dict: 包含深度和彩色相机内参的字典
        """
        # 优先从配置文件中读取内参
        if 'intrinsics' in self.camera_config and 'depth' in self.camera_config['intrinsics'] and 'color' in self.camera_config['intrinsics']:
            # 从配置文件中读取内参
            intrinsics = {
                'color': self.camera_config['intrinsics']['color'].copy()
            }
            
            # 添加深度比例
            if 'depth_scale' in self.camera_config['intrinsics']:
                intrinsics['depth_scale'] = self.camera_config['intrinsics']['depth_scale']
            
            print("从配置文件读取相机内参")
            return intrinsics
        
        # 如果配置文件中没有内参或者内参不完整，则从相机对象中获取
        if not self.is_streaming:
            print("相机未启动，无法获取内参")
            return {}
        
        print("从相机对象读取相机内参")
        intrinsics = {
            'color': {
                'width': self.color_intrinsics.width,
                'height': self.color_intrinsics.height,
                'fx': self.color_intrinsics.fx,
                'fy': self.color_intrinsics.fy,
                'ppx': self.color_intrinsics.ppx,
                'ppy': self.color_intrinsics.ppy,
                'model': self.color_intrinsics.model,
                'coeffs': self.color_intrinsics.coeffs
            }
        }
        
        # 获取深度比例（如果可用）
        try:
            if self.is_streaming:
                depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
                intrinsics['depth_scale'] = depth_sensor.get_depth_scale()
        except Exception as e:
            print(f"获取深度比例失败: {e}，使用默认值 0.001")
            intrinsics['depth_scale'] = 0.001
        
        return intrinsics
    
    def print_intrinsics(self):
        """
        打印相机内参信息
        """
        # 获取内参（优先从配置文件读取）
        intrinsics = self.get_intrinsics()
        
        if not intrinsics:
            print("无法获取相机内参")
            return
        # 打印彩色相机内参
        print("\n彩色相机内参:")
        if 'color' in intrinsics:
            color = intrinsics['color']
            width = color.get('width', self.width)
            height = color.get('height', self.height)
            fx = color.get('fx', 0)
            fy = color.get('fy', 0)
            ppx = color.get('ppx', 0)
            ppy = color.get('ppy', 0)
            
            print(f"分辨率: {width}x{height}")
            print(f"焦距: fx={fx:.2f}, fy={fy:.2f}")
            print(f"主点: ppx={ppx:.2f}, ppy={ppy:.2f}")
        else:
            print("彩色相机内参不可用")
        
        # 打印深度比例
        if 'depth_scale' in intrinsics:
            print(f"\n深度比例: {intrinsics['depth_scale']}")
        else:
            print("\n深度比例不可用")
    
    def create_depth_colormap(self, depth_image: np.ndarray) -> np.ndarray:
        """
        创建深度图的彩色映射
        
        Args:
            depth_image: 深度图像
            
        Returns:
            np.ndarray: 彩色映射的深度图
        """
        return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    def create_combined_image(self, color_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        """
        创建彩色图和深度图的组合图像
        
        Args:
            color_image: 彩色图像
            depth_image: 深度图像
            
        Returns:
            np.ndarray: 组合图像
        """
        depth_colormap = self.create_depth_colormap(depth_image)
        
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        
        # 将RGB格式转换为BGR格式用于OpenCV显示
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image_bgr, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image_bgr, depth_colormap))
        
        return images
    
    def stop_streaming(self):
        """
        停止流式传输
        """
        if self.is_streaming and self.pipeline:
            self.pipeline.stop()
            self.is_streaming = False
            print("相机流式传输已停止")
    
    def __enter__(self):
        """上下文管理器入口"""
        if self.start_streaming():
            return self
        else:
            raise RuntimeError("无法启动相机")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_streaming()


def capture_single_frame(save_dir: str = "./data/realsense_captures") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    快速捕获单帧图像的便捷函数
    
    Args:
        save_dir: 保存目录
        
    Returns:
        Tuple[color_image, depth_image]: 彩色图像和深度图像
    """
    with RealSenseCamera(save_dir=save_dir) as camera:
        camera.print_intrinsics()
        return camera.get_frames()


def live_preview_with_save(save_dir: str = "./data/realsense_captures"):
    """
    实时预览并支持保存的便捷函数（原始脚本的功能）
    
    Args:
        save_dir: 保存目录
    """
    with RealSenseCamera(save_dir=save_dir) as camera:
        camera.print_intrinsics()
        
        try:
            while True:
                color_image, depth_image = camera.get_frames()
                
                if color_image is None or depth_image is None:
                    continue
                
                # 创建组合图像用于显示
                combined_image = camera.create_combined_image(color_image, depth_image)
                
                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', combined_image)
                
                # 键盘控制
                key = cv2.waitKey(1)
                
                # ESC键退出
                if key == 27:
                    break
                    
                # 按's'键保存图片
                elif key == ord('s'):
                    camera.save_images(color_image, depth_image)
                    
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # 运行实时预览（保持原始脚本的行为）
    # live_preview_with_save()
    # 方式1: 使用上下文管理器（推荐）
    with RealSenseCamera() as camera:
        color, depth = camera.get_frames()
        camera.print_intrinsics()
        if color is not None and depth is not None:
            camera.save_images(color, depth)

    # # 方式2: 手动管理
    # camera = RealSenseCamera(width=1280, height=720)
    # if camera.start_streaming():
    #     color, depth = camera.get_frames()
    #     intrinsics = camera.get_intrinsics()
    #     camera.stop_streaming()

    # # 方式3: 使用便捷函数
    # color, depth = capture_single_frame()
    # live_preview_with_save()  # 运行原始脚本功能