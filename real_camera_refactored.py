## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from typing import Tuple, Optional, Dict, Any

class RealSenseCamera:
    """RealSense相机封装类"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, save_dir: str = "./data/realsense_captures"):
        """
        初始化RealSense相机
        
        Args:
            width: 图像宽度
            height: 图像高度
            fps: 帧率
            save_dir: 保存目录
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.save_dir = save_dir
        
        # 创建保存图片的目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化相机组件
        self.pipeline = None
        self.config = None
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.is_streaming = False
        
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
            
            # Get device product line for setting a supporting resolution
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
            
            return True
            
        except Exception as e:
            print(f"启动流式传输失败: {e}")
            return False
    
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
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
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
                   timestamp: Optional[str] = None) -> Dict[str, str]:
        """
        保存图像
        
        Args:
            color_image: 彩色图像
            depth_image: 深度图像
            timestamp: 时间戳，如果为None则自动生成
            
        Returns:
            Dict[str, str]: 保存的文件路径
        """
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        try:
            # 保存RGB图像
            color_path = os.path.join(self.save_dir, 'varied_camera_raw.png')
            # 由于RealSense现在输出RGB格式，需要转换为BGR格式供OpenCV保存
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(color_path, color_image_bgr)
            saved_files['color'] = color_path
            
            # 保存原始深度数据为npy格式
            depth_path = os.path.join(self.save_dir, 'varied_camera_depth.npy')
            np.save(depth_path, depth_image)
            saved_files['depth'] = depth_path
            
            print(f'图片已保存到 {self.save_dir}:')
            print(f'- RGB图像: {color_path}')
            print(f'- 深度数据: {depth_path}')
            
        except Exception as e:
            print(f"保存图像失败: {e}")
        
        return saved_files
    
    def get_intrinsics(self) -> Dict[str, Any]:
        """
        获取相机内参
        
        Returns:
            Dict: 包含深度和彩色相机内参的字典
        """
        if not self.is_streaming:
            print("相机未启动，无法获取内参")
            return {}
        
        intrinsics = {
            'depth': {
                'width': self.depth_intrinsics.width,
                'height': self.depth_intrinsics.height,
                'fx': self.depth_intrinsics.fx,
                'fy': self.depth_intrinsics.fy,
                'ppx': self.depth_intrinsics.ppx,
                'ppy': self.depth_intrinsics.ppy,
                'model': self.depth_intrinsics.model,
                'coeffs': self.depth_intrinsics.coeffs
            },
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
        
        return intrinsics
    
    def print_intrinsics(self):
        """
        打印相机内参信息
        """
        if not self.is_streaming:
            print("相机未启动，无法打印内参")
            return
        
        print("\n深度相机内参:")
        print(f"分辨率: {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")
        print(f"焦距: fx={self.depth_intrinsics.fx:.2f}, fy={self.depth_intrinsics.fy:.2f}")
        print(f"主点: ppx={self.depth_intrinsics.ppx:.2f}, ppy={self.depth_intrinsics.ppy:.2f}")
        
        print("\n彩色相机内参:")
        print(f"分辨率: {self.color_intrinsics.width}x{self.color_intrinsics.height}")
        print(f"焦距: fx={self.color_intrinsics.fx:.2f}, fy={self.color_intrinsics.fy:.2f}")
        print(f"主点: ppx={self.color_intrinsics.ppx:.2f}, ppy={self.color_intrinsics.ppy:.2f}")
    
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