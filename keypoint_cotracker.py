import requests
import base64
import numpy as np
from PIL import Image
import io
import os
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import time
import requests

class CoTrackerClient:
    def __init__(self, base_url="http://localhost:5000", camera_config_path="./configs/camera_config.yaml"):
        """初始化CoTracker客户端
        
        Args:
            base_url (str): API服务器的基础URL
            camera_config_path (str): 相机配置文件路径
        """
        self.base_url = base_url
        self.register_url = f"{base_url}/register"
        self.track_url = f"{base_url}/track"
        self.camera_config_path = camera_config_path
        
        # 加载相机配置
        self.camera_config = self._load_camera_config()
        
        # RealSense相机配置
        self.pipeline = None
        self.config = None
        self.is_camera_initialized = False
    
    def _load_camera_config(self):
        """加载相机配置文件
        
        Returns:
            dict: 相机配置
        """
        try:
            import yaml
            with open(self.camera_config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载相机配置文件失败: {e}")
            # 返回默认配置
            return {
                'resolution': {'width': 640, 'height': 480, 'fps': 30},
                'intrinsics': {
                    'fx': 489.424683, 'fy': 489.424683,
                    'ppx': 325.761810, 'ppy': 212.508759,
                    'depth_scale': 0.001
                },
                'processing': {'resize_width': 640, 'resize_height': 480}
            }
    
    def initialize_camera(self):
        """初始化RealSense相机
        
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

            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("需要带有彩色传感器的深度相机")
                return False
            
            # 从配置文件获取分辨率和帧率
            width = self.camera_config['resolution']['width']
            height = self.camera_config['resolution']['height']
            fps = self.camera_config['resolution']['fps']
            
            print(f"使用相机配置: {width}x{height} @ {fps}fps")

            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

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
            self.is_camera_initialized = True
            print("RealSense相机初始化成功")
            return True
            
        except Exception as e:
            print(f"相机初始化失败: {e}")
            return False
    
    def capture_frame(self):
        """从RealSense相机捕获一帧图像
        
        Returns:
            tuple: (color_image, depth_image) 或 (None, None) 如果失败
        """
        if not self.is_camera_initialized:
            print("相机未初始化")
            return None, None
            
        try:
            # Wait for a coherent pair of frames: depth and color with timeout
            frames = self.pipeline.wait_for_frames(5000)  # 5秒超时
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert BGR to RGB for consistency with PIL
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"捕获帧失败: {e}")
            return None, None
    
    def stop_camera(self):
        """停止相机"""
        if self.pipeline:
            self.pipeline.stop()
            self.is_camera_initialized = False
            print("相机已停止")
    
    def img_to_base64(self, img):
        """将图像转换为base64编码字符串
        
        Args:
            img (numpy.ndarray): 输入图像
            
        Returns:
            str: base64编码的图像字符串
        """
        pil_img = Image.fromarray(img.astype(np.uint8))
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def register_frame(self, frame, keypoints):
        """注册初始帧和关键点
        
        Args:
            frame (numpy.ndarray): 初始帧图像
            keypoints (numpy.ndarray): 关键点坐标，格式为[[x, y], ...]
            
        Returns:
            dict: API响应结果
        """
        payload = {
            "frame": self.img_to_base64(frame),
            "keypoints": keypoints.tolist()
        }
        resp = requests.post(self.register_url, json=payload)
        return resp.status_code, resp.json()
    
    def track_frame(self, frame):
        """跟踪单帧图像
        
        Args:
            frame (numpy.ndarray): 要跟踪的帧图像
            
        Returns:
            dict: API响应结果
        """
        payload = {
            "frame": self.img_to_base64(frame)
        }
        resp = requests.post(self.track_url, json=payload)
        return resp.status_code, resp.json()
    
    def draw_keypoints(self, img, keypoints, color=(0, 255, 0), radius=5):
        """在图像上绘制关键点
        
        Args:
            img (numpy.ndarray): 输入图像
            keypoints (numpy.ndarray): 关键点坐标
            color (tuple): 关键点颜色 (B, G, R)
            radius (int): 关键点半径
            
        Returns:
            numpy.ndarray: 绘制了关键点的图像
        """
        img_with_keypoints = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for point in keypoints:
            x, y = point.astype(int)
            cv2.circle(img_with_keypoints, (x, y), radius, color, -1)
        return img_with_keypoints
    
    def save_frame_with_keypoints(self, img, keypoints, output_path, depth_image=None):
        """保存带有关键点的图像
        
        Args:
            img (numpy.ndarray): 输入图像
            keypoints (numpy.ndarray): 关键点坐标
            output_path (str): 输出文件路径
            depth_image (numpy.ndarray): 深度图像（可选）
        """
        img_with_keypoints = self.draw_keypoints(img, keypoints)
        cv2.imwrite(output_path, img_with_keypoints)
        
        # 如果提供了深度图像，也保存深度数据
        if depth_image is not None:
            depth_path = output_path.replace('.png', '_depth.npy').replace('.jpg', '_depth.npy')
            np.save(depth_path, depth_image)
        
        print(f'Frame saved to {output_path}')
    
    def real_time_tracking(self, save_dir="./data/cotracker_captures", 
                          interval_seconds=1.0, max_frames=None, 
                          initial_keypoints=None, show_preview=True):
        """实时跟踪和保存图像
        
        Args:
            save_dir (str): 保存目录
            interval_seconds (float): 保存间隔（秒）
            max_frames (int): 最大保存帧数，None表示无限制
            initial_keypoints (numpy.ndarray): 初始关键点
            show_preview (bool): 是否显示预览窗口
            
        Returns:
            bool: 是否成功完成
        """
        # 初始化相机
        if not self.initialize_camera():
            return False
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用默认关键点或提供的关键点
        if initial_keypoints is None:
            keypoints = np.array([[320, 200], [320, 280]], dtype=np.float32)  # 默认中心位置
        else:
            keypoints = initial_keypoints
        
        frame_count = 0
        last_save_time = 0
        is_registered = False
        
        try:
            print(f"开始实时跟踪，保存间隔: {interval_seconds}秒")
            print("按ESC键退出")
            
            while True:
                current_time = time.time()
                
                # 捕获帧
                color_image, depth_image = self.capture_frame()
                if color_image is None:
                    continue
                
                # 第一帧注册关键点
                if not is_registered:
                    status_code, response = self.register_frame(color_image, keypoints)
                    print(f"Register: {status_code}, {response}")
                    if status_code == 200:
                        is_registered = True
                        # 保存第一帧
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(save_dir, f'frame_{frame_count:04d}_{timestamp}.png')
                        self.save_frame_with_keypoints(color_image, keypoints, output_path, depth_image)
                        frame_count += 1
                        last_save_time = current_time
                else:
                    # 跟踪关键点
                    status_code, response = self.track_frame(color_image)
                    if status_code == 200:
                        tracked_keypoints = np.array(response.get('keypoints', []))
                        if len(tracked_keypoints) > 0:
                            keypoints = tracked_keypoints
                
                # 按时间间隔保存图像
                if is_registered and (current_time - last_save_time) >= interval_seconds:
                    if max_frames is None or frame_count < max_frames:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(save_dir, f'frame_{frame_count:04d}_{timestamp}.png')
                        self.save_frame_with_keypoints(color_image, keypoints, output_path, depth_image)
                        frame_count += 1
                        last_save_time = current_time
                        print(f"已保存第 {frame_count} 帧")
                    
                    if max_frames is not None and frame_count >= max_frames:
                        print(f"已达到最大帧数 {max_frames}，停止采集")
                        break
                
                # 显示预览（可选）
                if show_preview:
                    display_img = self.draw_keypoints(color_image, keypoints)
                    cv2.imshow('CoTracker Real-time', display_img)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC键
                        break
            
            print(f"总共保存了 {frame_count} 帧图像到 {save_dir}")
            return True
            
        except KeyboardInterrupt:
            print("\n用户中断")
            return True
        except Exception as e:
            print(f"实时跟踪过程中出错: {e}")
            return False
        finally:
            self.stop_camera()
            if show_preview:
                cv2.destroyAllWindows()
    
    def process_video_frames(self, frame_dir="frames_test", output_dir="output_frames", 
                           initial_keypoints=None):
        """处理视频帧序列
        
        Args:
            frame_dir (str): 输入帧目录
            output_dir (str): 输出目录
            initial_keypoints (numpy.ndarray): 初始关键点，如果为None则使用默认值
            
        Returns:
            bool: 处理是否成功
        """
        # 获取帧文件列表
        frame_files = sorted([os.path.join(frame_dir, f) 
                            for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        
        if len(frame_files) < 2:
            print("请先用 video_frame_extractor.py 生成至少两帧测试图片")
            return False
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 加载第一帧
        first_img = np.array(Image.open(frame_files[0]))
        
        # 使用默认关键点或提供的关键点
        if initial_keypoints is None:
            keypoints = np.array([[475, 177], [469, 237]], dtype=np.float32)
        else:
            keypoints = initial_keypoints
        
        # 注册初始帧
        status_code, response = self.register_frame(first_img, keypoints)
        print("Register:", status_code, response)
        
        # 保存初始帧
        initial_output_path = os.path.join(output_dir, 'frame_1.jpg')
        self.save_frame_with_keypoints(first_img, keypoints, initial_output_path)
        
        # 跟踪后续帧
        for i, fname in enumerate(frame_files[1:]):
            img = np.array(Image.open(fname))
            status_code, response = self.track_frame(img)
            print(f"Track frame {i+2}:", status_code, response)
            
            if status_code == 200:
                tracked_keypoints = np.array(response.get('keypoints', []))
                if len(tracked_keypoints) == 0:
                    print(f'Frame {i+2} 未正确跟踪，未返回 keypoints。')
                    continue
                
                # 保存带关键点的图像
                output_path = os.path.join(output_dir, f'frame_{i+2}.jpg')
                self.save_frame_with_keypoints(img, tracked_keypoints, output_path)
        
        return True

def main():
    """主函数，演示如何使用CoTrackerClient类"""
    # 创建客户端实例
    client = CoTrackerClient()
    
    # 选择模式
    print("选择模式:")
    print("1. 实时跟踪和保存 (默认)")
    print("2. 处理已有视频帧")
    
    choice = input("请输入选择 (1 或 2，默认为 1): ").strip()
    
    if choice == "2":
        # 处理视频帧
        client.process_video_frames()
    else:
        # 实时跟踪模式
        print("\n实时跟踪模式配置:")
        
        # 获取保存间隔
        try:
            interval = float(input("请输入保存间隔（秒，默认1.0）: ") or "1.0")
        except ValueError:
            interval = 1.0
        
        # 获取最大帧数
        try:
            max_frames_input = input("请输入最大保存帧数（默认无限制）: ").strip()
            max_frames = int(max_frames_input) if max_frames_input else None
        except ValueError:
            max_frames = None
        
        # 获取初始关键点（可选）
        keypoints_input = input("请输入初始关键点坐标 (格式: x1,y1,x2,y2，默认使用中心点): ").strip()
        if keypoints_input:
            try:
                coords = [float(x) for x in keypoints_input.split(',')]
                if len(coords) >= 4:
                    initial_keypoints = np.array([[coords[0], coords[1]], [coords[2], coords[3]]], dtype=np.float32)
                else:
                    initial_keypoints = None
            except ValueError:
                initial_keypoints = None
        else:
            initial_keypoints = None
        
        # 开始实时跟踪
        client.real_time_tracking(
            interval_seconds=interval,
            max_frames=max_frames,
            initial_keypoints=initial_keypoints,
            show_preview=True
        )

if __name__ == "__main__":
    main()