import yaml
import pyrealsense2 as rs

def initialize_realsense():
    # Load camera config
    with open('./configs/camera_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['realsense']
    
    # Initialize pipeline
    pipeline = rs.pipeline()
    rs_config = rs.config()
    
    # Find device by serial number
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
        if config['serial_number'] and dev_serial == config['serial_number']:
            print(f"  找到匹配的设备: {dev_name} (序列号: {dev_serial})")
            rs_config.enable_device(dev_serial)
            device_found = True
            break
    
    # 如果没有找到匹配的设备，使用第一个可用的设备
    if not device_found:
        if len(devices) > 0:
            first_dev = devices[0]
            dev_name = first_dev.get_info(rs.camera_info.name)
            dev_serial = first_dev.get_info(rs.camera_info.serial_number)
            print(f"  未找到序列号为 {config['serial_number']} 的设备，使用第一个可用设备: {dev_name} (序列号: {dev_serial})")
            rs_config.enable_device(dev_serial)
        else:
            print("  未找到任何相机设备")
            raise RuntimeError("未找到任何相机设备")
    
    # Configure streams
    rs_config.enable_stream(rs.stream.depth, config['resolution']['width'], 
                          config['resolution']['height'], rs.format.z16, config['fps'])
    rs_config.enable_stream(rs.stream.color, config['resolution']['width'], 
                          config['resolution']['height'], rs.format.rgb8, config['fps'])
    
    return pipeline, rs_config


def load_camera_intrinsics(self):
    # 尝试从配置文件加载相机内参
    try:
        import yaml
        with open('./configs/camera_config.yaml', 'r') as f:
            camera_config = yaml.safe_load(f)
            
        # 统一使用彩色相机内参
        color_intrinsics = camera_config['intrinsics']['color']
        depth_scale = camera_config['intrinsics']['depth_scale']
        
        # 创建与RealSense内参格式兼容的对象
        class CustomIntrinsics:
            def __init__(self, fx, fy, ppx, ppy, width, height):
                self.fx = fx
                self.fy = fy
                self.ppx = ppx
                self.ppy = ppy
                self.width = width
                self.height = height
        
        intrinsics = CustomIntrinsics(
            fx=color_intrinsics['fx'],
            fy=color_intrinsics['fy'],
            ppx=color_intrinsics['ppx'],
            ppy=color_intrinsics['ppy'],
            width=camera_config['resolution']['width'],
            height=camera_config['resolution']['height']
        )
        
        print("已从配置文件加载彩色相机内参")
        return intrinsics, depth_scale
        
    except Exception as e:
        print(f"从配置文件加载相机内参失败: {e}，尝试从相机获取内参")
        
        # 如果从配置文件加载失败，则从相机获取内参
        pipeline, config = initialize_realsense() # perception module
        
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
        
        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        intrinsics = depth_profile.get_intrinsics()

        pipeline.stop()

        return intrinsics, depth_scale
 