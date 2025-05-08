## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# 创建保存图片的目录
save_dir = "./data/realsense_captures"
os.makedirs(save_dir, exist_ok=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# 获取相机内参
profile = pipeline.get_active_profile()

# 获取深度流内参
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

# 获取彩色流内参
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()

# 打印内参信息
print("\n深度相机内参:")
print(f"分辨率: {depth_intrinsics.width}x{depth_intrinsics.height}")
print(f"焦距: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")
print(f"主点: ppx={depth_intrinsics.ppx:.2f}, ppy={depth_intrinsics.ppy:.2f}")

print("\n彩色相机内参:")
print(f"分辨率: {color_intrinsics.width}x{color_intrinsics.height}")
print(f"焦距: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
print(f"主点: ppx={color_intrinsics.ppx:.2f}, ppy={color_intrinsics.ppy:.2f}")

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        
        # 键盘控制
        key = cv2.waitKey(1)
        
        # ESC键退出
        if key == 27:
            break
            
        # 按's'键保存图片
        elif key == ord('s'):
            # 生成时间戳作为文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 保存RGB图像
            color_path = os.path.join(save_dir, 'varied_camera_raw.png')
            cv2.imwrite(color_path, color_image)
            
            # 保存原始深度数据为npy格式
            depth_path = os.path.join(save_dir, 'varied_camera_depth.npy')
            np.save(depth_path, depth_image)  # 保存原始深度数据
            
            # 保存深度可视化图像（用于查看）
            # depth_vis_path = os.path.join(save_dir, f'depth_vis_{timestamp}.png')
            # cv2.imwrite(depth_vis_path, depth_colormap)
            
            # 保存组合图像
            # combined_path = os.path.join(save_dir, f'combined_{timestamp}.png')
            # cv2.imwrite(combined_path, images)
            
            # 保存内参信息
            # intrinsics_dir = os.path.join(save_dir, "intrinsics")
            # os.makedirs(intrinsics_dir, exist_ok=True)

            # 保存深度内参
            # depth_intrin_path = os.path.join(intrinsics_dir, f'depth_intrin_{timestamp}.txt')
            # with open(depth_intrin_path, 'w') as f:
            #     f.write(f"Width: {depth_intrinsics.width}\n")
            #     f.write(f"Height: {depth_intrinsics.height}\n")
            #     f.write(f"fx: {depth_intrinsics.fx}\n")
            #     f.write(f"fy: {depth_intrinsics.fy}\n")
            #     f.write(f"ppx: {depth_intrinsics.ppx}\n")
            #     f.write(f"ppy: {depth_intrinsics.ppy}\n")
            #     f.write(f"Distortion Model: {depth_intrinsics.model}\n")
            #     f.write(f"Distortion Coefficients: {depth_intrinsics.coeffs}\n")

            # 保存彩色内参
            # color_intrin_path = os.path.join(intrinsics_dir, f'color_intrin_{timestamp}.txt')
            # with open(color_intrin_path, 'w') as f:
            #     f.write(f"Width: {color_intrinsics.width}\n")
            #     f.write(f"Height: {color_intrinsics.height}\n")
            #     f.write(f"fx: {color_intrinsics.fx}\n")
            #     f.write(f"fy: {color_intrinsics.fy}\n")
            #     f.write(f"ppx: {color_intrinsics.ppx}\n")
            #     f.write(f"ppy: {color_intrinsics.ppy}\n")
            #     f.write(f"Distortion Model: {color_intrinsics.model}\n")
            #     f.write(f"Distortion Coefficients: {color_intrinsics.coeffs}\n")

            print(f'图片已保存到 {save_dir}:')
            print(f'- RGB图像: color_{timestamp}.png')
            print(f'- 深度数据: depth_{timestamp}.npy')
            # print(f'- 深度可视化图像: depth_vis_{timestamp}.png')
            # print(f'- 组合图像: combined_{timestamp}.png')
            # print(f'- 深度内参文件: depth_intrin_{timestamp}.txt')
            # print(f'- 彩色内参文件: color_intrin_{timestamp}.txt')

finally:
    # Stop streaming
    pipeline.stop()
