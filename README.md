
# ReKep-Reproduction-PiPER
这个项目是在松灵机器人的PiPER机械臂上复现[Rekep](https://rekep-robot.github.io/)。

## 安装
创建环境：
```bash
conda create -n rekep-piper python=3.10
conda activate rekep-piper
cd ./Rekep-Reproduction-PiPER
pip install -r requirements.txt
sudo apt update && sudo apt install can-utils ethtool
```
根据PiPER官方说明安装piper_sdk
需要先完成手眼标定[PiPER手眼标定教程](./docs/hand-eye-calib.md)


## 进展
- [x] 实现PiPER控制器
- [ ] 实机测试PiPER控制器
- [ ] 增加点跟踪部分[cotracker-rekep-api](https://github.com/tailong-wu/cotracker-rekep-api)


## 文件说明
+ [real_camera.py](./real_camera.py): 获取realsense相机出厂默认内参，并拍摄图像


metadata.json 中保存的是初始关键点三维位置（相机坐标）[x,y,z]（注意：基于realsense光学坐标系）、初始关键点二维图像位置[y,x]（与 OpenCV 的像素坐标约定一致。）、以及其他必要信息。


## 参考代码仓库
https://github.com/heyjiacheng/Rekep-ur5/tree/main

https://github.com/agilexrobotics/piper_sdk/tree/master
