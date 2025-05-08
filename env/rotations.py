import numpy as np
from scipy.spatial.transform import Rotation as R

"""
UR5 represents the orientation in axis angle representation
"""

# convert rotation vector into quaternions
def rotvec_2_quat(rotvec):
    return R.from_rotvec(rotvec).as_quat()

# convert quaternions into rotation vector
def quat_2_rotvec(quat):
    return R.from_quat(quat).as_rotvec()

# convert quaternions into euler angle (axi sequence: xyz)
def quat_2_euler(quat):
    return R.from_quat(quat).as_euler('xyz')

# convert quaternions into mrp 
def quat_2_mrp(quat):
    return R.from_quat(quat).as_mrp()

# convert euler angle into quaternions 
def euler_2_quat(euler):
    return R.from_euler(euler).as_quat()

# convert pose into quaternions, pose is [x,y,z,rx,ry,rz], pose is [x,y,z,qw,qx,qy,qz]
def pose2quat(rotvec_pose) -> np.ndarray:
    return np.concatenate((rotvec_pose[:3], rotvec_2_quat(rotvec_pose[3:])))

# convert pose into quaternions
def pose2rotvec(quat_pose) -> np.ndarray:
    return np.concatenate((quat_pose[:3], quat_2_rotvec(quat_pose[3:])))
