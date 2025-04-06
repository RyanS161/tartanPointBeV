import numpy as np
from scipy.spatial.transform import Rotation as R
import json

"""
Reference frame conventions:
tartanAir front camera (NED):
z = fw, x = right, y = down

PointBeV base frame (ENU):
x = fw, y = left, z = up

PointBeV camera frame (front camera):
z = fw, x = right, y = down

PointBeV global frame:
x,y = ground plane, z = up
"""


###### Transform pose file from NED (TartanAir) to ENU (pointbev) ######

# Rotation to convert NED (TartanAir) to ENU (FLU) by 180° about X-axis
R_ned_to_enu = R.from_euler('x', 180, degrees=True)  # flips Y and Z

tartanair_poses_basepath = "/home/michael/Documents/Master_Sem_2/PLR/PointBeV/data/tartanground/Downtown/Data_easy/P0006"


for cam_idx in range(6):
    cam_name = f"custom{cam_idx}"
    tartanair_poses_path = f"{tartanair_poses_basepath}/pose_lcam_{cam_name}_pinhole.txt"
    print(f"Processing {cam_name}...")

    tartanair_poses = []  # list of (tx, ty, tz, qx, qy, qz, qw) from TartanAir
    try:
        with open(tartanair_poses_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 7:
                    tx, ty, tz, qx, qy, qz, qw = map(float, parts)
                    tartanair_poses.append((tx, ty, tz, qx, qy, qz, qw))
    except FileNotFoundError:
        print(f"Pose file {tartanair_poses_path} not found. Skipping...")
        continue

    converted_poses = [] # poses in FLU (pointbev) frame
    for tx, ty, tz, qx, qy, qz, qw in tartanair_poses:
        # Original TartanAir pose rotation (NED frame)
        R_ta = R.from_quat([qx, qy, qz, qw])  # quaternion in [x,y,z,w]
        # Apply 180° rotation about X to convert to FLU frame
        R_flu = R_ned_to_enu * R_ta  # first rotate coordinate axes (NED->ENU), then apply original orientation
        qx_flu, qy_flu, qz_flu, qw_flu = R_flu.as_quat()
        
        # Convert translation: (x, y, z)_NED -> (x, -y, -z)_FLU
        x_flu, y_flu, z_flu = R_ned_to_enu.apply([tx, ty, tz])  # or simply (tx, -ty, -tz)
        
        converted_poses.append((x_flu, y_flu, z_flu, qx_flu, qy_flu, qz_flu, qw_flu))
        # The pose (x_flu, y_flu, z_flu, qx_flu, qy_flu, qz_flu, qw_flu) is now in ego vehicle coordinates (FLU: x-forward, y-left, z-up)
        # It can be written to the output pose file per frame.
        
    with open(tartanair_poses_path, 'w') as f:
        for pose in converted_poses:
            tx, ty, tz, qx, qy, qz, qw = pose
            f.write(f"{tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

print("All camera poses processed.")



###### Update static camera transformations ######

def R_yaw_matrix(angle):
    R_yaw = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    return R_yaw

def clean_matrix(R, tol=1e-10):
    """Round near-zero and near-integer values"""
    return np.where(np.abs(R) < tol, 0, 
           np.where(np.abs(R-1) < tol, 1,
           np.where(np.abs(R+1) < tol, -1, R)))

camera_extrinsics = {}

# Static transformations from ego frame (NED) to camera frames:
# Ego X (forward) -> Cam Z (forward), Ego Y (left) -> Cam -X (right), Ego Z (up) -> Cam -Y (down).
R_front = np.array([
    [ 0, -1,  0],  # column 0: image of basis [1,0,0]_ego = [0,0,1]_cam (forward ego -> forward cam)
    [ 0,  0, -1],  # column 1: image of basis [0,1,0]_ego = [-1,0,0]_cam (left ego -> left in image)
    [ 1,  0,  0]   # column 2: image of basis [0,0,1]_ego = [0,-1,0]_cam (up ego -> down cam)
])

camera_angles = {
    'front': 0,
    'front_left': 55, 
    'front_right': -55,
    'back_left': 110,
    'back_right': -110,
    'back': 180
}

camera_extrinsics = {}

for position, angle in camera_angles.items():
    if angle == 0:
        camera_extrinsics[position] = R_front
    else:
        # negative because a camera facing +θ° left in ego frame requires a -θ° rotation of the ego frame into the camera frame
        angle_rad = np.radians(-angle)  
        R_yaw = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                 0,                1]
        ])
        camera_extrinsics[position] = clean_matrix(R_front @ R_yaw)


cam_position_map = dict(enumerate(['front', 'front_left', 'front_right', 'back_left', 'back_right', 'back']))

for cam_idx, position in cam_position_map.items():
    file_path = f"{tartanair_poses_basepath}/image_lcam_custom{cam_idx}_pinhole/camera_model_params_lcam_image_custom{cam_idx}_pinhole.json"
    # file_path = f"{tartanair_poses_basepath}/image_lcam_{cam_position_map[cam_idx]}_pinhole/camera_model_params_image_lcam_{cam_position_map[cam_idx]}_pinhole.json"

    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
        
        params["R_raw_new"] = [[float(val) for val in row] for row in camera_extrinsics[position]]
        print(f"Camera {cam_idx} ({position}) rotation matrix:\n{params['R_raw_new']}")
        
        with open(file_path, 'w') as f:
            json.dump(params, f, indent=4)
            
        print(f"Updated camera {cam_idx} ({position})")
    except FileNotFoundError:
        print(f"File not found: {file_path}")