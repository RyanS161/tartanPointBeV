import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from transforms3d.quaternions import quat2mat
import json
import cv2

plot_trajectory = False
plot_static_cam_trafos = True
num_frames = 250

camera_name = "front_left"

base_path = "/home/michael/Documents/Master_Sem_2/PLR/PointBeV/data/tartanground/Downtown/Data_easy/P0006"
pose_file = f"{base_path}/pose_lcam_{camera_name}_pinhole.txt"
camera_extrinsics_file = f"{base_path}/image_lcam_{camera_name}_pinhole/camera_model_params_image_lcam_{camera_name}_pinhole.json"
image_path = f"{base_path}/image_lcam_{camera_name}_pinhole/000001_image_lcam_{camera_name}_pinhole.png"


###### Verify the camera trajectory and the orientation of the front camera ######
if plot_trajectory:

    xs, ys, zs = [], [], []
    dirs = []  # for forward direction vector of each camera

    with open(pose_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i >= num_frames:
            break

        parts = line.split()
        if len(parts) < 7:
            continue
        
        x, y, z = map(float, parts[0:3])
        qx, qy, qz, qw = map(float, parts[3:7])

        xs.append(x)
        ys.append(y)
        zs.append(z)

        # Convert quaternion -> rotation matrix (camera to world)
        # camera -> world. transforms3d.quaternions quat2mat expects [w, x, y, z]
        R = quat2mat([qw, qx, qy, qz])

        forward_cam = np.array([0.0, 0.0, 1.0]) # forward vector in camera coords
        forward_world = R.dot(forward_cam)

        dirs.append(forward_world)


    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(xs, ys, zs, marker='o', linestyle='-', color='blue', label='Camera Path')
    ax.scatter(xs[0], ys[0], zs[0], color='green', s=200, label='Starting Point', marker='*')

    for i in range(len(xs)):
        ax.quiver(xs[i], ys[i], zs[i],
                dirs[i][0], dirs[i][1], dirs[i][2],
                length=2.4, color='red', arrow_length_ratio=0.01)

    ax.set_title("Camera Centers + Orientation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


###### Verify the static camera transformations ######
if plot_static_cam_trafos:
    # pointbev / nuscenes base frame convention x=forward, y=left, z=up
    point_ego = np.array([4, 0, 0, 1])  # a point x meters in front of the ego vehicle (using pointbev base frame convention)

    with open(camera_extrinsics_file, "r") as f:
        camera_params = json.load(f)
        fx = camera_params["params"]["fx"]
        fy = camera_params["params"]["fy"]
        cx = camera_params["params"]["cx"]
        cy = camera_params["params"]["cy"]
        R_cam = np.array(camera_params["R_raw_new"])
        t_cam = np.array([0, 0, 0])

    P = np.eye(4)
    P[:3, :3] = R_cam
    P[:3, 3] = t_cam

    point_cam = P @ point_ego
    x, y, z = point_cam[:3]

    if z <= 0:
        print("Point is behind the camera, cannot project.")
    else:
        u = fx * x / z + cx
        v = fy * y / z + cy

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.scatter([u], [v], c="red", label="Projected point", s=60)
        plt.title("Projection of 3D ego point into camera image")
        plt.legend()
        plt.show()
