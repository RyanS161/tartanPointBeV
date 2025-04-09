import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from transforms3d.quaternions import quat2mat
import json
import cv2

### FOR PLOTS TO WORK WE HAVE TO GENERATE THE EXTRINSICS WITHOUT INVERTING IT AT THE END IN tartanair_to_pointbev_extrinsics.py ###

plot_trajectory = True
plot_static_cam_trafos = True
num_frames = 250

camera_name = "front"

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

        forward_cam = np.array([1.0, 0.0, 0.0]) # forward vector in camera coords
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


def compare_multi_camera_projection():
    cameras = ["front_left", "front", "front_right"]
    
    # pointbev / nuscenes base frame convention x=forward, y=left, z=up
    point_ego = np.array([5, 1.5, 2, 1])
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    for i, camera_name in enumerate(cameras):
        camera_extrinsics_file = f"{base_path}/image_lcam_{camera_name}_pinhole/camera_model_params_image_lcam_{camera_name}_pinhole.json"
        image_path = f"{base_path}/image_lcam_{camera_name}_pinhole/000001_image_lcam_{camera_name}_pinhole.png"
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
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
        
        ax = axes[i]
        ax.imshow(img_rgb)
        ax.set_title(f"{camera_name} View")
        
        if z <= 0:
            ax.text(10, 30, "Point behind camera", color='red', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.8))
        else:
            # Project 3D point to image
            u = fx * x / z + cx
            v = fy * y / z + cy

            print(f"R_cam: {R_cam}")
            print(f"Projected point in {camera_name}: ({u:.1f}, {v:.1f})")
            
            # Check if point is within image bounds
            h, w = img_rgb.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                ax.scatter([u], [v], c="red", s=80, marker='o')
                ax.text(u+10, v+10, f"({u:.1f}, {v:.1f})", color='red', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.7))
            else:
                ax.text(10, 30, f"Point outside frame: ({u:.1f}, {v:.1f})", 
                        color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        forward_vec = R_cam[:, 2]
        print("forward_vec:", forward_vec)
        ax.text(10, img_rgb.shape[0]-20, f"Forward: [{forward_vec[0]:.2f}, {forward_vec[1]:.2f}, {forward_vec[2]:.2f}]", 
                color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.7))
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Projection of Point ({point_ego[0]}, {point_ego[1]}, {point_ego[2]}) meters in Ego Frame", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show()

def multicam_projection_zoomed_out():
    # Define the cameras we want to compare
    cameras = ["front_left", "front", "front_right"]
    
    # pointbev / nuscenes base frame convention x=forward, y=left, z=up
    point_ego = np.array([5, 0.9, 2, 1])  # point in ego coordinates
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    for i, camera_name in enumerate(cameras):
        # Load camera params and image
        camera_extrinsics_file = f"{base_path}/image_lcam_{camera_name}_pinhole/camera_model_params_image_lcam_{camera_name}_pinhole.json"
        image_path = f"{base_path}/image_lcam_{camera_name}_pinhole/000001_image_lcam_{camera_name}_pinhole.png"
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Get camera parameters
        with open(camera_extrinsics_file, "r") as f:
            camera_params = json.load(f)
            fx = camera_params["params"]["fx"]
            fy = camera_params["params"]["fy"]
            cx = camera_params["params"]["cx"]
            cy = camera_params["params"]["cy"]
            R_cam = np.array(camera_params["R_raw_new"])
            t_cam = np.array([0, 0, 0])
        
        # Create projection matrix and transform point
        P = np.eye(4)
        P[:3, :3] = R_cam
        P[:3, 3] = t_cam
        
        point_cam = P @ point_ego
        x, y, z = point_cam[:3]
        
        # Setup the axis with extended bounds
        ax = axes[i]
        ax.imshow(img_rgb)
        
        # Set larger axis limits to show points outside the image
        margin = 400  # Extra margin to show points outside
        ax.set_xlim(-margin, w + margin)
        ax.set_ylim(h + margin, -margin)  # Remember y-axis is flipped in images
        
        # Draw image boundaries for reference
        ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], 'w--', linewidth=2)
        
        # Project and mark the point
        if z <= 0:
            ax.text(10, 30, "Point behind camera", color='red', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.8))
        else:
            # Project 3D point to image
            u = fx * x / z + cx
            v = fy * y / z + cy
            
            print(f"R_cam: {R_cam}")
            print(f"Projected point in {camera_name}: ({u:.1f}, {v:.1f})")
            
            # Always display the point, regardless of whether it's in frame
            ax.scatter([u], [v], c="red", s=100, marker='o')
            
            # Draw lines connecting the point to the image borders if outside
            if u < 0 or u >= w or v < 0 or v >= h:
                border_u = min(max(0, u), w-1)
                border_v = min(max(0, v), h-1)
                ax.plot([u, border_u], [v, border_v], 'r--', linewidth=1)
                
                ax.text(u, v-10, f"({u:.1f}, {v:.1f})", color='red', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.7))
            else:
                ax.text(u+10, v-10, f"({u:.1f}, {v:.1f})", color='red', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.7))
        
        forward_vec = R_cam[:, 2]
        print("forward_vec:", forward_vec)
        ax.text(10, img_rgb.shape[0]-20, f"Forward: [{forward_vec[0]:.2f}, {forward_vec[1]:.2f}, {forward_vec[2]:.2f}]", 
                color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.7))
        
        ax.set_title(f"{camera_name} View")
        
    plt.tight_layout()
    plt.suptitle(f"Projection of Point ({point_ego[0]}, {point_ego[1]}, {point_ego[2]}) meters in Ego Frame", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show()

def singlecam_projection():
    # pointbev / nuscenes base frame convention x=forward, y=left, z=up
    point_ego = np.array([2, 0, 0, 1])  # a point x meters in front of the ego vehicle (using pointbev base frame convention)

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

    u = fx * x / z + cx
    v = fy * y / z + cy
    print(f"Projected point: ({u}, {v})")
    print(f"Point in camera coordinates: {point_cam[:3]}")
    print(f"Point in ego coordinates: {point_ego[:3]}")

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

###### Verify the static camera transformations ######
if plot_static_cam_trafos:
    singlecam_projection()
    compare_multi_camera_projection()
    multicam_projection_zoomed_out()