import open3d as o3d
import numpy as np
import os
from PIL import Image
from configs import *
from tqdm import tqdm



def process_lidar_to_bev(input_folder, output_folder):
    for file_name in tqdm(os.listdir(input_folder)):
        if file_name.endswith(".ply"):
            file_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".ply", ".png").replace("_lcam_front_lidar", ""))

            # Load the point cloud
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)

            # Crop to bounding box
            mask = (
                (points[:, 0] >= -BOUNDING_BOX_SIZE_M / 2) & (points[:, 0] <= BOUNDING_BOX_SIZE_M / 2) &
                (points[:, 1] >= -BOUNDING_BOX_SIZE_M / 2) & (points[:, 1] <= BOUNDING_BOX_SIZE_M / 2)
            )
            cropped_points = points[mask]

            # Project to top-down view (x, y plane)
            top_down_view = cropped_points[:, :2]

            # Normalize to image coordinates
            normalized = (top_down_view + BOUNDING_BOX_SIZE_M / 2) / BOUNDING_BOX_SIZE_M
            pixel_coords = (normalized * IMAGE_SIZE_PX).astype(int)
            pixel_coords = np.clip(pixel_coords, 0, IMAGE_SIZE_PX - 1)

            # Create binary mask
            bev_image = np.zeros((IMAGE_SIZE_PX, IMAGE_SIZE_PX), dtype=np.uint8)
            bev_image[pixel_coords[:, 1], pixel_coords[:, 0]] = 255

            # Save as PNG
            Image.fromarray(bev_image).save(output_path)

# Example usage
input_folder = os.path.join(TRAJ_ROOT, "lidar/")
output_folder = os.path.join(TRAJ_ROOT, "gt_output/lidar_bev/")
process_lidar_to_bev(input_folder, output_folder)