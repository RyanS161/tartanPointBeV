import open3d as o3d
import numpy as np
import os
from PIL import Image

def process_lidar_to_bev(input_folder, output_folder, bounding_box_size=50, image_size=100):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ply"):
            file_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".ply", ".png"))

            # Load the point cloud
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)

            # Crop to bounding box
            mask = (
                (points[:, 0] >= -bounding_box_size / 2) & (points[:, 0] <= bounding_box_size / 2) &
                (points[:, 1] >= -bounding_box_size / 2) & (points[:, 1] <= bounding_box_size / 2)
            )
            cropped_points = points[mask]

            # Project to top-down view (x, y plane)
            top_down_view = cropped_points[:, :2]

            # Normalize to image coordinates
            normalized = (top_down_view + bounding_box_size / 2) / bounding_box_size
            pixel_coords = (normalized * image_size).astype(int)
            pixel_coords = np.clip(pixel_coords, 0, image_size - 1)

            # Create binary mask
            bev_image = np.zeros((image_size, image_size), dtype=np.uint8)
            bev_image[pixel_coords[:, 1], pixel_coords[:, 0]] = 255

            # Save as PNG
            Image.fromarray(bev_image).save(output_path)
            print(f"Processed and saved: {output_path}")

# Example usage
input_folder = "/Users/ryanslocum/Documents/current_courses/PLR/data/tartanground/Downtown/Data_easy/P0006/lidar"
output_folder = "/Users/ryanslocum/Documents/current_courses/PLR/repos/tartanPointBeV/python_scripts/output/lidar"
process_lidar_to_bev(input_folder, output_folder)