import os
from PIL import Image
import numpy as np
import open3d as o3d
from tqdm import tqdm
import json

from configs import *
from sklearn.cluster import KMeans


def create_binned_elev_maps(input_dir, output_dir):
    np_bins = np.linspace(-MAX_Z_VALUE_M, MAX_Z_VALUE_M, NUM_Z_BINS)
    # Process each .npy file in the input directory
    for file_name in tqdm(os.listdir(input_dir), desc="Creating Binned Elevation Maps"):
        if file_name.endswith(".npy"):
            # Load the elevation data
            file_path = os.path.join(input_dir, file_name)
            elevation_data = np.load(file_path)
            # Create the bins and one-hot encode
            binned_data = np.digitize(elevation_data, bins=np_bins) - 1
            one_hot_encoded = np.eye(NUM_Z_BINS, dtype=np.uint8)[binned_data]

            # Save the one-hot encoded data
            output_file_path = os.path.join(output_dir, file_name)
            np.save(output_file_path, one_hot_encoded)


def binary_mask_helper(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGBA")
    pixels = image.load()

    # Create blank images for the masks
    color_mask = Image.new("1", image.size)  # Binary mask for the target color
    transparent_mask = Image.new("1", image.size)  # Binary mask for transparency

    color_mask_pixels = color_mask.load()
    transparent_mask_pixels = transparent_mask.load()
    # Get all unique colors in the image
    # unique_colors = set()
    # for y in range(image.size[1]):
    #     for x in range(image.size[0]):
    #         unique_colors.add(pixels[x, y])
    # unique_colors = unique_colors
    # Iterate through each pixel
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            r, g, b, a = pixels[x, y]

            # Check for the target color
            color_mask_pixels[x, y] = 1 if (r, g, b) == CAR_RGB else 0

            # Check for transparency
            transparent_mask_pixels[x, y] = 0 if a == 0 else 1

    return color_mask, transparent_mask


def create_car_masks(input_dir, color_mask_dir, valid_mask_dir):
    # Process all images in the input directory
    for filename in tqdm(os.listdir(input_dir), desc="Creating Car Masks"):
        if filename.lower().endswith((".png")):
            image_path = os.path.join(input_dir, filename)

            # Create the binary masks
            color_mask, transparent_mask = binary_mask_helper(image_path)

            # Save the masks
            color_mask.save(
                os.path.join(color_mask_dir, f"{os.path.splitext(filename)[0]}.png")
            )
            transparent_mask.save(
                os.path.join(valid_mask_dir, f"{os.path.splitext(filename)[0]}.png")
            )


def process_lidar_to_bev(input_folder, output_folder):
    for file_name in tqdm(os.listdir(input_folder), desc="Processing LIDAR to BEV"):
        if file_name.endswith(".ply"):
            file_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(
                output_folder,
                file_name.replace(".ply", ".png").replace("_lcam_front_lidar", ""),
            )

            # Load the point cloud
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)

            # Crop to bounding box
            mask = (
                (points[:, 0] >= -BOUNDING_BOX_SIZE_M / 2)
                & (points[:, 0] <= BOUNDING_BOX_SIZE_M / 2)
                & (points[:, 1] >= -BOUNDING_BOX_SIZE_M / 2)
                & (points[:, 1] <= BOUNDING_BOX_SIZE_M / 2)
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
            bev_image = np.flipud(bev_image)
            # Save as PNG
            Image.fromarray(bev_image).save(output_path)

def class_to_num_grouping(json_file, preferred_clustering):
    class_to_num_mapping = {}
    with open(json_file, 'r') as f:
        original_semantics = json.load(f)['name_map']
        for new_class, old_classes in preferred_clustering.items():
            for old_class in old_classes:
                if new_class not in class_to_num_mapping:
                    class_to_num_mapping[new_class] = []
                class_to_num_mapping[new_class].append(original_semantics[old_class])
            
    
    return class_to_num_mapping