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


def create_masks(input_dir, output_dirs=[], layer_nums=[]):
    # Process all images in the input directory
    for filename in tqdm(os.listdir(input_dir), desc=f"Creating Masks"):
        if filename.lower().endswith((".npy")):
            data_path = os.path.join(input_dir, filename)
            original_data = np.load(data_path)
            # Create the masks
            for out_dir, layer_num in zip(output_dirs, layer_nums):
                mask = original_data[:,:,layer_num]
                if layer_num == 0:
                    mask = np.logical_not(mask, mask) # Invert for valid mask
                # Save the mask
                np.save(os.path.join(out_dir, filename), mask)


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


def load_color_array(file_path):
    """Load segmentation ID to RGB mapping from a file."""
    mapping = {}
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            rgb_values = tuple(
                map(int, line.strip().split(","))
            )  # Convert to (R, G, B) tuple
            mapping[idx] = rgb_values  # Store in dictionary

    color_array = np.array(
        [mapping[i] for i in range(len(mapping))], dtype=np.uint8
    )
    return color_array

def create_binned_semantics(input_dir, output_dir):
    color_array = load_color_array(SEG_RGB)
    color_array = np.hstack((color_array, np.full(color_array.shape[0], 255).reshape(-1, 1)))
    dictionary = {tuple(color_array[i]): i for i in range(len(color_array))}

    # Process each .npy file in the input directory
    for file_name in tqdm(os.listdir(input_dir), desc="Creating Binned Semantic Maps"):
        if file_name.endswith(".png"):
            # Load the elevation data
            file_path = os.path.join(input_dir, file_name)

            image = Image.open(file_path).convert("RGBA")
            pixels = image.load()

            class_image = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

            for y in range(image.size[1]):
                for x in range(image.size[0]):
                    class_image[y,x] = dictionary.get(tuple(pixels[x, y]), 0)


            # one hot encode blank_image
            one_hot_encoded = np.eye(np.max(class_image) + 1, dtype=np.uint8)[class_image]

            # Visualize with a subplot for each layer
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(1, np.max(class_image) + 1, figsize=(15, 5))
            # for i in range(np.max(class_image) + 1):
            #     axs[i].imshow(one_hot_encoded[:,:,i], cmap='gray')
            #     axs[i].axis('off')
            # plt.tight_layout()
            # plt.show()



            # Save the one-hot encoded data
            output_file_path = os.path.join(output_dir, file_name.replace(".png", ".npy"))
            np.save(output_file_path, one_hot_encoded)