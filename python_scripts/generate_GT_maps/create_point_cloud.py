#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import os
import cv2
import json
from scipy.spatial.transform import Rotation
from os.path import join
from tqdm import tqdm
from configs import SEG_RGB
from train_data_tools import load_color_array

VOXEL_SIZE = 0.1  # Voxel size for downsampling
FRAME_SUBSAMPLE = 10  # Subsampling factor for frames


def collapsed_color_mapping(collapsed_groups):
    original_color_array = load_color_array(SEG_RGB)

    seg_colors = np.loadtxt(SEG_RGB, delimiter=',', dtype=np.uint8)
    segcolor_to_segid = {seg_colors[k, 2]: k for k in range(1, len(seg_colors)-1)}
    segid_to_segcolor = {k: seg_colors[k, 2] for k in range(1, len(seg_colors)-1)}

    new_color_array = np.copy(original_color_array)

    for idx, group in enumerate(collapsed_groups.values()):
        for color_idx in group:
            previous_sem_color = segid_to_segcolor[color_idx]
            new_color_array[previous_sem_color] = original_color_array[idx + 1]
            
    return new_color_array



# Camera names
CAMERAS = ["front", "back", "left", "right"]


def fast_apply_rgb_mapping(seg_map, color_array):
    """Map segmentation IDs to RGB colors efficiently using numpy indexing."""
    return color_array[seg_map]


def transform_pc_numpy(points, pose_source, pose_target):
    """
    Transform point cloud from one pose to another.

    Args:
        points: numpy array (Nx3)
        pose_source: numpy array (4x4 SE(3) matrix)
        pose_target: numpy array (4x4 SE(3) matrix)

    Returns:
        Transformed point cloud (Nx3)
    """
    points_transpose = points.T  # 3 x N
    T_rel = np.linalg.inv(pose_target) @ pose_source
    points_homogeneous = np.vstack(
        (points_transpose, np.ones((1, points.shape[0])))
    )  # Convert to homogeneous coordinates
    points_trans = T_rel @ points_homogeneous
    return points_trans[:3, :].T  # Convert back to Nx3


def depth_to_point_cloud(
    depth,
    focalx=320.0,
    focaly=320.0,
    pu=320.0,
    pv=320.0,
    filtermin=-1,
    filtermax=100,
    colorimg=None,
):
    """
    Convert depth image to point cloud.

    Args:
        depth: depth image
        colorimg: a colored image that aligns with the depth (optional)
        filtermin: minimum depth threshold
        filtermax: maximum depth threshold

    Returns:
        (points, colors) where:
        - points: Nx3 numpy array
        - colors: Nx3 numpy array (if colorimg is provided)
    """
    h, w = depth.shape
    wIdx, hIdx = np.meshgrid(
        np.arange(w) + 0.5, np.arange(h) + 0.5
    )  # Optical center at image middle

    valid_mask = (depth > filtermin) & (depth < filtermax)

    depth_filtered = depth[valid_mask]
    u = wIdx[valid_mask]
    v = hIdx[valid_mask]

    x = (u - pu) * depth_filtered / focalx
    y = (v - pv) * depth_filtered / focaly

    points = np.stack(
        [depth_filtered, x, y], axis=1
    )  # Convert to NED coordinates (Since the pose is in NED)

    colors = None
    if colorimg is not None:
        colors = colorimg[valid_mask]

    return points, colors


def pose_to_SE(pose):
    """
    Convert pose (7D) to SE(3) homogeneous matrix (4x4).
    Args:
        pose: [x, y, z, qx, qy, qz, qw]

    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


class LocalMappingRegister:
    def __init__(self, data_dir, collapsed_semantic_classes=None):
        self.data_dir = data_dir
        self.collapsed_semantic_classes = collapsed_semantic_classes
        self.color_array = load_color_array(SEG_RGB)
        if self.collapsed_semantic_classes is not None:
            self.color_array = collapsed_color_mapping(self.collapsed_semantic_classes)

    def read_depth(self, depth_path):
        """Load depth image from PNG or NPY."""
        if depth_path.endswith(".npy"):
            return np.load(depth_path)
        else:
            depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_rgba is None:
                return None
            return depth_rgba.view("<f4").squeeze()

    def read_segmentation(self, seg_path):
        """Load segmentation image from PNG or NPY."""
        if seg_path.endswith(".npy"):
            return np.load(seg_path)
        return cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

    def process_trajectory(self, traj_path):
        """Process all frames in a trajectory."""
        pcd = o3d.geometry.PointCloud()

        for cam in CAMERAS:
            pose_file = join(traj_path, f"pose_lcam_{cam}.txt")
            depth_dir = join(traj_path, f"depth_lcam_{cam}")
            seg_dir = join(traj_path, f"seg_lcam_{cam}")

            if not os.path.exists(pose_file):
                print(f"Skipping {pose_file} - File not found!")
                continue

            # Load poses
            poses = np.loadtxt(pose_file)

            # Load depth images
            depth_files = sorted(
                [f for f in os.listdir(depth_dir) if f.endswith(".png")]
            )
            # subsample to reduce memory usage
            depth_files = depth_files[::FRAME_SUBSAMPLE]
            poses = poses[::FRAME_SUBSAMPLE]

            for k, depth_filename in tqdm(
                enumerate(depth_files),
                total=len(depth_files),
                desc=f"🔹 {traj_path}/{cam}",
            ):
                depth_path = join(depth_dir, depth_filename)
                seg_path = join(seg_dir, depth_filename.replace("depth", "seg"))

                depth = self.read_depth(depth_path)
                segmentation = self.read_segmentation(seg_path)

                if depth is None or segmentation is None:
                    continue

                seg_map = (
                    fast_apply_rgb_mapping(segmentation, self.color_array).astype(
                        np.float32
                    )
                    / 255.0
                )

                points, colors = depth_to_point_cloud(depth, colorimg=seg_map)

                # Transform points to world frame
                current_pose = pose_to_SE(poses[k])
                points_transformed = transform_pc_numpy(points, current_pose, np.eye(4))

                # Convert to Open3D point cloud
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(points_transformed)
                temp_pcd.colors = o3d.utility.Vector3dVector(
                    colors
                )  # Normalized 0-1 already

                pcd += temp_pcd  # Merge point cloud
        return pcd

    def run(self):
        pcd = self.process_trajectory(self.data_dir)
        # pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        # pcd = downsample_point_cloud_with_max_pooling(pcd)
        pcd = downsample_point_cloud_with_max_pooling(pcd)

        # Visualize point cloud
        # downsampled_viewing_pcd = pcd.voxel_down_sample(voxel_size=1.0)
        # o3d.visualization.draw_geometries([downsampled_viewing_pcd])

        # Save Open3D point cloud
        pcd_filename = join(self.data_dir, f"point_cloud.pcd")
        o3d.io.write_point_cloud(pcd_filename, pcd)
        print(f"✅ Saved: {pcd_filename}")


def downsample_point_cloud_with_max_pooling(input_pcd):
    print("Creating voxel grid from point cloud...")
    # TODO: Basically need to rewrite voxel_down_sample to use max pooling
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        input_pcd,
        voxel_size=VOXEL_SIZE,
        pooling_mode=o3d.geometry.VoxelGrid.VoxelPoolingMode.MAX,
    )

    print("Creating point cloud from voxel grid...")
    voxels = voxel_grid.get_voxels()
    grid_indices = np.array([voxel.grid_index for voxel in voxels])
    colors = np.array([voxel.color for voxel in voxels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.array(grid_indices * voxel_grid.voxel_size + voxel_grid.origin + 1e-2)
    )  # +1e-2 to avoid boundary issues :/
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print("Downsampled point cloud created.")
    return pcd
