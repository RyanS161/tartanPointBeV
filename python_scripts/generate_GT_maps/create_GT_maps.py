import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from collections import defaultdict
from configs import *

VISUALIZE_VOXEL_SIZE = 0.5


def pose_to_SE(pose):
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    return T


def save_elevation_data(elevation_data, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    np.save(file_path, elevation_data)


def save_semantic_plot(elevation_data, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    plt.imsave(file_path, elevation_data)


class GroundTruthMapGenerator:
    def __init__(self, traj_path, grid_resolution=1.0):
        self.traj_path = traj_path
        self.pc_path = os.path.join(traj_path, "point_cloud.pcd")
        self.grid_resolution = grid_resolution
        self.load_data()

    def load_data(self):
        print(f"Loading point cloud from {self.pc_path}")
        self.point_cloud = o3d.io.read_point_cloud(self.pc_path)
        assert (
            self.point_cloud.dimension() == 3
            and self.point_cloud.has_points()
            and self.point_cloud.has_colors()
        )

        # extract all points
        self.points = np.asarray(self.point_cloud.points).astype(np.float32)
        self.colors = (np.asarray(self.point_cloud.colors) * 255.0).astype(np.uint8)
        self.colors = np.hstack(
            (self.colors, np.full((self.colors.shape[0], 1), 255))
        ).astype(np.uint8)

        # transform all points
        self.homogenous_points = np.hstack(
            [self.points, np.ones((self.points.shape[0], 1))]
        ).T.astype(np.float32)

        self.grid_min_bound = [
            -np.ceil(BOUNDING_BOX_SIZE_M / 2),
            -np.ceil(BOUNDING_BOX_SIZE_M / 2),
            -MAX_Z_VALUE_M,
        ]
        self.grid_max_bound = [
            np.ceil(BOUNDING_BOX_SIZE_M / 2),
            np.ceil(BOUNDING_BOX_SIZE_M / 2),
            MAX_Z_VALUE_M,
        ]

    def transform_and_clean_point_cloud(self, transformation):
        # print("Cleaning point cloud data")

        # self.point_cloud = self.point_cloud.remove_duplicated_points()
        # self.point_cloud = self.point_cloud.remove_non_finite_points()
        # self.point_cloud, _ = self.point_cloud.remove_radius_outlier(
        #     nb_points=5, radius=1.0, print_progress=True
        # )

        # Crop point cloud to specified bounds
        # Copy point cloud to new_pc
        new_pc = o3d.geometry.PointCloud(self.point_cloud)
        # Apply transformation
        new_pc.transform(transformation)
        # Crop point cloud to specified bounds
        bb_size = np.ceil(BOUNDING_BOX_SIZE_M / 2)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[-bb_size, -bb_size, -MAX_Z_VALUE_M],
            max_bound=[bb_size, bb_size, MAX_Z_VALUE_M],
        )
        new_pc = new_pc.crop(bbox)
        min_bound = new_pc.get_min_bound()
        # Remove points with elevation < 0
        points = np.asarray(new_pc.points)
        colors = np.asarray(new_pc.colors)
        # inverting height to be z-up
        points[:, 2] *= -1.0
        return points, colors, min_bound

    # @line_profiler.profile
    def transform_point_cloud_with_numpy(self, transformation):
        transformed_points = np.dot(
            transformation.astype(np.float32), self.homogenous_points
        ).T
        transformed_points = transformed_points[:, :3]

        # remove according to bounding box
        mask = np.all(
            (transformed_points >= self.grid_min_bound)
            & (transformed_points <= self.grid_max_bound),
            axis=1,
        )
        filtered_points = transformed_points[mask]
        colors = self.colors[mask]
        # inverting height to be z-up
        filtered_points[:, 2] *= -1.0

        return filtered_points, colors

    def visualize_point_cloud(self, point_cloud, vis_voxel_size):
        vis_pc = point_cloud.voxel_down_sample(voxel_size=vis_voxel_size)
        o3d.visualization.draw_geometries([vis_pc])

    # @line_profiler.profile
    def create_maps(self):
        print("Creating 2.5D maps")
        # downsampled_pc = self.point_cloud.voxel_down_sample(
        #     voxel_size=(self.grid_resolution / 2)
        # )
        pose_file = os.path.join(self.traj_path, "pose_lcam_front.txt")
        poses = np.loadtxt(pose_file)

        for frame_idx, pose in tqdm(enumerate(poses)):
            current_pose = pose_to_SE(pose)
            transformation = np.linalg.inv(current_pose)
            points, colors = self.transform_point_cloud_with_numpy(transformation)
            # classes = np.array([float_color_to_seg_color(color) for color in colors])
            grid_coords = np.floor(
                (points[:, :2] - self.grid_min_bound[:2])
                / (BOUNDING_BOX_SIZE_M / IMAGE_SIZE_PX)
            ).astype(int)
            grid_coords[:, 0] = (
                IMAGE_SIZE_PX - 1 - grid_coords[:, 0]
            )  # Flip y-axis to show front camera towards top of image
            unique_grid_coords, inv_indices = np.unique(
                grid_coords, axis=0, return_inverse=True
            )
            elevation_layers = np.full((IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3), np.nan)
            semantic_layers = np.full(
                (IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3, 4), 0, dtype=np.uint8
            )
            # semantic_points = []

            grid_to_point_indices = defaultdict(list)
            for idx, grid_cell in enumerate(inv_indices):
                grid_to_point_indices[grid_cell].append(idx)

            for i in range(len(unique_grid_coords)):
                pillar_points = points[grid_to_point_indices[i]]
                if len(pillar_points) > MIN_ELEV_TUNING_FACTOR:
                    x, y = unique_grid_coords[i]
                    if x < 0 or x >= IMAGE_SIZE_PX or y < 0 or y >= IMAGE_SIZE_PX:
                        print(f"Skipping out-of-bounds grid cell: ({x}, {y})")
                        continue
                    pillar_colors = colors[grid_to_point_indices[i]]
                    sorted_indices = np.argsort(pillar_points[:, 2])
                    sorted_z_values = pillar_points[sorted_indices, 2]
                    sorted_colors = pillar_colors[sorted_indices]
                    # Min ground heuristic
                    elevation_layers[x, y, 0] = np.mean(
                        sorted_z_values[:MIN_ELEV_TUNING_FACTOR]
                    )
                    semantic_layers[x, y, 0] = sorted_colors[0]
                    # Max ground and ceiling heuristics
                    non_min_points = sorted_z_values[MIN_ELEV_TUNING_FACTOR:]
                    non_min_colors = sorted_colors[MIN_ELEV_TUNING_FACTOR:]
                    gaps = np.diff(non_min_points)
                    gap_index = np.where(gaps > DESIRED_CEILING_GAP)[0]
                    if gap_index.size > 0:
                        first_gap_index = gap_index[0]
                        elevation_layers[x, y, 1] = non_min_points[
                            first_gap_index
                        ]  # Max ground
                        semantic_layers[x, y, 1] = non_min_colors[first_gap_index]
                        elevation_layers[x, y, 2] = non_min_points[first_gap_index + 1]
                        semantic_layers[x, y, 2] = non_min_colors[first_gap_index + 1]
                    else:
                        elevation_layers[x, y, 1] = non_min_points[-1]
                        semantic_layers[x, y, 1] = non_min_colors[-1]

            # Plot each elevation map
            save_elevation_data(
                elevation_layers[:, :, 0],
                os.path.join(
                    self.traj_path, f"gt_output/elev/min_ground/{frame_idx:06d}.npy"
                ),
            )
            save_elevation_data(
                elevation_layers[:, :, 1],
                os.path.join(
                    self.traj_path, f"gt_output/elev/max_ground/{frame_idx:06d}.npy"
                ),
            )
            save_elevation_data(
                elevation_layers[:, :, 2],
                os.path.join(
                    self.traj_path, f"gt_output/elev/ceiling/{frame_idx:06d}.npy"
                ),
            )

            # Plot each semantic map
            save_semantic_plot(
                semantic_layers[:, :, 0],
                os.path.join(
                    self.traj_path, f"gt_output/sem/min_ground/{frame_idx:06d}.png"
                ),
            )
            save_semantic_plot(
                semantic_layers[:, :, 1],
                os.path.join(
                    self.traj_path, f"gt_output/sem/max_ground/{frame_idx:06d}.png"
                ),
            )
            save_semantic_plot(
                semantic_layers[:, :, 2],
                os.path.join(
                    self.traj_path, f"gt_output/sem/ceiling/{frame_idx:06d}.png"
                ),
            )
