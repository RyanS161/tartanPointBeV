import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from scipy.spatial.transform import Rotation

VISUALIZE = True
VISUALIZE_VOXEL_SIZE = 0.5
GRID_RESOLUTION = 0.5
GRID_SIZE = 50  # 25 meters -> 50 meters by 50 meters grid
GRID_SIZE_PIXELS = int(GRID_SIZE / GRID_RESOLUTION)
MAX_Z_VALUE = 15.0  # 15 meters, we probably don't care about anything higher
MIN_ELEV_TUNING_FACTOR = 2
DESIRED_CEILING_GAP = 2.0

SEG_RGBS_PATH = "/Users/ryanslocum/Documents/current_courses/PLR/repos/misc/files_from_manthan/seg_rgbs.txt"


def pose_to_SE(pose):
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    return T

def save_elevation_data(elevation_data, file_path):
    np.save(file_path, elevation_data)

def save_semantic_plot(elevation_data, file_path):
    plt.imsave(file_path, elevation_data)

def load_rgb_mapping(file_path):
    """Load segmentation ID to RGB mapping from a file."""
    colors = []
    with open(file_path, "r") as f:
        for line in f:
            rgb_values = tuple(
                map(int, line.strip().split(","))
            )  # Convert to (R, G, B) tuple
            colors.append(rgb_values)
    return colors


COLORS_ARR = load_rgb_mapping(SEG_RGBS_PATH)


def float_color_to_seg_color(rgb):
    round_then_int_x = lambda x: int(np.round(x))
    int_bgr_tuple = tuple(map(round_then_int_x, rgb * 255))[::-1]
    try:
        idx = COLORS_ARR.index(int_bgr_tuple)
        return idx
    except ValueError:
        print("Color not found in mapping")
        return np.nan


class GroundTruthMapGenerator:
    def __init__(self, pc_path, grid_resolution=1.0):
        self.pc_path = pc_path
        self.parent_dir = os.path.dirname(pc_path)
        self.grid_resolution = grid_resolution
        self.load_data()
        # self.visualize_point_cloud(vis_voxel_size=VISUALIZE_VOXEL_SIZE)
        # self.clean_data()
        # self.visualize_point_cloud(vis_voxel_size=VISUALIZE_VOXEL_SIZE)

    def load_data(self):
        print(f"Loading point cloud from {self.pc_path}")
        self.point_cloud = o3d.io.read_point_cloud(self.pc_path)
        assert (
            self.point_cloud.dimension() == 3
            and self.point_cloud.has_points()
            and self.point_cloud.has_colors()
        )

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
        BBOX_SIZE = np.ceil(GRID_SIZE/2)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[-BBOX_SIZE, -BBOX_SIZE, -MAX_Z_VALUE],
            max_bound=[BBOX_SIZE, BBOX_SIZE, MAX_Z_VALUE],
        )
        new_pc = new_pc.crop(bbox)
        min_bound = new_pc.get_min_bound()
        # Remove points with elevation < 0
        points = np.asarray(new_pc.points)
        colors = np.asarray(new_pc.colors)
        # inverting height to be z-up because it really fucks with my head otherwise
        points[:, 2] *= -1.0
        # remove invalid seg colors
        # clean points
        return points, colors, min_bound

    def visualize_point_cloud(self, point_cloud, vis_voxel_size):
        vis_pc = point_cloud.voxel_down_sample(voxel_size=vis_voxel_size)
        o3d.visualization.draw_geometries([vis_pc])

    def create_maps(self):
        print("Creating 2.5D maps")
        # downsampled_pc = self.point_cloud.voxel_down_sample(
        #     voxel_size=(self.grid_resolution / 2)
        # )
        pose_file = os.path.join(self.parent_dir, "pose_lcam_front.txt")
        poses = np.loadtxt(pose_file)

        self.point_cloud = self.point_cloud.voxel_down_sample(
            voxel_size=self.grid_resolution/2
        )

        for frame_idx, pose in tqdm(enumerate(poses)):
            current_pose = pose_to_SE(pose)
            transformation = np.linalg.inv(current_pose)
            points, colors, min_bounds = self.transform_and_clean_point_cloud(transformation)
            # classes = np.array([float_color_to_seg_color(color) for color in colors])
            grid_coords = np.floor(
                (points[:, :2] - min_bounds[:2]) / self.grid_resolution
            ).astype(int)
            unique_grid_coords, inv_indices = np.unique(
                grid_coords, axis=0, return_inverse=True
            )
            elevation_layers = np.full((GRID_SIZE_PIXELS, GRID_SIZE_PIXELS, 3), np.nan)
            semantic_layers = np.full((GRID_SIZE_PIXELS, GRID_SIZE_PIXELS, 3, 3), np.nan)
            # semantic_points = []

            for i in range(len(unique_grid_coords)):
                x, y = unique_grid_coords[i]
                pillar_points = points[inv_indices == i]
                if len(pillar_points) > MIN_ELEV_TUNING_FACTOR:
                    pillar_colors = colors[inv_indices == i]
                    sorted_indices = np.argsort(pillar_points[:, 2])
                    sorted_z_values = pillar_points[sorted_indices, 2]
                    sorted_colors = pillar_colors[sorted_indices]
                    # Min ground heuristic
                    elevation_layers[x,y,0] = np.mean(
                        sorted_z_values[:MIN_ELEV_TUNING_FACTOR]
                    )
                    semantic_layers[x,y,0] = sorted_colors[0]
                    # Max ground and ceiling heuristics
                    non_min_points = sorted_z_values[MIN_ELEV_TUNING_FACTOR:]
                    non_min_colors = sorted_colors[MIN_ELEV_TUNING_FACTOR:]
                    gaps = np.diff(non_min_points)
                    gap_index = np.where(gaps > DESIRED_CEILING_GAP)[0]
                    if gap_index.size > 0:
                        first_gap_index = gap_index[0]
                        elevation_layers[x,y,1] = non_min_points[first_gap_index] # Max ground
                        semantic_layers[x,y,1] = non_min_colors[first_gap_index]
                        elevation_layers[x,y,2] = non_min_points[first_gap_index + 1]
                        semantic_layers[x,y,2] = non_min_colors[first_gap_index + 1]
                    else:
                        elevation_layers[x,y,1] = non_min_points[-1]
                        semantic_layers[x,y,1] = non_min_colors[-1]

            if VISUALIZE:
                # Plot each elevation map
                save_elevation_data(elevation_layers[:, :, 0], f"./output/elev/min_ground/{frame_idx:06d}.npy")
                save_elevation_data(elevation_layers[:, :, 1], f"./output/elev/max_ground/{frame_idx:06d}.npy")
                save_elevation_data(elevation_layers[:, :, 2], f"./output/elev/ceiling/{frame_idx:06d}.npy")

                # Plot each semantic map
                save_semantic_plot(semantic_layers[:, :, 0], f"./output/sem/min_ground/{frame_idx:06d}.png")
                save_semantic_plot(semantic_layers[:, :, 1], f"./output/sem/max_ground/{frame_idx:06d}.png")
                save_semantic_plot(semantic_layers[:, :, 2], f"./output/sem/ceiling/{frame_idx:06d}.png")

                #TODO: Add mask
                # Make save as raw images
                # Make lidar scan images with same format


if __name__ == "__main__":
    PC_PATH = "/Users/ryanslocum/Documents/current_courses/PLR/data/tartanground/Downtown/Data_easy/P0006/point_cloud.pcd"
    generator = GroundTruthMapGenerator(PC_PATH, grid_resolution=GRID_RESOLUTION)
    generator.create_maps()
