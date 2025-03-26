import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

VISUALIZE = True
VISUALIZE_VOXEL_SIZE = 0.5
GRID_RESOLUTION = 1.0

SEG_RGBS_PATH = "/Users/ryanslocum/Documents/current_courses/PLR/repos/misc/files_from_manthan/seg_rgbs.txt"


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
        self.clean_data()
        # self.visualize_point_cloud(vis_voxel_size=VISUALIZE_VOXEL_SIZE)

    def load_data(self):
        print(f"Loading point cloud from {self.pc_path}")
        self.point_cloud = o3d.io.read_point_cloud(self.pc_path)
        assert (
            self.point_cloud.dimension() == 3
            and self.point_cloud.has_points()
            and self.point_cloud.has_colors()
        )

    def clean_data(self):
        print("Cleaning point cloud data")

        # self.point_cloud = self.point_cloud.remove_duplicated_points()
        # self.point_cloud = self.point_cloud.remove_non_finite_points()
        # self.point_cloud, _ = self.point_cloud.remove_radius_outlier(
        #     nb_points=5, radius=1.0, print_progress=True
        # )

        # Crop point cloud to specified bounds
        # bbox = o3d.geometry.AxisAlignedBoundingBox(
        #     min_bound=[-100, -100, -100], max_bound=[100, 100, 100]
        # )
        # self.point_cloud = self.point_cloud.crop(bbox)

        # Remove points with elevation < 0
        points = np.asarray(self.point_cloud.points)
        colors = np.asarray(self.point_cloud.colors)
        # inverting height to be z-up because it really fucks with my head otherwise
        points[:, 2] *= -1.0
        # remove invalid seg colors
        # clean points
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

    def visualize_point_cloud(self, vis_voxel_size):
        vis_pc = self.point_cloud.voxel_down_sample(voxel_size=vis_voxel_size)
        o3d.visualization.draw_geometries([vis_pc])

    def create_maps(self):
        # PC_DOWNSAMPLE_RATE = 2
        MIN_ELEV_TUNING_FACTOR = 10
        DESIRED_CEILING_GAP = (
            2.0  # https://www.anybotics.com/anymal-technical-specifications.pdf
        )
        print("Creating 2.5D maps")
        # downsampled_pc = self.point_cloud.voxel_down_sample(
        #     voxel_size=(self.grid_resolution / 2)
        # )
        downsampled_pc = self.point_cloud
        points = np.asarray(downsampled_pc.points)
        colors = np.asarray(downsampled_pc.colors)
        # classes = np.array([float_color_to_seg_color(color) for color in colors])
        min_bounds = downsampled_pc.get_min_bound()
        grid_coords = np.floor(
            (points[:, :2] - min_bounds[:2]) / self.grid_resolution
        ).astype(int)
        unique_grid_coords, inv_indices = np.unique(
            grid_coords, axis=0, return_inverse=True
        )
        min_ground_layer_elev, max_ground_layer_elev, ceiling_layer_elev= [], [], []
        min_ground_layer_sem, max_ground_layer_sem, ceiling_layer_sem = [], [], []
        # semantic_points = []
        for i in tqdm(range(len(unique_grid_coords))):
            pillar_points = points[inv_indices == i]
            pillar_colors = colors[inv_indices == i]
            grid_x = unique_grid_coords[i, 0] * self.grid_resolution + min_bounds[0]
            grid_y = unique_grid_coords[i, 1] * self.grid_resolution + min_bounds[1]

            sorted_z_values = np.sort(pillar_points[:, 2])
            sorted_colors = pillar_colors[np.argsort(pillar_points[:, 2])]
            if len(sorted_z_values) > MIN_ELEV_TUNING_FACTOR:
                # Min ground heuristic
                min_ground_elevation = np.mean(sorted_z_values[:MIN_ELEV_TUNING_FACTOR])
                min_ground_sem = sorted_colors[0]
                # Max ground and ceiling heuristics
                non_min_points = sorted_z_values[MIN_ELEV_TUNING_FACTOR:]
                non_min_colors = sorted_colors[MIN_ELEV_TUNING_FACTOR:]
                gaps = np.diff(non_min_points)
                gap_index = np.where(gaps > DESIRED_CEILING_GAP)[0]
                if gap_index.size > 0:
                    first_gap_index = gap_index[0]
                    max_ground_elevation = non_min_points[first_gap_index]
                    max_ground_sem = non_min_colors[first_gap_index]
                    ceiling_elevation = non_min_points[first_gap_index + 1]
                    ceiling_sem = non_min_colors[first_gap_index + 1]
                else:
                    max_ground_elevation = non_min_points[-1]
                    max_ground_sem = non_min_colors[-1]
                    ceiling_elevation = np.nan
                    ceiling_sem = (np.nan,)*3

            else:
                min_ground_elevation, max_ground_elevation, ceiling_elevation = np.nan, np.nan, np.nan
                min_ground_sem, max_ground_sem, ceiling_sem = (np.nan,)*3, (np.nan,)*3, (np.nan,)*3

            min_ground_layer_elev.append([grid_x, grid_y, min_ground_elevation])
            max_ground_layer_elev.append([grid_x, grid_y, max_ground_elevation])
            ceiling_layer_elev.append([grid_x, grid_y, ceiling_elevation])

            min_ground_layer_sem.append([grid_x, grid_y, *min_ground_sem])
            max_ground_layer_sem.append([grid_x, grid_y, *max_ground_sem])
            ceiling_layer_sem.append([grid_x, grid_y, *ceiling_sem])
            # pillar_classes = classes[inv_indices == i]
            # semantic_points.append(
            #     [grid_x, grid_y, pillar_classes[np.bincount(pillar_classes.astype(int)).argmax()]]
            # )
        min_ground_layer_elev = np.array(min_ground_layer_elev)
        max_ground_layer_elev = np.array(max_ground_layer_elev)
        ceiling_layer_elev = np.array(ceiling_layer_elev)

        min_ground_layer_sem = np.array(min_ground_layer_sem)
        max_ground_layer_sem = np.array(max_ground_layer_sem)
        ceiling_layer_sem = np.array(ceiling_layer_sem)
        # semantic_points = np.array(semantic_points)

        if VISUALIZE:
            # Create a figure with 3 subplots for the three elevation maps
            fig, axes = plt.subplots(2, 3, figsize=(18, 6))

            # Function to plot elevation data on a given axis
            def plot_elevation_map(ax, elevation_data, title):
                grid_x = np.unique(elevation_data[:, 0])
                grid_y = np.unique(elevation_data[:, 1])
                elev_matrix = np.full((len(grid_y), len(grid_x)), np.nan)
                for point in elevation_data:
                    x_idx = np.where(grid_x == point[0])[0][0]
                    y_idx = np.where(grid_y == point[1])[0][0]
                    elev_matrix[y_idx, x_idx] = point[2]
                im = ax.imshow(
                    elev_matrix,
                    extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
                    origin="lower",
                    cmap="viridis",
                    vmax=10.0,  # Set maximum elevation to 10 meters
                )
                fig.colorbar(im, ax=ax, label="Elevation")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(title)
            
            def plot_color_image(ax, color_data, title):
                grid_x = np.unique(color_data[:, 0])
                grid_y = np.unique(color_data[:, 1])
                color_matrix = np.full((len(grid_y), len(grid_x), 3), np.nan)
                for point in color_data:
                    x_idx = np.where(grid_x == point[0])[0][0]
                    y_idx = np.where(grid_y == point[1])[0][0]
                    color_matrix[y_idx, x_idx] = point[2:]
                ax.imshow(
                    color_matrix,
                    extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
                    origin="lower",
                )
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(title)

            # Plot each elevation map
            plot_elevation_map(axes[0][0], min_ground_layer_elev, "Minimum Ground Elevation")
            plot_elevation_map(axes[0][1], max_ground_layer_elev, "Maximum Ground Elevation")
            plot_elevation_map(axes[0][2], ceiling_layer_elev, "Ceiling Elevation")
            plot_color_image(axes[1][0], min_ground_layer_sem, "Minimum Ground Semantic")
            plot_color_image(axes[1][1], max_ground_layer_sem, "Maximum Ground Semantic")
            plot_color_image(axes[1][2], ceiling_layer_sem, "Ceiling Semantic")

            plt.tight_layout()
            # save plt plot to file
            plt.savefig("elevation_semantic_maps.png")

            # pickle the relevant data in one object
            with open("elevation_semantic_maps.pkl", "wb") as f:
                pickle.dump(
                    {
                        "min_ground_layer_elev": min_ground_layer_elev,
                        "max_ground_layer_elev": max_ground_layer_elev,
                        "ceiling_layer_elev": ceiling_layer_elev,
                        "min_ground_layer_sem": min_ground_layer_sem,
                        "max_ground_layer_sem": max_ground_layer_sem,
                        "ceiling_layer_sem": ceiling_layer_sem,
                    },
                    f)


if __name__ == "__main__":
    PC_PATH = "/Users/ryanslocum/Documents/current_courses/PLR/data/tartanground/Downtown/Data_easy/P0006/point_cloud.pcd"
    generator = GroundTruthMapGenerator(PC_PATH, grid_resolution=GRID_RESOLUTION)
    generator.create_maps()
