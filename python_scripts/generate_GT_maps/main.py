from train_data_tools import (
    create_binned_elev_maps,
    create_car_masks,
    process_lidar_to_bev,
)
from create_GT_maps import GroundTruthMapGenerator
import os


GENERATE_GT_MAPS = True
GENERATE_TRAINING_DATA = True
TRAJ_ROOT = "/Users/ryanslocum/Documents/current_courses/PLR/data/tartanground/Downtown/Data_easy/P0006"


def main():
    if GENERATE_GT_MAPS:
        generator = GroundTruthMapGenerator(TRAJ_ROOT)
        generator.create_maps()

    if GENERATE_TRAINING_DATA:
        # Create binned elevation maps
        input_dir = os.path.join(TRAJ_ROOT, "gt_output/elev/max_ground")
        output_dir = os.path.join(input_dir, "binned_maps")
        os.makedirs(output_dir, exist_ok=True)
        create_binned_elev_maps(input_dir, output_dir)

        # Create binary masks
        input_dir = os.path.join(TRAJ_ROOT, "gt_output/sem/max_ground")
        color_mask_dir = os.path.join(TRAJ_ROOT, "gt_output/sem/max_ground/car_masks")
        valid_mask_dir = os.path.join(TRAJ_ROOT, "gt_output/sem/max_ground/valid_masks")
        os.makedirs(color_mask_dir, exist_ok=True)
        os.makedirs(valid_mask_dir, exist_ok=True)
        create_car_masks(input_dir, color_mask_dir, valid_mask_dir)

        # Create LIDAR BEV
        input_folder = os.path.join(TRAJ_ROOT, "lidar")
        output_folder = os.path.join(TRAJ_ROOT, "gt_output/lidar_bev")
        os.makedirs(output_folder, exist_ok=True)
        process_lidar_to_bev(input_folder, output_folder)


if __name__ == "__main__":
    main()
