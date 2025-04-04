from train_data_tools import (
    create_binned_elev_maps,
    create_car_masks,
    process_lidar_to_bev,
    class_to_num_grouping
)
from create_GT_maps import GroundTruthMapGenerator
from create_point_cloud import LocalMappingRegister
import os


def gt_pipeline(
    root_dir,
    collapsed_semantic_classes=None,
    generate_point_cloud=True,
    generate_gt_maps=True,
    generate_training_data=True,
    resample_camera_images=True,
    create_tar=True,
    delete_temp_files=False,
):
    if generate_point_cloud:
        processor = LocalMappingRegister(root_dir, collapsed_semantic_classes=collapsed_semantic_classes)
        processor.run()

    if generate_gt_maps:
        generator = GroundTruthMapGenerator(TRAJ_ROOT)
        generator.create_maps()

    # Generate lidar scans

    if generate_training_data:
        # # Collapse semantics into different channels
        # input_dir = os.path.join(TRAJ_ROOT, "gt_output/sem/max_ground")
        # output_dir = os.path.join(input_dir, "collapsed")
        # os.makedirs(output_dir, exist_ok=True)
        # collapse_semantics(input_dir, output_dir)

        # Create binned elevation maps
        input_dir = os.path.join(TRAJ_ROOT, "gt_output/elev/max_ground")
        output_dir = os.path.join(input_dir, "binned_maps")
        os.makedirs(output_dir, exist_ok=True)
        create_binned_elev_maps(input_dir, output_dir)

        # Create binary masks
        input_dir = os.path.join(TRAJ_ROOT, "gt_output/sem/max_ground")
        color_mask_dir = os.path.join(input_dir, "car_masks")
        valid_mask_dir = os.path.join(input_dir, "valid_masks")
        os.makedirs(color_mask_dir, exist_ok=True)
        os.makedirs(valid_mask_dir, exist_ok=True)
        create_car_masks(input_dir, color_mask_dir, valid_mask_dir)

        # Create LIDAR BEV
        input_folder = os.path.join(TRAJ_ROOT, "lidar")
        output_folder = os.path.join(TRAJ_ROOT, "gt_output/lidar_bev")
        os.makedirs(output_folder, exist_ok=True)
        process_lidar_to_bev(input_folder, output_folder)

    # resample camera images
    if resample_camera_images:
        pass

    # postprocess/compress/delete
    if create_tar:
        dirs_to_tar = [
            os.path.join(TRAJ_ROOT, "gt_output/sem/max_ground/valid_masks"),
            os.path.join(TRAJ_ROOT, "gt_output/sem/max_ground/car_masks"),
            os.path.join(TRAJ_ROOT, "gt_output/elev/max_ground/binned_maps"),
            os.path.join(TRAJ_ROOT, "image_lcam_front"),
            os.path.join(TRAJ_ROOT, "image_lcam_left"),
            os.path.join(TRAJ_ROOT, "image_lcam_right"),
            os.path.join(TRAJ_ROOT, "image_lcam_back"),
        ]
        # Create tar file in root directory with all dirs_to_tar
        tar_file_path = os.path.join(TRAJ_ROOT, "compressed_training_data.tar")
        print(f"Creating tar file at {tar_file_path}")
        os.system(f"tar -cf {tar_file_path} -C {TRAJ_ROOT} {' '.join(dirs_to_tar)}")

    if delete_temp_files:
        pass


if __name__ == "__main__":
    TRAJ_ROOT = "/Users/ryanslocum/Documents/current_courses/PLR/data/tartanground/Downtown/Data_easy/P0006"

    original_semantics = "/Users/ryanslocum/Documents/current_courses/PLR/data/tartanground/Downtown/semantic_labels_Donwtown.json"
    preferred_grouping = {
        "ground": ["ground", "sidewalk"],
        "structures": ["building", "wall", "fence", "scaffolding", "door", "trim"],
        "vegetation": ["tree", "bush", "plant", "planter", "instancedfoliageactor"],
        "furniture_objects": ["bench", "table", "chair", "box", "rock", "umbrella"],
        "vehicles": ["vehicle", "trafficcone", "trafficlight"],
        "sky": ["sky"],
        "other": ["ball", "barriermetal", "bikerack", "busstop", "concretebarrier", "container", "crosswalksignal", "flag", "garbagecan", "light", "pillar", "trash"],
    }
    # collapsed_semantic_classes = [
    #     range(1, 18),
    #     range(18, 36),
    # ]
    collapsed_semantic_classes = class_to_num_grouping(original_semantics, preferred_grouping)
    gt_pipeline(
        TRAJ_ROOT,
        collapsed_semantic_classes=collapsed_semantic_classes,
        generate_point_cloud=True,
        generate_gt_maps=True,
        resample_camera_images=False,

    )

