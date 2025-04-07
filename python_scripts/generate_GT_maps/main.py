from train_data_tools import (
    create_binned_elev_maps,
    create_masks,
    process_lidar_to_bev,
    class_to_num_grouping,
    create_binned_semantics,
)
from create_GT_maps import GroundTruthMapGenerator
from create_point_cloud import LocalMappingRegister
import os


def gt_pipeline(
    root_dir,
    environment,
    difficulty,
    trajectory,
    original_semantics=None,
    preferred_grouping=None,
    generate_point_cloud=True,
    generate_gt_maps=True,
    generate_lidar_scans=True,
    generate_training_data=True,
    resample_camera_images=True,
    create_tar=True,
    delete_temp_files=False,
):

    traj_dir = os.path.join(root_dir, environment, f"Data_{difficulty}", trajectory)

    if original_semantics is not None and preferred_grouping is not None:
        # Create a mapping from original classes to new classes
        collapsed_semantic_classes = class_to_num_grouping(
            original_semantics, preferred_grouping
        )
    else:
        collapsed_semantic_classes = None

    if generate_point_cloud:
        processor = LocalMappingRegister(
            root_dir, collapsed_semantic_classes=collapsed_semantic_classes
        )
        processor.run()

    if generate_gt_maps:
        generator = GroundTruthMapGenerator(traj_dir)
        generator.create_maps()

    # Generate lidar scans
    if generate_lidar_scans:
        pass

    if generate_training_data:
        # Create binned semantics
        input_dir = os.path.join(traj_dir, "gt_output/sem/max_ground")
        output_dir = os.path.join(input_dir, "binned_maps")
        os.makedirs(output_dir, exist_ok=True)
        create_binned_semantics(input_dir, output_dir)

        # Create binned elevation maps
        input_dir = os.path.join(traj_dir, "gt_output/elev/max_ground")
        output_dir = os.path.join(input_dir, "binned_maps")
        os.makedirs(output_dir, exist_ok=True)
        create_binned_elev_maps(input_dir, output_dir)

        # Create binary masks
        input_dir = os.path.join(traj_dir, "gt_output/sem/max_ground/binned_maps")
        output_dirs = [
            os.path.join(traj_dir, "gt_output/sem/max_ground/valid_masks"),
            os.path.join(traj_dir, "gt_output/sem/max_ground/car_masks"),
        ]
        [os.makedirs(dir, exist_ok=True) for dir in output_dirs]
        create_masks(input_dir, output_dirs, [0, 5])

        # Create LIDAR BEV
        input_folder = os.path.join(traj_dir, "lidar")
        output_folder = os.path.join(traj_dir, "gt_output/lidar_bev")
        os.makedirs(output_folder, exist_ok=True)
        process_lidar_to_bev(input_folder, output_folder)

    # resample camera images
    if resample_camera_images:
        from resample_cameras import (
            resample_cameras,
        )  # Not sure if this is good practice but I can't run it without a GPU otherwise

        resample_cameras(data_root, environment, difficulty, trajectory)

    # postprocess/compress/delete
    if create_tar:
        dirs_to_tar = [
            "gt_output/sem/max_ground/valid_masks",
            "gt_output/sem/max_ground/car_masks",
            "gt_output/elev/max_ground/binned_maps",
            # "image_lcam_front",
            # "image_lcam_left",
            # "image_lcam_right",
            # "image_lcam_back",
        ]
        # Create tar file in root directory with all dirs_to_tar
        tar_file_path = os.path.join(traj_dir, "training_data.tar")
        print(f"Creating tar file at {tar_file_path}")
        os.system(f"tar -C {traj_dir} -cf {tar_file_path} {' '.join(dirs_to_tar)}")

    if delete_temp_files:
        pass


if __name__ == "__main__":
    data_root = "/Users/ryanslocum/Documents/current_courses/PLR/data/tartanground"
    environment = "Downtown"
    difficulty = "easy"
    trajectory = "P0006"
    original_semantics = "/Users/ryanslocum/Documents/current_courses/PLR/data/tartanground/Downtown/semantic_labels_Donwtown.json"
    preferred_grouping = {
        "ground": ["ground", "sidewalk"],
        "structures": ["building", "wall", "fence", "scaffolding", "door", "trim"],
        "vegetation": ["tree", "bush", "plant", "planter", "instancedfoliageactor"],
        "furniture_objects": [
            "bench",
            "table",
            "chair",
            "box",
            "rock",
            "umbrella",
            "trafficcone",
            "trafficlight",
        ],
        "vehicles": ["vehicle"],
        "sky": ["sky"],
        "other": [
            "ball",
            "barriermetal",
            "bikerack",
            "busstop",
            "concretebarrier",
            "container",
            "crosswalksignal",
            "flag",
            "garbagecan",
            "light",
            "pillar",
            "trash",
        ],
    }

    gt_pipeline(
        data_root,
        environment,
        difficulty,
        trajectory,
        original_semantics=original_semantics,
        preferred_grouping=preferred_grouping,
        generate_point_cloud=False,
        generate_gt_maps=False,
        resample_camera_images=False,
        generate_training_data=True,
        generate_lidar_scans=False,
        create_tar=False
    )
