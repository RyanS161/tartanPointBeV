import os
import shutil

cam_names = ["front", "front_left", "front_right", "back_left", "back_right", "back"]
modalities = ["pose", "image"]
base_path = "/home/michael/Documents/Master_Sem_2/PLR/PointBeV/data/tartanground/Downtown/Data_easy/P0006"

for modality in modalities:
    for idx, cam_name in enumerate(cam_names):

        if modality == "pose":
            old_pose_name = f"{base_path}/{modality}_lcam_custom{idx}_pinhole.txt"
            new_pose_name = f"{base_path}/{modality}_lcam_{cam_name}_pinhole.txt"

            # Rename pose files
            if os.path.exists(old_pose_name):
                if os.path.exists(new_pose_name):
                    os.remove(new_pose_name)
                os.rename(old_pose_name, new_pose_name)
                print(f"Renamed: {old_pose_name} → {new_pose_name}")
            else:
                print(f"Missing pose file: {old_pose_name}")

        if modality == "image":
            old_folder_name = f"{modality}_lcam_custom{idx}_pinhole"
            new_folder_name = f"{modality}_lcam_{cam_name}_pinhole"

            old_json_name = f"{base_path}/{old_folder_name}/camera_model_params_lcam_{modality}_custom{idx}_pinhole.json"
            new_json_name = f"{base_path}/{old_folder_name}/camera_model_params_{modality}_lcam_{cam_name}_pinhole.json"

            # Rename json files
            if os.path.exists(old_json_name):
                if os.path.exists(new_json_name):
                    os.remove(new_json_name)
                os.rename(old_json_name, new_json_name)
                print(f"Renamed JSON: {old_json_name} → {new_json_name}")
            else:
                print(f"Missing JSON: {old_json_name}")

            # Rename images
            directory = os.path.join(base_path, old_folder_name)
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    if filename.endswith(".png"):
                        old_image_name = os.path.join(directory, filename)
                        new_filename = filename.replace(f"lcam_{modality}_custom{idx}_pinhole", f"{modality}_lcam_{cam_name}_pinhole")
                        new_image_name = os.path.join(directory, new_filename)

                        if os.path.exists(new_image_name):
                            os.remove(new_image_name)  # Overwrite
                        os.rename(old_image_name, new_image_name)
                        print(f"Renamed image: {old_image_name} → {new_image_name}")
            else:
                print(f"Missing image directory: {directory}")

            # Rename the folder itself
            old_folder_path = os.path.join(base_path, old_folder_name)
            new_folder_path = os.path.join(base_path, new_folder_name)
            if os.path.exists(old_folder_path):
                if os.path.exists(new_folder_path):
                    shutil.rmtree(new_folder_path)  # Overwrite
                os.rename(old_folder_path, new_folder_path)
                print(f"Renamed folder: {old_folder_path} → {new_folder_path}")
            else:
                print(f"Missing folder: {old_folder_path}")
