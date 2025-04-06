## Usage

1. resample_cameras.py
Generates the images in the correct views, a json file with the static camera transformation and intrinsics, and a pose file with the respective camera_world transformation for every image

2. tartanair_to_pointbev_extrinsics.py
Converts the poses and static camera transformations from the tartanair frame convention to the pointbev / nuscenes frame convention

3. rename_files.py
Renames the files into the naming convention that's needed

4. verify_extrinsics.py
Creates a plot of the trajectory with the corresponding orientation indicated as arrow
Creates a plot that backprojects a certain 3D point expressed in pointbev frame convention to the image plane