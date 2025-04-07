import tartanair as ta
from scipy.spatial.transform import Rotation
import torch
import numpy as np
import torch.multiprocessing as mp
# --- Monkey patch the blend_func method in BlendBy2ndOrderGradTorch ---
import tartanair.image_resampling.image_sampler.blend_function as bf

# Backup the original method
orig_blend_func = bf.BlendBy2ndOrderGradTorch.blend_func


def patched_blend_func(self, img):
    """
    1) If input is np.ndarray, convert -> torch.Tensor and expand dims to [B, C, H, W].
    2) Call the original blend_func (which uses kornia).
    3) Convert the result back to a squeezed NumPy array so cv2.remap() won't fail.
    """
    # --- 1) Convert input to Torch if it's NumPy ---
    if isinstance(img, np.ndarray):
        # If depth is 2D: [H, W] => [1, 1, H, W]
        if img.ndim == 2:
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
        # If somehow 3D (e.g. [H, W, C] or [C, H, W]) => just put a batch dimension
        else:
            img = torch.from_numpy(img).unsqueeze(0)
    # if it's already a tensor, no change

    # --- 2) Call original blend function ---
    output = orig_blend_func(self, img)

    # --- 3) Convert output back to NumPy for cv2.remap ---
    # Make sure output is a 2D or 3D NumPy array. Usually you want 2D for the "mask".
    if isinstance(output, torch.Tensor):
        # convert to CPU NumPy
        output = output.detach().cpu().numpy()
        # typically, we want 2D for a mask => .squeeze()
        output = np.squeeze(output)

    return output


def resample_cameras(data_root, env, difficulty, trajectory):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    orig_blend_func = bf.BlendBy2ndOrderGradTorch.blend_func
    # Assign our patched method
    bf.BlendBy2ndOrderGradTorch.blend_func = patched_blend_func

    try:
        ta.init(data_root)

        # arrangement used in the nuScenes dataset
        camera_configs = [
            {'name': 'front', 'angle': 0},
            {'name': 'front_left', 'angle': 55},
            {'name': 'front_right', 'angle': -55},
            {'name': 'back_left', 'angle': 110},
            {'name': 'back_right', 'angle': -110},
            {'name': 'back', 'angle': 180}
        ]

        cameras = []
        for config in camera_configs:
            R_raw_new = Rotation.from_euler('xyz', [0, -config['angle'], 0], degrees=True).as_matrix().tolist()

            cam_model = {
                'name': 'pinhole',
                'raw_side': 'left',
                'cam_orientation': config['name'],
                'params': {
                    'fx': 320,
                    'fy': 320,
                    'cx': 320,
                    'cy': 320,
                    'width': 640,
                    'height': 640
                },
                'R_raw_new': R_raw_new
            }
            cameras.append(cam_model)

        ta.customize(
            env=env,
            difficulty=difficulty,
            trajectory_id=trajectory,
            modality=['image'],  # ['image', 'depth', 'seg']
            new_camera_models_params=cameras,
            num_workers=10,
            device=DEVICE
        )

    finally:
        # --- Restore original blend_func afterward ---
        bf.BlendBy2ndOrderGradTorch.blend_func = orig_blend_func



# if __name__ == '__main__':
#     mp.set_start_method('spawn', force=True)
    
#     # Backup the original method
    

#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#     resample_cameras()