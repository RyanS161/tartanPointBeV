""" 
Author: Loick Chambon

Project points from 3D points to 2D images.
"""

from typing import Dict, List

import torch
from einops import rearrange, repeat
from torch import nn

from pointbev.utils.debug import debug_hook
import matplotlib.pyplot as plt # TODO only temporary for plotting
import numpy as np # TODO only temporary for plotting


class CamProjector(nn.Module):
    def __init__(
        self,
        spatial_bounds: List[float] = [-49.75, 49.75, -49.75, 49.75, -3.375, 5.375],
        voxel_ref="spatial",
        z_value_mode: str = "zero",
    ):
        super().__init__()
        self.register_forward_hook(debug_hook)
        assert z_value_mode in ["zero", "contract", "affine", None]
        self.z_value_mode = z_value_mode

        self.spatial_bounds = spatial_bounds

    def _stats(self, voxels):
        print(f"Max: {voxels.max().item():.4f}")
        print(f"Min: {voxels.min().item():.4f}")
        print(f"Mean: {voxels.mean().item():.4f}")
        print(f"Shape: {voxels.shape}")

    def _set_axis(self, vox_coords):
        """Deduce axis parameters: spatial (X,Y,Z) and camera (X_cam,Y_cam,Z_cam)."""
        X, Y, Z = vox_coords.shape[-3:]
        self.X, self.Y, self.Z = X, Y, Z
        self.X_cam, self.Y_cam, self.Z_cam = Y, Z, X

    # Voxel to cams
    def from_voxel_ref_to_cams(self, vox_coords, rots, trans, bev_aug, egoTin_to_seq, intrins):
        """Project points from voxel reference to camera reference.
        Args:
            - rots, trans: map points from cameras to ego. In Nuscenes, extrinsics
            are inverted compared to standard conventions. They map sensors to ego.

        Returns:
            - Voxel camera coordinates: coordinates of the voxels in the camera reference frame.
            - Voxel coordinates: coordinates of the voxels in the ego (sequence and augmentation) reference frame.
        """

        print("vox_coords min:", vox_coords.min().item(), 
              "max:", vox_coords.max().item(), 
              "mean:", vox_coords.mean().item())
        
        print("vox_coords min:", vox_coords.min().item(), 
              "max:", vox_coords.max().item(), 
              "mean:", vox_coords.mean().item())
        
        self.plot_3d_voxels(vox_coords, rots, trans, intrins)

        print("START min/mean/max:", vox_coords.min().item(), vox_coords.mean().item(), vox_coords.max().item())
        vox_coords = self.from_spatial_to_seqaug(vox_coords, bev_aug, egoTin_to_seq)
        print("AFTER seqaug min/mean/max:", vox_coords.min().item(), ...)
        voxcam_coords = self.from_spatial_to_cams(vox_coords, rots, trans)
        print("AFTER to_cams min/mean/max:", vox_coords.min().item(), ...)
        
        return voxcam_coords, vox_coords

    def from_spatial_to_seqaug(self, vox_coords, bev_aug, egoTin_to_seq):
        """Map points from spatial reference frame to augmented reference frame.

        Decomposition:
            - ego to egoseq: (R0, T0)
            - egoseq to bevaug: (R1, T1)
        """
        # Prepare vox_coords
        vox_coords = rearrange(vox_coords, "bt i x y z -> bt i (x y z)", i=3)

        # Apply: egoTin_to_seq.
        egoTin_to_seq = torch.linalg.inv(
            repeat(egoTin_to_seq, "b t i j -> (b t) i j", i=4, j=4)
        )

        # Apply: bev_aug.
        bev_aug = repeat(bev_aug, "b t i j -> (b t) i j", i=4, j=4)

        vox_coords = torch.cat([vox_coords, torch.ones_like(vox_coords[:, :1])], dim=1)
        vox_coords_aug = torch.bmm(bev_aug, torch.bmm(egoTin_to_seq, vox_coords))
        return vox_coords_aug[:, :3]

    def from_spatial_to_cams(self, vox_coords, rots, trans):
        """
        Map points from augmented reference frame to camera reference frame.

        Decomposition:
            - ego to cameras: (R2^-1, -R2^-1 @ T2)

        Formula: spatial to cameras:
            - Rotation: R2^-1 @ R1
            - Translation: R2^-1 @ (T1 - T2)
        """
        # Alias
        bt, n, *_ = rots.shape

        # Prepare: from cameras to ego.
        homog_mat = torch.eye(4, device=rots.device).repeat(bt * n, 1, 1)
        homog_mat[:, :3, :3] = rots.flatten(0, 1)
        homog_mat[:, :3, -1:] = trans.flatten(0, 1)
        homog_mat = torch.linalg.inv(homog_mat)

        # Apply: camera transformations.
        vox_coords = torch.cat([vox_coords, torch.ones_like(vox_coords[:, :1])], dim=1)
        vox_coords = repeat(vox_coords, "bt i Npts -> (bt n) i Npts", n=n, i=4)
        voxcam_coords = torch.bmm(homog_mat, vox_coords)[:, :3]

        # Inside from_spatial_to_cams
        print(f"homog_mat[0]: {homog_mat[0]}")


        return rearrange(voxcam_coords, "(bt n) i Npts -> bt n i Npts", bt=bt, n=n, i=3)

    # Cams to pixels
    def from_cameras_to_pixels(self, voxels, intrins):
        """Transform points from camera reference frame to image reference frame."""
        # Alias
        bt, n, i, j = intrins.shape

        intrins = rearrange(intrins, "bt n i j -> (bt n) i j", bt=bt, n=n, i=i, j=j)
        voxels = rearrange(voxels, "bt n i Npts -> (bt n) i Npts", bt=bt, n=n, i=3)
        voxels = torch.bmm(intrins, voxels)
        return rearrange(voxels, "(bt n) i Npts -> bt n i Npts", bt=bt, n=n, i=3)

    # Valid points
    def normalize_z_cam(self, voxels, eps=1e-6):
        """By convention, the Z_cam-coordinate on image references is equal to 1, so we rescale X,Y such
        that their Z equals 1."""
        normalizer = voxels[..., 2:3, :].clip(min=eps)
        return voxels / normalizer

    def valid_points_in_pixels(self, voxels, img_res):
        """Since we will interpolate with align corner = False, we consider only points
        inside empty circle in https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663)

        Args:
            - Voxels: in image reference frame. (B,T,N,3,N_pts) where N_pts = X_cam*Y_cam*Z_cam.
        """
        # Alias
        H, W = img_res

        x_valid = (voxels[..., 0, :] > +0.5) & (voxels[..., 0, :] < W - 0.5)
        y_valid = (voxels[..., 1, :] > +0.5) & (voxels[..., 1, :] < H - 0.5)
        return x_valid, y_valid

    def valid_points_in_cam(self, voxels):
        """Points are valid in camera reference, if they are forward the Z_cam-axis."""
        return (voxels[..., -1, :] > 0.0).bool()

    # Prepare VT
    def normalize_vox(self, voxels, img_res, clamp_extreme=True):
        """
        Since we will interpolate with align corner = False, we need to map [0.5, W-0.5] to [-1,1].

        Note: z is supposed to be 1, after normalization, and the output should have a z equals to 0.
        """
        # Alias
        H, W = img_res
        device = voxels.device

        denom = rearrange(
            torch.tensor([W - 1, H - 1, 2], device=device), "i -> 1 1 i 1", i=3
        )
        add = rearrange(
            torch.tensor([(1 - W) / 2, (1 - H) / 2, 0], device=device),
            "i -> 1 1 i 1",
            i=3,
        )
        sub = rearrange(
            torch.tensor([1 / (W - 1), 1 / (H - 1), 0], device=device),
            "i -> 1 1 i 1",
            i=3,
        )
        voxels = 2.0 * ((voxels + add) / denom) - sub

        if clamp_extreme:
            voxels = voxels.clamp(-2, 2)
        return voxels

    def modify_z_value(self, voxels, z_before_norm, z_value_mode: bool = True):
        """Either set z to zero or adapt z to be in [-1,1] using the MIP-NeRF contraction."""
        if z_value_mode != "zero":
            zmin, zmax = self.spatial_bounds[:2]
            z_before_norm = z_before_norm / (max(abs(zmin), abs(zmax)) * 1.4142135)

        # Fix the last coordinates to zero.
        if z_value_mode == "zero":
            voxels = torch.cat(
                [voxels[..., :2, :], torch.zeros_like(voxels[..., :1, :])], dim=-2
            )

        # Contract z: [-1,1]
        elif z_value_mode == "contract":
            voxels[:, :, 2:3] = (
                torch.where(
                    z_before_norm.abs() <= 1,
                    z_before_norm,
                    (2 - 1 / z_before_norm.abs())
                    * (z_before_norm / z_before_norm.abs()),
                )
                - 1
            )

        # Affine transformation: [-1,1]
        elif z_value_mode == "affine":
            voxels[:, :, 2:3] = z_before_norm * 2 - 1
        return voxels

    def arange_voxels(self, voxcam_coords, vox_valid, vox_coords, b_t_n):
        """Arange shapes and normalize vox_coords in [-1,1]."""
        # Alias
        b, t, n = b_t_n

        list_out = []
        for _, i in zip([voxcam_coords, vox_valid], [3, 1]):
            list_out.append(
                rearrange(
                    _,
                    "(b t) n i (zcam xcam ycam) -> b t n zcam ycam xcam i",
                    b=b,
                    t=t,
                    n=n,
                    i=i,
                    zcam=self.Z_cam,
                    xcam=self.X_cam,
                    ycam=self.Y_cam,
                )
            )

        vox_coords = rearrange(
            vox_coords,
            "(b t) i (zcam xcam ycam) -> b t zcam ycam xcam i",
            b=b,
            t=t,
            zcam=self.Z_cam,
            xcam=self.X_cam,
            ycam=self.Y_cam,
            i=3,
        )

        # Normalize vox coords for GS embedding.
        # 1.2 usefull to avoid border effects when having seqaug matrix.
        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = self.spatial_bounds
        vox_coords = vox_coords / torch.tensor(
            [
                1.2 * max(abs(XMIN), abs(XMAX)),
                1.2 * max(abs(YMIN), abs(YMAX)),
                1.2 * max(abs(ZMIN), abs(ZMAX)),
            ],
            device=vox_coords.device,
            dtype=vox_coords.dtype,
        )
        list_out.append(vox_coords)

        return list_out

    # Forward
    def forward(self, dict_mat, dict_shape, dict_vox) -> Dict[str, torch.Tensor]:
        # Unpack
        rots, trans, intrins, bev_aug, egoTin_to_seq = (
            dict_mat["rots"],
            dict_mat["trans"],
            dict_mat["intrins"],
            dict_mat["bev_aug"],
            dict_mat["egoTin_to_seq"],
        )
        vox_coords = dict_vox.get("vox_coords", None)

        # Alias
        (b, n, t) = [dict_shape[k] for k in ["b", "n", "t"]]
        img_feats_res = (dict_shape["Hfeats"], dict_shape["Wfeats"])

        # Set axis range.
        self._set_axis(vox_coords)

        print("START min/mean/max:", vox_coords.min().item(), vox_coords.mean().item(), vox_coords.max().item())


        # Ego to cams.
        voxcam_coords, vox_coords = self.from_voxel_ref_to_cams(
            vox_coords,
            rots,
            trans,
            bev_aug,
            egoTin_to_seq,
            intrins
        )
        z_valid = self.valid_points_in_cam(voxcam_coords)
        print("START min/mean/max:", vox_coords.min().item(), vox_coords.mean().item(), vox_coords.max().item())


        # Cams to pixels.
        voxcam_coords = self.from_cameras_to_pixels(voxcam_coords, intrins)
        if self.z_value_mode != "zero":
            z_before_norm = voxcam_coords[:, :, 2:3]  # Get z before normalization.
        else:
            z_before_norm = None
        voxcam_coords = self.normalize_z_cam(voxcam_coords)
        x_valid, y_valid = self.valid_points_in_pixels(voxcam_coords, img_feats_res)

        # # Filter valid points.
        vox_valid = (x_valid & y_valid & z_valid).unsqueeze(-2)
        voxcam_coords = self.normalize_vox(voxcam_coords, img_feats_res)

        # Adapt z:
        voxcam_coords = self.modify_z_value(
            voxcam_coords, z_before_norm, self.z_value_mode
        )

        # Get features in img space.
        voxcam_coords, vox_valid, vox_coords = self.arange_voxels(
            voxcam_coords, vox_valid, vox_coords, (b, t, n)
        )
        print("START min/mean/max:", vox_coords.min().item(), vox_coords.mean().item(), vox_coords.max().item())

        self.visualize_valid_voxels_3d(vox_valid, vox_coords, rots, trans)

        self.visualize_camera_coverage_bev_improved(vox_valid)
            
        return dict(
            {
                "voxcam_coords": voxcam_coords,
                "vox_valid": vox_valid,
                "vox_coords": vox_coords,
            }
        )
    


    def plot_3d_voxels(self, vox_coords, rots, trans, intrins):
        axis_len = 5.0
        frustum_len = 1.5

        # voxel to ego frame
        with torch.no_grad():
            coords_np = vox_coords[0].cpu().numpy()
            coords_np = coords_np.reshape(3, -1)
            x, y, z = coords_np[0], coords_np[1], coords_np[2]

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection="3d")
            ax.scatter(x[::1], y[::1], z[::1], s=3, alpha=0.5, label="Voxel centers")

        rots_np = rots[0].cpu().numpy()               # [6, 3, 3]
        trans_np = trans[0, :, :, 0].cpu().numpy()    # [6, 3]

        for cam_idx in range(rots_np.shape[0]):
            R = rots_np[cam_idx]
            t = trans_np[cam_idx]

            print(f"Camera {cam_idx} position: {t}, R shape: {R.shape}")

            x_axis = R @ np.array([1, 0, 0])  # right
            y_axis = R @ np.array([0, 1, 0])  # down
            z_axis = R @ np.array([0, 0, 1])  # forward

            ax.scatter(t[0], t[1], t[2], color='black', marker='o')
            ax.text(t[0], t[1], t[2], f"Cam {cam_idx}", color='black')

            ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2],
                    length=axis_len, color='red', label='Z axis' if cam_idx == 0 else None)
            ax.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2],
                    length=axis_len, color='green', label='X axis' if cam_idx == 0 else None)
            ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2],
                    length=axis_len, color='blue', label='Y axis' if cam_idx == 0 else None)

            fx = intrins[0, cam_idx, 0, 0].item()
            fy = intrins[0, cam_idx, 1, 1].item()
            w = intrins[0, cam_idx, 0, 2].item() * 2
            h = intrins[0, cam_idx, 1, 2].item() * 2
            fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
            fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

            cx = np.tan(fov_x / 2) * frustum_len
            cy = np.tan(fov_y / 2) * frustum_len

            frustum_points_cam = np.array([
                [0, 0, 0],                # camera center
                [-cx, -cy, frustum_len],  # top-left
                [ cx, -cy, frustum_len],  # top-right
                [ cx,  cy, frustum_len],  # bottom-right
                [-cx,  cy, frustum_len],  # bottom-left
            ])

            frustum_points_ego = (R @ frustum_points_cam.T).T + t

            # lines from cam center to corners
            for i in range(1, 5):
                ax.plot(
                    [frustum_points_ego[0, 0], frustum_points_ego[i, 0]],
                    [frustum_points_ego[0, 1], frustum_points_ego[i, 1]],
                    [frustum_points_ego[0, 2], frustum_points_ego[i, 2]],
                    color='orange', linestyle='--', linewidth=1.0
                )

            # frustum base square
            for i in range(1, 5):
                j = 1 if i == 4 else i + 1
                ax.plot(
                    [frustum_points_ego[i, 0], frustum_points_ego[j, 0]],
                    [frustum_points_ego[i, 1], frustum_points_ego[j, 1]],
                    [frustum_points_ego[i, 2], frustum_points_ego[j, 2]],
                    color='orange', linestyle='-', linewidth=1.0
                )

        ax.set_xlabel("X (Forward)")
        ax.set_ylabel("Y (Left)")
        ax.set_zlabel("Z (Up)")
        ax.set_title("Voxel Grid + Camera Positions in Ego Frame")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def visualize_valid_voxels_3d(self, vox_valid, vox_coords, rots, trans):
        """3D voxels colored by their validity for each camera"""

        # vox_valid = [B=1, T=1, N=6, Z=2500, Y=8, X=1, i=1] whereas i stand for the fact that we have binary data. For vox_coords the i=3 spatial for the 3 dimensions
        vox_valid_np = vox_valid[0, 0].squeeze(-1).cpu().numpy()  # [6=cams, 2500=xy ground plane (50x50), 8=z up, 1=singleton that can be squeezed)] however they label it z=2500, y=8, x=1 for some arbitrary reason
        vox_coords_np = vox_coords[0, 0].cpu().numpy()  # (2500, 8, 1, 3)
        
        rots_np = rots[0].cpu().numpy()               # [6, 3, 3]
        trans_np = trans[0, :, :, 0].cpu().numpy()    # [6, 3]
        
        Z, Y, X = vox_coords_np.shape[:-1] # Z=2500, Y=8, X=1
        
        for cam_idx in [0,1,2,3,4,5]:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            valid_mask = vox_valid_np[cam_idx]  # (Z, Y, X)
            
            # for performance
            sample_rate = 0.2  # 20% of voxels
            random_mask = np.random.rand(Z, Y, X) < sample_rate
            
            valid_points = valid_mask & random_mask
            invalid_points = (~valid_mask) & random_mask
            
            valid_coords = vox_coords_np[valid_points]
            invalid_coords = vox_coords_np[invalid_points]
            
            # valid voxels in green, invalid in red
            if len(valid_coords) > 0:
                ax.scatter(valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2], 
                        c='green', s=10, alpha=0.5, label='Valid voxels')
            if len(invalid_coords) > 0:
                ax.scatter(invalid_coords[:, 0], invalid_coords[:, 1], invalid_coords[:, 2], 
                        c='red', s=10, alpha=0.1, label='Invalid voxels')
            
            t = trans_np[cam_idx]
            R = rots_np[cam_idx]
            z_axis = R @ np.array([0, 0, 1])  # forward direction
            
            ax.scatter([t[0]], [t[1]], [t[2]], color='black', s=100, marker='o')
            ax.quiver(t[0], t[1], t[2], z_axis[0]*5, z_axis[1]*5, z_axis[2]*5,
                    color='blue', arrow_length_ratio=0.2, linewidth=2)

            ax.set_xlabel('X (forward)')
            ax.set_ylabel('Y (left)')
            ax.set_zlabel('Z (up)')
            ax.set_title(f'Camera {cam_idx} Valid Voxels (green) vs Invalid Voxels (red)')
            ax.legend()
            
            plt.tight_layout()

    def visualize_camera_coverage_bev_improved(self, vox_valid):
        """BEV coverage for sparse voxel points as a grid"""

        # vox_valid = [B=1, T=1, N=6, Z=2500, Y=8, X=1, i=1] whereas i stand for the fact that we have binary data. For vox_coords the i=3 spatial for the 3 dimensions
        vox_valid_np = vox_valid[0, 0].squeeze(-1).cpu().numpy()  # [6=cams, 2500=xy ground plane (50x50), 8=z up, 1=singleton that can be squeezed)] however they label it z=2500, y=8, x=1 for some arbitrary reason
        vox_valid_np = vox_valid_np.squeeze(-1) # [6, 2500, 8]. squeezing out the x which is only a placeholder since the xy plane for the bev is stored in z = 50x50

        vox_valid_grid = vox_valid_np.reshape(6, 50, 50, 8) # reshaping 2500 into 50 x 50
        print(f"After reshape: {vox_valid_grid.shape}")

        def visualize_height_slices(vox_valid_grid, cam_idx=0):
            """
            Plot each height slice individually for a single camera in a grid of subplots.
            Args:
            vox_valid_grid: shape (n_cams, X, Y, Z).
            cam_idx: which camera index to plot.
            """
            
            n_cams, X, Y, Z = vox_valid_grid.shape
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 8 slices => 2 rows x 4 cols
            
            for z in range(Z):
                ax = axes[z // 4, z % 4]
                # shape (X, Y) for this slice
                slice2d = vox_valid_grid[cam_idx, :, :, z]
                
                im = ax.imshow(slice2d, cmap='viridis')
                ax.set_title(f"Camera {cam_idx}, height slice {z}")
                fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

            plt.tight_layout()
            plt.show()


        visualize_height_slices(vox_valid_grid, cam_idx=0)

        plt.figure(figsize=(15, 10))
        cam_names = ['Front', 'Front Right', 'Right', 'Back', 'Left', 'Front Left']

        for cam_idx in range(vox_valid_grid.shape[0]):
            plt.subplot(2, 3, cam_idx+1)
            
            # Sum over height dimension (the last axis => -1)
            height_visibility = vox_valid_grid[cam_idx].sum(axis=-1)  # resulting shape: (50, 50)

            print("height_visibility shape =", height_visibility.shape)
            nonzero_coords = np.argwhere(height_visibility > 0)
            print("Number of nonzero cells =", len(nonzero_coords))

            unique_vals = np.unique(height_visibility)
            print("Unique values in height_visibility:", unique_vals)

            # If you want to see the bounding box of nonzero area:
            if len(nonzero_coords) > 0:
                min_x, min_y = nonzero_coords.min(axis=0)
                max_x, max_y = nonzero_coords.max(axis=0)
                print(f"Nonzero region spans from ({min_x},{min_y}) to ({max_x},{max_y})")


            plt.imshow(height_visibility, cmap='viridis')

            plt.title(f"Camera {cam_idx}: {cam_names[cam_idx]}")
            plt.colorbar(label="Height levels visible")
            
            # plt.plot(0, 0, 'rx', markersize=10)  # Ego
            plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.show()