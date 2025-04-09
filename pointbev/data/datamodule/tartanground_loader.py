""" 
Author: Ryan Slocum, Michael Lötscher

Datamodule for the Tartanground dataset.
"""


from typing import Optional

import pytorch_lightning as pl
import torch
from tartanair import TartanAirDataset
from torch.utils.data import random_split
from scipy.spatial.transform import Rotation
import json
import os
import numpy as np
import torchvision.transforms.functional as TF  # TODO only temporary needed
import torchvision.transforms as T # TODO only temporary needed
from PIL import Image
import matplotlib.pyplot as plt
from pointbev.utils.imgs import NORMALIZE_IMG

class TartangroundDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        # Tartanground
        envs,
        dataroot,
        # Grid
        grid,
        # Images
        img_loader,
        img_params,
        # Dataloader
        batch_size=1,
        valid_batch_size=None,
        num_workers=10,
        pin_memory=True,
        prefetch_factor=2,
        train_drop_last=True,
        train_shuffle=False,
        val_shuffle=False,
        normalize_img=True,
        **kwargs):
        super().__init__()

        # Nuscenes
        self.envs = list(envs)
        self.dataroot = dataroot
        # Grid
        self.grid = grid
        # Images
        self.img_loader = img_loader
        self.img_params = img_params
        # Coefficients
        # Dataloader
        self.batch_size = int(batch_size)
        self.valid_batch_size = (
            int(valid_batch_size) if valid_batch_size is not None else int(batch_size)
        )
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.train_drop_last = train_drop_last
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle

    def setup(self, stage: Optional[str] = None):
        data = TartanAirDataset(self.dataroot)
        data.create_image_dataset(self.envs,
                                  modality=['image'],
                                  camera_name=list(self.img_params.cams))
        wrapped_dataset = IndexWrapper(data.dataset) # adding indices

        self.train_data = wrapped_dataset  # Use all data for training
        val_len = min(250, len(wrapped_dataset)//5)  # Use ~20% or max 250 samples
        self.val_data = torch.utils.data.Subset(wrapped_dataset, list(range(val_len)))
        print(f"Training set size: {len(self.train_data)}, Validation set size: {len(self.val_data)}")


        self.camera_params = {}
        env = self.envs[0]  # Assuming all cameras share same params across environments
        difficulty = "Data_easy"  # hardcoded for now. Can parameterize in tartanground.yaml later
        traj = "P0006"  # hardcoded for now. Can parameterize in tartanground.yaml later

        for cam in self.img_params.cams:            
            param_file = os.path.join(
                self.dataroot, 
                env, 
                difficulty,
                traj,
                f"image_{cam}_pinhole",
                f"camera_model_params_image_{cam}_pinhole.json"
            )

            pose_file = os.path.join(
                self.dataroot, 
                env, 
                difficulty,
                traj,
                f"pose_{cam}_pinhole.txt"
            )

            if os.path.exists(param_file):
                with open(param_file, 'r') as f:
                    content = json.load(f)
                    self.camera_params[cam] = {
                        'intrinsics': [
                            [content['params']['fx'], 0, content['params']['cx']],
                            [0, content['params']['fy'], content['params']['cy']],
                            [0, 0, 1]
                        ],
                        'R_body_camera': content['R_raw_new']
                    }
            else:
                print(f"Warning: Camera parameter file not found: {param_file}")
                # Fall back to default params
                self.camera_params[cam] = {
                    'intrinsics': [[320.0, 0.0, 320.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]],
                    'R_body_camera': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                }

            if not os.path.exists(pose_file):
                raise FileNotFoundError(f"Camera pose file not found: {pose_file}")
            with open(pose_file, 'r') as f:
                content = f.readlines()
                self.camera_params[cam]['T_world_camera'] = [
                    [float(x) for x in line.strip().split()] for line in content
                ]

        self.binimg_path = "/home/michael/Desktop/training_data/gt_output/sem/max_ground/car_masks"
        self.valid_binimg_path = "/home/michael/Desktop/training_data/gt_output/sem/max_ground/valid_masks"
            

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=self.train_drop_last,
            # worker_init_fn=worker_rnd_init,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.valid_batch_size,
            shuffle=self.val_shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valdata, #TODO fix later
            batch_size=self.valid_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    """
    current batch sturcture:
    {
        'lcam_front': {
            'image_0': np.array([H, W, 3]),
            'image_1': np.array([H, W, 3]),
            'motion': np.array([6]),
        },
        'lcam_back': {
            'image_0': np.array([H, W, 3]),
            'image_1': np.array([H, W, 3]),
            'motion': np.array([6]),
        },
    
    expected batch structure:
    {
    'imgs': {cam_name: tensor(B, T, 3, H, W)},   # camera → batch of images
    'rots': tensor(B, T, N, 3, 3),     static transformation of the camera to base frame
    'trans': tensor(B, T, N, 3, 1),       static transformation of the camera to the base frame
    'intrins': tensor(B, T, N, 3, 3),
    'bev_aug': tensor(B, T, 4, 4),
    'egoTin_to_seq': tensor(B, T, 4, 4)
    }  
    """
    def collate_fn(self, batch):
        """
        Convert list of samples → batch dict that matches PointBEV forward() inputs.
        """
        B = len(batch) # batch size
        N = len(self.img_params.cams)  # number of cameras
        egoTin_to_seq_list = [] # egoTin_to_seq is not camera specific but for base_frame (front camera)
        binimgs = []
        valid_binimgs = []

        for i, sample in enumerate(batch):
            frame_idx = sample["frame_idx"]

            binimg_file       = os.path.join(self.binimg_path, f"{frame_idx:06d}.npy")
            valid_binimg_file = os.path.join(self.valid_binimg_path, f"{frame_idx:06d}.npy")

            if os.path.exists(binimg_file):
                binimg_np = np.load(binimg_file)
                binimg_ = torch.from_numpy(binimg_np).float().unsqueeze(0) # / 255.0
            else:
                print(f"{binimg_file} not found.")
                binimg_ = torch.zeros((1, 100, 100), dtype=torch.float32)

            if os.path.exists(valid_binimg_file):
                valid_np = np.load(valid_binimg_file)
                valid_binimg_ = torch.from_numpy(valid_np).bool().unsqueeze(0)
            else:
                valid_binimg_ = torch.zeros((1, 100, 100), dtype=torch.bool)

            binimg_ = TF.resize(binimg_, [200, 200], interpolation=T.InterpolationMode.NEAREST)
            valid_binimg_ = TF.resize(valid_binimg_.float(), [200, 200], interpolation=T.InterpolationMode.NEAREST).bool()

            # print(f"binimg_ final sum before append: {binimg_.sum().item()}")
            binimgs.append(binimg_)
            valid_binimgs.append(valid_binimg_)

            egoTin_ = self.poses_to_egoTin_to_seq(
                self.camera_params["lcam_front"]["T_world_camera"],
                [frame_idx],  # T=1
            )
            egoTin_to_seq_list.append(egoTin_)

        all_cams_imgs   = []
        all_cams_intr   = []
        all_cams_trans  = []
        all_cams_rots   = []

        for cam in self.img_params.cams:
            cam_imgs   = []
            cam_intr   = []
            cam_trans  = []
            cam_rots   = []

            for i, sample in enumerate(batch):
                frame_idx = sample["frame_idx"]

                img = sample[cam]['image_0']
                img_pil = Image.fromarray(img)
                img_tensor = NORMALIZE_IMG(img_pil)

                # TODO resizing to 128x128 is only temporary to save vram on the local gpg
                # TODO this needs to be done for the images and also the intrinsics!
                img_tensor = TF.resize(img_tensor, [128, 128], antialias=True)
                cam_imgs.append(img_tensor)

                intr = torch.tensor(self.camera_params[cam]['intrinsics'], dtype=torch.float32)
                scale_factor = 128.0 / 640.0

                # Scale fx, fy, cx, cy
                intr[0, 0] *= scale_factor
                intr[1, 1] *= scale_factor
                intr[0, 2] *= scale_factor 
                intr[1, 2] *= scale_factor

                cam_intr.append(intr)

                R_body_camera = torch.tensor(self.camera_params[cam]['R_body_camera'], dtype=torch.float32)
                position = torch.zeros((3, 1), dtype=torch.float32)
                cam_rots.append(R_body_camera)
                cam_trans.append(position)

            # stack across the batch dimension
            cam_imgs  = torch.stack(cam_imgs, dim=0)   # [B, 3, H, W]
            cam_intr  = torch.stack(cam_intr, dim=0)   # [B, 3, 3]
            cam_trans = torch.stack(cam_trans, dim=0)  # [B, 3, 1]
            cam_rots  = torch.stack(cam_rots, dim=0)   # [B, 3, 3]

            all_cams_imgs.append(cam_imgs)
            all_cams_intr.append(cam_intr)
            all_cams_trans.append(cam_trans)
            all_cams_rots.append(cam_rots)

        all_cams_imgs   = torch.stack(all_cams_imgs, dim=0)      # [N, B, 3, H, W]
        all_cams_intr   = torch.stack(all_cams_intr, dim=0)      # [N, B, 3, 3]
        all_cams_trans  = torch.stack(all_cams_trans, dim=0)     # [N, B, 3, 1]
        all_cams_rots   = torch.stack(all_cams_rots, dim=0)      # [N, B, 3, 3]

        # bring dimensions into order [B, T=1, N, 3, H, W]
        imgs    = all_cams_imgs.permute(1, 0, 2, 3, 4).unsqueeze(1)
        intrins = all_cams_intr.permute(1, 0, 2, 3).unsqueeze(1)
        trans   = all_cams_trans.permute(1, 0, 2, 3).unsqueeze(1)
        rots    = all_cams_rots.permute(1, 0, 2, 3).unsqueeze(1)

        binimg = torch.stack(binimgs, dim=0).unsqueeze(1)        # [B, 1, 1, 200, 200]
        valid_binimg = torch.stack(valid_binimgs, dim=0).unsqueeze(1)

        egoTin_to_seq = torch.stack(egoTin_to_seq_list, dim=0)   # [B, T=1, 4, 4]
        # egoTin_to_seq = egoTin_to_seq.unsqueeze(1)               # if T=1

        egoTin_to_seq_dummy = torch.eye(4).expand(B, 1, 4, 4).clone() # dummy for now to verify the other stuff works

        bev_aug = torch.eye(4).expand(B, 1, 4, 4).clone() # no augmentation for now

        return {
            "frame_idx": [sample["frame_idx"] for sample in batch],
            "imgs":     imgs,
            "rots":     rots,
            "trans":    trans,
            "intrins":  intrins,
            "bev_aug":  bev_aug,
            "egoTin_to_seq":   egoTin_to_seq_dummy,
            "binimg":          binimg,
            "valid_binimg":    valid_binimg,
        }

    
    @staticmethod
    def poses_to_egoTin_to_seq(pose_entries, frame_indices):
        """
        Compute egoTin_to_seq: transformation from ego frame at each t to the reference frame (t_ref = last in the list).
        
        Args:
            pose_entries: list of [x, y, z, qx, qy, qz, qw] from T_world_camera
            frame_indices: list of T integers indicating the selected frame indices (length T)
        
        Returns:
            egoTin_to_seq: tensor of shape (T, 4, 4), each matrix = T_ref^-1 @ T_i
        """
        poses = [TartangroundDatamodule.pose_vec_to_matrix(pose_entries[i]) for i in frame_indices]
        
        # # Reference pose (usually last)
        # T_ref = poses[-1]
        # T_ref_inv = np.linalg.inv(T_ref)

        # # Compute egoTin_to_seq[i] = T_ref^-1 @ T_i
        # T_rel = [T_ref_inv @ T_i for T_i in poses]
        # return torch.tensor(np.stack(T_rel), dtype=torch.float32)  # shape (T, 4, 4)
        
        # new approach: we just return the relative pose to the global origin
        return torch.tensor(np.stack(poses), dtype=torch.float32)  # shape (T=1, 4, 4)

    
    def pose_vec_to_matrix(pose):
        """Convert [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix"""
        trans = np.array(pose[:3])
        quat = np.array(pose[3:7])  # format [qx, qy, qz, qw]
        rot_mat = Rotation.from_quat(quat).as_matrix() # expects [qx, qy, qz, qw]
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = trans
        return T
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        if "binimg" in batch.keys():
            batch["binimg"] = batch["binimg"].float()
        
        if "egoTin_to_seq" in batch and "egoTout_to_seq" not in batch:
            batch["egoTout_to_seq"] = batch["egoTin_to_seq"].clone()
        
        return batch
    

class IndexWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        # must be a dict so we can add keys
        sample = sample.copy()
        sample["frame_idx"] = idx
        return sample