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
        wrapped_dataset = IndexWrapper(data.dataset) # Wrap the dataset to add indices

        self.train_data = wrapped_dataset  # Use all data for training
        val_len = min(250, len(wrapped_dataset)//5)  # Use ~20% or max 250 samples
        self.val_data = torch.utils.data.Subset(wrapped_dataset, list(range(val_len)))
        print(f"Training set size: {len(self.train_data)}, Validation set size: {len(self.val_data)}")


        # TODO could probably achieve that this is integrated into the data.dataset, but then need to adapt the 'create_image_dataset' func
        self.camera_params = {}
        env = self.envs[0]  # Assuming all cameras share same params across environments
        difficulty = "Data_easy"  # hardcoded for now. Can parameterize in tartanground.yaml later
        traj = "P0006"  # hardcoded for now. Can parameterize in tartanground.yaml later

        for cam in self.img_params.cams: # iterates over lcam_front, front_right, front_left, back_right, back_left back            
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
    'imgs': {cam_name: tensor(B, 3, H, W)},   # camera → batch of images
    'rots': tensor(B, N, 3, 3),     seems to be the static transformation of the camera to base frame
    'trans': tensor(B, N, 3),       seems to be the static transformation of the camera to the base frame
    'intrins': tensor(B, N, 3, 3),
    'bev_aug': tensor(B, 2, 3),
    'egoTin_to_seq': tensor(B, 4, 4)
    }  
    """
    def collate_fn(self, batch):
        """
        Convert list of samples → batch dict that matches PointBEV forward() inputs.
        """
        # print("batch frame indices:", [sample["frame_idx"] for sample in batch])
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
                # binimg_np = np.array(Image.open(binimg_file))
                binimg_np = np.load(binimg_file)
                # print(f"{binimg_file} → shape={binimg_np.shape}, dtype={binimg_np.dtype}, unique={np.unique(binimg_np)}")
                binimg_ = torch.from_numpy(binimg_np).float().unsqueeze(0) # / 255.0
            else:
                print(f"{binimg_file} not found.")
                binimg_ = torch.zeros((1, 100, 100), dtype=torch.float32)

            if os.path.exists(valid_binimg_file):
                # valid_np = np.array(Image.open(valid_binimg_file))
                valid_np = np.load(valid_binimg_file)
                # print(f"{valid_binimg_file} → shape={valid_np.shape}, dtype={valid_np.dtype}, unique={np.unique(valid_np)}")
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
                # The dictionary for this sample, for this camera
                frame_idx = sample["frame_idx"]  # again so you know which sample

                img = sample[cam]['image_0']
                img_pil = Image.fromarray(img)
                img_tensor = NORMALIZE_IMG(img_pil)
                # img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 # TODO IS NORMALIZING NEEDED SINCE THEY HAVE SOME PARAM CALLED 'normalize_img = True'
                # If you want smaller size:
                img_tensor = TF.resize(img_tensor, [128, 128], antialias=True)
                cam_imgs.append(img_tensor)

                intr = torch.tensor(self.camera_params[cam]['intrinsics'], dtype=torch.float32)
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

        bev_aug = torch.eye(4).expand(B, 1, 4, 4).clone() # no augmentation for now

        return {
            "frame_idx": [sample["frame_idx"] for sample in batch],
            "imgs":     imgs,
            "rots":     rots,
            "trans":    trans,
            "intrins":  intrins,
            "bev_aug":  bev_aug,
            "egoTin_to_seq":   egoTin_to_seq,
            "binimg":          binimg,
            "valid_binimg":    valid_binimg,
        }


        # for cam in self.img_params.cams:
        #     cam_imgs = []
        #     cam_trans = []
        #     cam_rots = []

        #     # TODO the i won't be correct anymore if the batches are shuffled sometime. Identifyer needs to be encoded in the batches itself
        #     for i, sample in enumerate(batch):
        #         frame_idx = i
        #         img = sample[cam]['image_0']
        #         img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        #         img_tensor = TF.resize(img_tensor, [128, 128], antialias=True)  # or [256, 256] if you prefer    # TODO only temporary to save vram on the local gpu
        #         cam_imgs.append(img_tensor)

        #         # position = torch.tensor(self.camera_params[cam]['T_world_camera'][frame_idx][:3], dtype=torch.float32)
        #         position = torch.zeros((3, 1), dtype=torch.float32) # Assuming the l_cam is the base frame
        #         cam_trans.append(position)

        #         # TODO function is assuming order [qx, q,y, qz, qw]. Verify that this is actually the case in the txt's
        #         # quat_rotation_world_camera = self.camera_params[cam]['T_world_camera'][frame_idx][3:7]
        #         # matrix_rotation_world_camera = Rotation.from_quat(quat_rotation_world_camera).as_matrix()
        #         # matrix_rotation_world_camera = torch.from_numpy(matrix_rotation_world_camera).float()
        #         # TODO not sure if we need to transform frame from camera frame to body frame to world frame
        #         # R_body_camera = torch.tensor(self.camera_params[cam]['R_body_camera'], dtype=torch.float32) # converting R_body_camera to tensor
        #         # combined_rotation_matrix = torch.matmul(matrix_rotation_world_camera, R_body_camera)
        #         R_body_camera = torch.tensor(self.camera_params[cam]['R_body_camera'], dtype=torch.float32) # converting R_body_camera to tensor
        #         cam_rots.append(R_body_camera)

        #         if cam == 'lcam_front':
        #             egoTin_to_seq = TartangroundDatamodule.poses_to_egoTin_to_seq(self.camera_params[cam]['T_world_camera'], [frame_idx]) # (T=1, 4, 4)
        #             egoTin_to_seq_list.append(egoTin_to_seq)

        #             binimg_file = os.path.join(self.binimg_path, f"{frame_idx:06d}.png")
        #             valid_binimg_file = os.path.join(self.valid_binimg_path, f"{frame_idx:06d}.png")

        #             # Load binary mask if exists
        #             if os.path.exists(binimg_file):
        #                 img_np = np.array(Image.open(binimg_file))
        #                 binimg = torch.from_numpy(img_np).float().unsqueeze(0)
        #             else:
        #                 binimg = torch.zeros((1, 200, 200), dtype=torch.float32)


        #             # Load valid mask if exists, otherwise use all ones or zeros based on your needs
        #             if os.path.exists(valid_binimg_file):
        #                 valid_binimg = torch.from_numpy(
        #                     np.array(Image.open(valid_binimg_file))
        #                 ).bool().unsqueeze(0)
        #             else:
        #                 valid_binimg = torch.zeros((1, 200, 200), dtype=torch.float32)
            

        #             binimg = TF.resize(binimg, [200, 200], antialias=True)
        #             valid_binimg = TF.resize(valid_binimg.float(), [200, 200], antialias=True).bool()
        #             binimgs.append(binimg)
        #             valid_binimgs.append(valid_binimg)
        
        

        #     imgs_list.append(torch.stack(cam_imgs))  # B x 3 x H x W

        #     intrins_tensor = torch.tensor(self.camera_params[cam]['intrinsics'], 
        #                                   dtype=torch.float32
        #                                   )
        #     # Repeat for batch size
        #     intrins_list.append(intrins_tensor.unsqueeze(0).repeat(B, 1, 1))
        #     trans_list.append(torch.stack(cam_trans))  # [B, 3, 1]
        #     rots_list.append(torch.stack(cam_rots))  # [B, 3, 3]

        # imgs = torch.stack(imgs_list, dim=0).permute(1, 0, 2, 3, 4).unsqueeze(1)  # [B, T=1, N, C=3, H, W]
        # intrins = torch.stack(intrins_list, dim=1).unsqueeze(1)  # [B, T=1, N, 3, 3]
        # trans = torch.stack(trans_list, dim=1).unsqueeze(1)  # [B, T=1, N, 3, 1]
        # rots = torch.stack(rots_list, dim=1).unsqueeze(1)  # [B, T=1, N, 3, 3]
        # egoTin_to_seq = torch.stack(egoTin_to_seq_list, dim=0)  # (B, T=1, 4, 4)


        # bev_aug = torch.stack([torch.eye(4).clone() for _ in range(B)]).unsqueeze(1)  # (B, T=1, 4, 4) no aug
        # # egoTin_to_seq = torch.stack([torch.eye(4).clone() for _ in range(B)]).unsqueeze(1)  # (B, T=1, 4, 4) for T=1 we consider only current pose

        # # binimg = torch.zeros(B, 1, 200, 200).unsqueeze(1)  # (B, T=1, H, W) no binimg for now. 200 is hardcoded as its the size of the inputs
        # # valid_binimg = torch.zeros(B, 1, 200, 200).unsqueeze(1)  # (B, T=1, H, W) no binimg for now. 200 is hardcoded as its the size of the inputs
        # # binimg should be float for BCE loss
        # # binimg = torch.zeros(B, 1, 200, 200, dtype=torch.float32).unsqueeze(1)
        # # valid_binimg = torch.zeros(B, 1, 200, 200, dtype=torch.bool).unsqueeze(1)
        # binimg = torch.stack(binimgs).unsqueeze(1)  # [B=1, T=1, C=1, H, W]
        # valid_binimg = torch.stack(valid_binimgs).unsqueeze(1)  # [B=1, T=1, C=1, H, W]
    

        # return {
        #     "imgs": imgs,
        #     "rots": rots,
        #     "trans": trans,
        #     "intrins": intrins,
        #     "bev_aug": bev_aug,
        #     "egoTin_to_seq": egoTin_to_seq,
        #     # task specific labels:
        #     "binimg": binimg,
        #     "valid_binimg": valid_binimg
        # }
    
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
        # Convert binimg to float if it exists (needed for loss calculation)
        if "binimg" in batch.keys():
            batch["binimg"] = batch["binimg"].float()
        
        # Add egoTout_to_seq tensor which is expected by the inference code
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


    
    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     for key in ["binimg", "binimg_aug"]:
    #         if key in batch.keys():
    #             # Some outputs are stored as int, but we need them as float for the loss.
    #             batch[key] = batch[key].float()

    #     # Object detection activated
    #     if self.keep_input_detection:
    #         batch["classes"] = [[elem for elem in b] for b in batch["classes"]]
    #         batch["classes_aug"] = [[elem for elem in b] for b in batch["classes_aug"]]

    #         batch["bbox_attr"] = [[elem for elem in b] for b in batch["bbox_attr"]]
    #         batch["bbox_attr_aug"] = [
    #             [elem for elem in b] for b in batch["bbox_attr_aug"]
    #         ]

    #         batch["centers"] = [[elem for elem in b] for b in batch["centers"]]
    #         batch["centers_aug"] = [[elem for elem in b] for b in batch["centers_aug"]]

    #     # HDMaps
    #     if self.keep_input_hdmap:
    #         batch["hdmap"] = batch["hdmap"].float()

    #     if self.keep_input_offsets_map:
    #         batch["offsets_map_dist"] = torch.sqrt(
    #             batch["offsets_map"][:, :, 0] ** 2 + batch["offsets_map"][:, :, 1] ** 2
    #         ).unsqueeze(2)
    #         batch["offsets_map_dist_aug"] = torch.sqrt(
    #             batch["offsets_map_aug"][:, :, 0] ** 2
    #             + batch["offsets_map_aug"][:, :, 1] ** 2
    #         ).unsqueeze(2)

    #     return batch


# def worker_rnd_init(x):
#     np.random.seed(13 + x)


# def collate_batch(batch: List[Tensor]):
#     key_as_list_of_tensor = [
#         "classes",
#         "classes_aug",
#         "bbox_attr",
#         "bbox_attr_aug",
#         "centers",
#         "centers_aug",
#         "bboxes",
#         "bboxes_aug",
#         "bbox_egopose",
#         "bbox_egopose_aug",
#         "tokens",
#     ]
#     keys = batch[0].keys()
#     out_dict = {
#         k: torch.stack([b[k] for b in batch])
#         for k in keys
#         if k not in key_as_list_of_tensor
#     }

#     for k in key_as_list_of_tensor:
#         if k in keys:
#             out_dict.update({k: [b[k] for b in batch]})
#     return out_dict