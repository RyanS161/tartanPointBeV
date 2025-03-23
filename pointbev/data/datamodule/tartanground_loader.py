""" 
Author: Ryan Slocum

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
        train_len = int(0.8*len(data.dataset))
        self.train_data, self.val_data = random_split(data.dataset, [train_len, len(data.dataset)-train_len])

        # TODO could probably achieve that this is integrated into the data.dataset, but then need to adapt the 'create_image_dataset' func
        # Load camera intrinsics from JSON files
        self.camera_params = {}
        env = self.envs[0]  # Assuming all cameras share same params across environments
        difficulty = "Data_easy"  # hardcoded for now. Can parameterize in tartanground.yaml later
        traj = "P0006"  # hardcoded for now. Can parameterize in tartanground.yaml later

        for cam in self.img_params.cams: # iterates over lcam_front, back, left, right, top, bottom            
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
                f"image_{cam}_pinhole.txt"
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
    'rots': tensor(B, N, 3, 3),
    'trans': tensor(B, N, 3),
    'intrins': tensor(B, N, 3, 3),
    'bev_aug': tensor(B, 2, 3),
    'egoTin_to_seq': tensor(B, 4, 4)
    }  
    """
    def collate_fn(self, batch):
        """
        Convert list of samples → batch dict that matches PointBEV forward() inputs.
        """
        B = len(batch) # batch size
        N = len(self.img_params.cams)  # number of cameras
        imgs_list = []
        intrins_list = []
        trans_list = []
        rots_list = []

        for cam in self.img_params.cams:
            cam_imgs = []
            cam_trans = []
            cam_rots = []

            for sample in batch:
                img = sample[cam]['image_0']
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                cam_imgs.append(img_tensor)

                frame_idx = sample['frame_idx']
                position = torch.tensor(self.camera_params[cam]['T_world_camera'][frame_idx][:3], dtype=torch.float32)
                cam_trans.append(position)

                # TODO function is assuming order [qx, q,y, qz, qw]. Verify that this is actually the case in the txt's
                quat_rotation_world_camera = self.camera_params[cam]['T_world_camera'][frame_idx][3:7]
                matrix_rotation_world_camera = Rotation.from_quat(quat_rotation_world_camera).as_matrix()
                matrix_rotation_world_camera = torch.from_numpy(matrix_rotation_world_camera).float()
                # TODO not sure if we need to transform frame from camera frame to body frame to world frame
                R_body_camera = torch.tensor(self.camera_params[cam]['R_body_camera'], dtype=torch.float32) # converting R_body_camera to tensor and multiplying it
                combined_rotation_matrix = torch.matmul(matrix_rotation_world_camera, R_body_camera)
                cam_rots.append(combined_rotation_matrix)


            imgs_list.append(torch.stack(cam_imgs))  # B x 3 x H x W

            intrins_tensor = torch.tensor(self.camera_params[cam]['intrinsics'], 
                                          dtype=torch.float32
                                          )
            # Repeat for batch size
            intrins_list.append(intrins_tensor.unsqueeze(0).repeat(B, 1, 1))
            trans_list.append(torch.stack(cam_trans))  # [B, 3]
            rots_list.append(torch.stack(cam_rots))  # [B, 3, 3]

        imgs = torch.stack(imgs_list, dim=0).permute(1, 0, 2, 3, 4).unsqueeze(1)  # B x T=1 x N x C=3 x H x W
        intrins = torch.stack(intrins_list, dim=1)  # [B, N, 3, 3]
        trans = torch.stack(trans_list, dim=1)  # [B, N, 3]
        rots = torch.stack(rots_list, dim=1)  # [B, N, 3, 3]
        

        # Dummy camera intrinsics/extrinsics for now
        # H, W = img_tensor.shape[1:]
        # fx = fy = 320
        # cx = cy = W // 2
        # intrinsic_matrix = torch.tensor([[fx, 0, cx],
        #                                 [0, fy, cy],
        #                                   [0, 0, 1]], dtype=torch.float32)
        # rots = torch.stack([torch.eye(3).clone() for _ in range(B * N)]).view(B, N, 3, 3)
        # intrins = torch.stack([intrinsic_matrix.clone() for _ in range(B * N)]).view(B, N, 3, 3)
        # rots = torch.stack([torch.eye(3).clone() for _ in range(B * N)]).view(B, N, 3, 3) # no rotation for now

        # TODO rot with static and world camera pose
        bev_aug = torch.stack([torch.eye(2, 3).clone() for _ in range(B)])  # (B, 2, 3) no aug for now
        egoTin_to_seq = torch.stack([torch.eye(4).clone() for _ in range(B)]).unsqueeze(1)  # (B, T=1, 4, 4) static camera for now

        return {
            "imgs": imgs,
            "rots": rots,
            "trans": trans,
            "intrins": intrins,
            "bev_aug": bev_aug,
            "egoTin_to_seq": egoTin_to_seq,
        }

    
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