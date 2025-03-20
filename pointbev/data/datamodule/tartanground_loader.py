""" 
Author: Ryan Slocum

Datamodule for the Tartanground dataset.
"""


from typing import Optional

import pytorch_lightning as pl
import torch
from tartanair import TartanAirDataset
from torch.utils.data import random_split


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
    ):
        super().__init__()

        # Nuscenes
        self.envs = envs
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
                                  camera_name=self.img_params.cams)
        self.train_data, self.val_data = random_split(data.dataset, [int(0.8*len(data.dataset)), int(0.2*len(data.dataset))])
        

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=self.train_drop_last,
            # worker_init_fn=worker_rnd_init,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.valid_batch_size,
            shuffle=self.val_shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor
        )

    # def test_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.valdata,
    #         batch_size=self.valid_batch_size,
    #         shuffle=False,
    #         drop_last=False,
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #         prefetch_factor=self.prefetch_factor,
    #         collate_fn=self.collate_fn,
    #     )

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
