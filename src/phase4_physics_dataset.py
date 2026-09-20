import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sun_utils import rotate_for_sun


GRADIENT_SCALE = 1.039898


class Phase4PhysicsDataset(Dataset):
    def __init__(
        self,
        metadata,
        image_dir,
        has_labels=True,
        augment=False,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = image_dir
        self.has_labels = has_labels
        self.augment = augment

        self.transform = A.Compose([
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent={
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                },
                rotate=0,
                shear=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.15,
                p=0.5,
            ),
            A.GaussNoise(
                std_range=(0.01, 0.03),
                mean_range=(0.0, 0.0),
                p=0.25,
            ),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        image_id = row["image_id"]
        sun_azimuth = float(row["sun_azimuth_angle"])

        image_path = os.path.join(
            self.image_dir,
            image_id
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        if image.shape != (256, 256):
            raise ValueError(
                f"Unexpected image shape for {image_id}: "
                f"{image.shape}"
            )

        # --------------------------------------------------
        # 1. Normalize illumination orientation
        # Competition-mandated rotation: -sun_azimuth_angle
        # --------------------------------------------------
        image = rotate_for_sun(
            image,
            sun_azimuth
        )

        # Convert to [0, 1]
        image = image.astype(np.float32) / 255.0

        # --------------------------------------------------
        # 2. Safe augmentation
        # Applied after sun normalization.
        # No arbitrary rotation / flips.
        # --------------------------------------------------
        if self.augment:
            transformed = self.transform(
                image=image
            )
            image = transformed["image"]

        image = np.clip(
            image,
            0.0,
            1.0
        ).astype(np.float32)

        # --------------------------------------------------
        # 3. Physics-informed gradient
        #
        # After canonicalization, the modeling reference
        # sun direction is +x (right).
        #
        # Therefore:
        #     dI/du = dI/dx
        #
        # Sobel dx gives the signed directional gradient.
        # --------------------------------------------------
        gradient = cv2.Sobel(
            image,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        # Fixed scale calibrated only from the Phase 3K
        # training split.
        gradient = gradient / GRADIENT_SCALE

        gradient = np.clip(
            gradient,
            -3.0,
            3.0
        )

        # --------------------------------------------------
        # 4. Stack channels
        # [2, 256, 256]
        # --------------------------------------------------
        image_tensor = torch.from_numpy(
            image
        ).unsqueeze(0)

        gradient_tensor = torch.from_numpy(
            gradient
        ).unsqueeze(0)

        tensor = torch.cat(
            [
                image_tensor,
                gradient_tensor
            ],
            dim=0
        )

        if self.has_labels:
            label = int(row["label"])
            return (
                tensor,
                sun_azimuth,
                image_id,
                label,
            )

        return (
            tensor,
            sun_azimuth,
            image_id,
        )