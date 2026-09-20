import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phase4_physics_dataset import Phase4PhysicsDataset
from phase4_physics_model import Phase4PhysicsResNet18


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_DIR = os.path.join(
    PROJECT_ROOT,
    "Train",
    "train_images"
)

TRAIN_METADATA = os.path.join(
    PROJECT_ROOT,
    "Train",
    "train_metadata.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "final_physics_model.pth"
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
NUM_EPOCHS = 2

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

# Full training set:
# Depth = 2854
# Rise  = 5000
POS_WEIGHT = 2854 / 5000


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Load full metadata
# ============================================================

train_df = pd.read_csv(
    TRAIN_METADATA
)

print()
print("Total training samples:", len(train_df))

print()
print("Class distribution:")
print(
    train_df["label"]
    .value_counts()
    .sort_index()
)

# Verify expected dataset size
assert len(train_df) == 7854

# Verify labels
assert set(
    train_df["label"].unique()
) == {0, 1}

depth_count = (
    train_df["label"] == 0
).sum()

rise_count = (
    train_df["label"] == 1
).sum()

pos_weight = (
    depth_count / rise_count
)

print()
print(
    f"Depth samples: {depth_count}"
)

print(
    f"Rise samples: {rise_count}"
)

print(
    f"pos_weight: {pos_weight:.6f}"
)


# ============================================================
# Dataset
# ============================================================

train_dataset = Phase4PhysicsDataset(
    train_df,
    TRAIN_DIR,
    has_labels=True,
    augment=True,
)


# ============================================================
# DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# Model
# ============================================================

model = Phase4PhysicsResNet18(
    pretrained=True
).to(device)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print()
print(
    "Model parameters:",
    total_params
)


# ============================================================
# Loss
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [pos_weight],
        dtype=torch.float32,
        device=device
    )
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# AMP
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ============================================================
# Training
# ============================================================

print()
print("=" * 60)
print("FINAL TRAINING")
print("=" * 60)

for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    sample_count = 0

    for images, _, _, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.float().to(
            device,
            non_blocking=True
        ).unsqueeze(1)

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        sample_count += batch_size

    epoch_loss = (
        running_loss /
        sample_count
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {epoch_loss:.4f}"
    )


# ============================================================
# Save final model
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "epochs": NUM_EPOCHS,
        "pos_weight": pos_weight,
        "threshold": 0.37,
        "gradient_scale": 1.039898,
        "architecture": "Phase4PhysicsResNet18",
    },
    CHECKPOINT_PATH
)

print()
print("=" * 60)
print("FINAL TRAINING COMPLETE")
print("=" * 60)

print(
    "Checkpoint:"
)

print(
    CHECKPOINT_PATH
)