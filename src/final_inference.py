import os
import numpy as np
import pandas as pd
import cv2
import torch

from phase4_physics_model import Phase4PhysicsResNet18
from sun_utils import rotate_for_sun


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEST_DIR = os.path.join(
    PROJECT_ROOT,
    "Test",
    "eval_images"
)

TEST_METADATA = os.path.join(
    PROJECT_ROOT,
    "Test",
    "test_metadata.csv"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "final_physics_model.pth"
)

SUBMISSION_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "submission.csv"
)

GRADIENT_SCALE = 1.039898
THRESHOLD = 0.37
BATCH_SIZE = 32


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
# Load test metadata
# ============================================================

test_df = pd.read_csv(
    TEST_METADATA
)

print()
print(
    "Evaluation samples:",
    len(test_df)
)

assert len(test_df) == 2000

assert list(
    test_df.columns
) == [
    "image_id",
    "sun_azimuth_angle",
]


# ============================================================
# Verify evaluation files
# ============================================================

metadata_ids = test_df[
    "image_id"
].tolist()

disk_files = sorted(
    [
        filename
        for filename in os.listdir(TEST_DIR)
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]
)

print(
    "Image files:",
    len(disk_files)
)

assert len(disk_files) == 2000

assert set(metadata_ids) == set(
    disk_files
)

assert len(set(metadata_ids)) == 2000


# ============================================================
# Load model
# ============================================================

model = Phase4PhysicsResNet18(
    pretrained=True
).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print()
print(
    "Checkpoint:",
    CHECKPOINT_PATH
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Gradient scale:",
    GRADIENT_SCALE
)


# ============================================================
# Build physics tensor
# ============================================================

def make_physics_tensor(
    image,
    sun_azimuth
):
    """
    Create:

        channel 0 = sun-normalized image
        channel 1 = signed sun-relative gradient

    After mandated sun normalization,
    the modeling reference direction is +x.
    Therefore the gradient channel is Sobel dx.
    """

    image = rotate_for_sun(
        image,
        sun_azimuth
    )

    image = (
        image.astype(np.float32)
        / 255.0
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    gradient = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gradient = (
        gradient
        / GRADIENT_SCALE
    )

    gradient = np.clip(
        gradient,
        -3.0,
        3.0
    )

    tensor = np.stack(
        [
            image,
            gradient,
        ],
        axis=0
    )

    return torch.from_numpy(
        tensor.astype(np.float32)
    )


# ============================================================
# Inference
# ============================================================

predictions = []

print()
print("=" * 60)
print("FINAL INFERENCE")
print("=" * 60)

with torch.no_grad():

    for start in range(
        0,
        len(test_df),
        BATCH_SIZE
    ):

        batch_df = test_df.iloc[
            start:start + BATCH_SIZE
        ]

        batch_tensors = []

        for _, row in batch_df.iterrows():

            image_id = row[
                "image_id"
            ]

            sun_azimuth = float(
                row[
                    "sun_azimuth_angle"
                ]
            )

            image_path = os.path.join(
                TEST_DIR,
                image_id
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                raise FileNotFoundError(
                    f"Could not read: "
                    f"{image_path}"
                )

            if image.shape != (
                256,
                256
            ):
                raise ValueError(
                    f"Unexpected shape for "
                    f"{image_id}: "
                    f"{image.shape}"
                )

            tensor = make_physics_tensor(
                image,
                sun_azimuth
            )

            batch_tensors.append(
                tensor
            )

        batch = torch.stack(
            batch_tensors
        ).to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            logits = model(batch)

        probabilities = torch.sigmoid(
            logits
        ).squeeze(1)

        batch_predictions = (
            probabilities >= THRESHOLD
        ).long().cpu().numpy()

        predictions.extend(
            batch_predictions.tolist()
        )

        processed = min(
            start + BATCH_SIZE,
            len(test_df)
        )

        print(
            f"Processed "
            f"{processed}/"
            f"{len(test_df)}"
        )


# ============================================================
# Final validation
# ============================================================

predictions = np.array(
    predictions,
    dtype=np.int64
)

print()
print(
    "Prediction count:",
    len(predictions)
)

assert len(predictions) == 2000

assert set(
    predictions.tolist()
).issubset({0, 1})


# ============================================================
# Create submission
# ============================================================

submission = pd.DataFrame({
    "image_id": test_df[
        "image_id"
    ],
    "label": predictions,
})


# ============================================================
# Submission integrity checks
# ============================================================

assert list(
    submission.columns
) == [
    "image_id",
    "label",
]

assert len(submission) == 2000

assert submission[
    "image_id"
].isna().sum() == 0

assert submission[
    "label"
].isna().sum() == 0

assert submission[
    "image_id"
].duplicated().sum() == 0

assert set(
    submission["label"].unique()
).issubset({0, 1})

assert (
    submission["image_id"].tolist()
    ==
    test_df["image_id"].tolist()
)


# ============================================================
# Save
# ============================================================

submission.to_csv(
    SUBMISSION_PATH,
    index=False
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 60)
print("SUBMISSION CREATED")
print("=" * 60)

print(
    "Path:",
    SUBMISSION_PATH
)

print(
    "Rows:",
    len(submission)
)

print()
print("Prediction distribution:")
print(
    submission[
        "label"
    ].value_counts().sort_index()
)

print()
print("First 10 rows:")
print(
    submission.head(10).to_string(
        index=False
    )
)

print()
print(
    "ALL SUBMISSION CHECKS PASSED"
)