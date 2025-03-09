import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "dataset/vangogh2photo"
TRAIN_DIR = DATA_DIR
VAL_DIR = DATA_DIR

BATCH_SIZE = 2
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_GEN_VG = "models/G_vangogh2photo_B2A.pth.tar"
CHECKPOINT_GEN_PH = "models/G_vangogh2photo_A2B.pth.tar"
CHECKPOINT_CRITIC_VG = "models/critic_vangogh.pth.tar"
CHECKPOINT_CRITIC_PH = "models/critic_photo.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

inference_transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)
