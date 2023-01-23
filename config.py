
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/test"   # change val to test in eval mode
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0.5*LAMBDA_CYCLE
NUM_WORKERS = 1
NUM_EPOCHS = 10
LOAD_MODEL = True  # change to true in eval mode
SAVE_MODEL = False # change to false in eval mode
CHECKPOINT_GEN_D = "gend.pth.tar"
CHECKPOINT_GEN_N = "genn.pth.tar"
CHECKPOINT_CRITIC_D = "criticd_640x480_200.pth.tar"
CHECKPOINT_CRITIC_N = "criticn_640x480_200.pth.tar"
HEIGHT = 480
WIDTH = 640

transforms = A.Compose(
    [
        A.Resize(width=WIDTH, height=HEIGHT),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
