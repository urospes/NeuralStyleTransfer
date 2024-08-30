import os
import math
import time
import torch
from utils import utils, train


if __name__ == "__main__":

    #VIDEOS_PATH = "videos/training"
    #VIDEOS_FRAMES_DATASET = os.path.join(VIDEOS_PATH, "dataset", "frames")

    #utils.prepare_video_dataset(videos_folder_path=VIDEOS_PATH, dst_path=VIDEOS_FRAMES_DATASET)

    STYLE_IMGS_PATH = "images/style_images"
    STYLE_IMG_NAME = "candy.jpg"
    TRAINING_IMGS = "videos/training"
    DATASET_NAME = "dataset"  # promeni dataset po zelji

    MODELS_PATH = "models"
    utils.mkdirs(dirs=(STYLE_IMGS_PATH, MODELS_PATH))

    TRAINING_IMGS_PATH = os.path.join(TRAINING_IMGS, DATASET_NAME)
    STYLE_IMG_PATH = os.path.join(STYLE_IMGS_PATH, STYLE_IMG_NAME)

    TRAINING_ARGS = {
        "learning_rate": 3e-4,
        "batch_size": 4,
        "epochs": 2,
        "content_w": 1e0,
        "style_w": 5e4,
        "tv_w": 0,
        "temp_w": 5e3
    }

    IMG_SIZE = 256

    MODEL_NAME = f'st_{STYLE_IMG_NAME.split(".")[0]}_tr_{DATASET_NAME}_{str(math.floor(time.time()))}'
    MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
    os.makedirs(MODEL_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train.train(training_imgs_path=TRAINING_IMGS_PATH, style_image_path=STYLE_IMG_PATH,
                training_args=TRAINING_ARGS, img_size=IMG_SIZE, model_path=MODEL_PATH, device=device, use_temporal_loss=True)
