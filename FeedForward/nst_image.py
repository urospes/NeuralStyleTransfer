import os
import math
import time
import torch
from utils import utils, train, stylize

if __name__ == "__main__":

    TRAINING_MODE = False  # Biramo jel hocemo i trening ili samo transofrmaciju slika vec postojecim modelom

    STYLE_IMGS_PATH = "images/style_images"
    CONTENT_IMGS_PATH = "images/content_images"
    OUTPUT_IMGS_PATH = "images/output_images"
    MODELS_PATH = "models"
    utils.mkdirs(dirs=(STYLE_IMGS_PATH, CONTENT_IMGS_PATH, OUTPUT_IMGS_PATH, MODELS_PATH))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if TRAINING_MODE:
        STYLE_IMG_NAME = "starry_night.jpg"
        #TRAINING_IMGS = "images/training"
        TRAINING_IMGS = "videos/training"
        DATASET_NAME = "dataset"  # promeni dataset po zelji

        TRAINING_IMGS_PATH = os.path.join(TRAINING_IMGS, DATASET_NAME)
        STYLE_IMG_PATH = os.path.join(STYLE_IMGS_PATH, STYLE_IMG_NAME)

        #   trening parametri (najcesce je potrebnno eksperimentisati sa LR, CW i SW) batch size probati sto veci moguci
        #   broj epoha neka ostane na 2
        TRAINING_ARGS = {
            "learning_rate": 3e-4,
            "batch_size": 4,
            "epochs": 2,
            "content_w": 1e0,
            "style_w": 5e4,
            "tv_w": 0
        }

        IMG_SIZE = 256

        MODEL_NAME = f'st_{STYLE_IMG_NAME.split(".")[0]}_tr_{DATASET_NAME}_{str(math.floor(time.time()))}'
        MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
        os.makedirs(MODEL_PATH, exist_ok=True)

        train.train(training_imgs_path=TRAINING_IMGS_PATH, style_image_path=STYLE_IMG_PATH,
                    training_args=TRAINING_ARGS, img_size=IMG_SIZE, model_path=MODEL_PATH, device=device)

    else:
        CONTENT_IMAGE_NAME = "norica.jpeg"  # ovim parametrom navodimo koju sliku zelimo da stilizujemo
        MODEL_NAME = "st_mosaic_tr_ms_coco_40k_1723663921"  # navodimo ime vec istreniranog modela, konfigurisati po zelji

        OUTPUT_IMG_NAME = f'c_{CONTENT_IMAGE_NAME.split(".")[0]}_m_{MODEL_NAME.split(".")[0]}.jpg'

        MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME, "params.pth")
        CONTENT_IMG_PATH = os.path.join(CONTENT_IMGS_PATH, CONTENT_IMAGE_NAME)
        OUTPUT_IMG_PATH = os.path.join(OUTPUT_IMGS_PATH, OUTPUT_IMG_NAME)

        stylize.stylize_image(CONTENT_IMG_PATH, MODEL_PATH, OUTPUT_IMG_PATH, device=device)
