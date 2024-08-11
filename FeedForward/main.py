from typing import Dict, Union
import os
import time
import torch
from torch.optim import Adam
import utils.utils as utils
from neural_nets import transform_net, vgg


def train(training_imgs_path: str, style_image_path: str, img_size: int, training_args: Dict[str, Union[float, int]], model_path: str, device: torch.device):

    transformer_net = transform_net.ImageTransformNet().train().to(device)
    loss_net = vgg.Vgg16().to(device)
    optimizer = Adam(params=transformer_net.parameters(), lr=training_args["learning_rate"])

    style_img = utils.prepare_image(style_image_path, img_size=img_size)
    style_img = style_img.repeat(training_args["batch_size"], 1, 1, 1).to(device)

    features = loss_net(style_img)
    features_style = [utils.gram_matrix(x) for x in features]

    img_transformer = utils.get_training_transformer(img_size=img_size)
    image_loader = utils.get_data_loader(folder_path=training_imgs_path, transformer=img_transformer, batch_size=training_args["batch_size"])

    # training loop
    start = time.time()
    total_losses = []
    content_losses = []
    style_losses = []
    for _ in range(training_args["epochs"]):
        for i, (batch, _) in enumerate(image_loader):

            batch = batch.to(device)
            stylized_batch = transformer_net(batch)

            features_batch = loss_net(batch)
            features_stylized_batch = loss_net(stylized_batch)

            content_loss = training_args["content_w"] * torch.nn.MSELoss()(features_batch.relu_2_2, features_stylized_batch.relu_2_2)

            style_loss = 0.
            features_style_input = [utils.gram_matrix(x) for x in features_stylized_batch]
            for gram_target, gram_current in zip(features_style, features_style_input):
                style_loss += torch.nn.MSELoss()(gram_target, gram_current)
            style_loss = training_args["style_w"] * style_loss

            total_loss = content_loss + style_loss  # total variation loss should be added
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_losses.append(total_loss.item())
            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())

            if i % 20 == 0:
                print(f'Batch {i}')
                print(f'Total loss: {total_loss} |||||| Content loss: {content_loss} |||||| Style loss: {style_loss}')

    torch.save(transformer_net.state_dict(), os.path.join(model_path, "params.model"))

    end = time.time()

    utils.save_training_info(training_args=training_args, training_time=end-start, total_losses=total_losses,
                             content_losses=content_losses, style_losses=style_losses, model_path=model_path)


def stylize_image(content_image_path: str, model_path: str, output_image_path: str, device: torch.device):

    content_image = utils.prepare_image(content_image_path)
    content_image = content_image.unsqueeze(0).to(device)

    transformer_net = transform_net.ImageTransformNet()
    state_dict = torch.load(model_path)
    transformer_net.load_state_dict(state_dict, strict=True)
    transformer_net.to(device).eval()

    with torch.no_grad():
        stylized_image = transformer_net(content_image)
        stylized_image = utils.post_process_image(stylized_image.cpu().numpy()[0])
        utils.save_image(stylized_image, output_image_path)


if __name__ == "__main__":

    TRAINING_MODE = False   # Biramo jel hocemo i trening ili samo transofrmaciju slika vec postojecim modelom

    STYLE_IMGS_PATH = "images/style_images"
    CONTENT_IMGS_PATH = "images/content_images"
    OUTPUT_IMGS_PATH = "images/output_images"
    MODELS_PATH = "models"
    utils.mkdirs(dirs=(STYLE_IMGS_PATH, CONTENT_IMGS_PATH, OUTPUT_IMGS_PATH, MODELS_PATH))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if TRAINING_MODE:
        STYLE_IMG_NAME = "mosaic.jpg"
        TRAINING_IMGS = "images/training"
        DATASET_NAME = "ms_coco_200"    # promeni dataset po zelji

        TRAINING_IMGS_PATH = os.path.join(TRAINING_IMGS, DATASET_NAME)
        STYLE_IMG_PATH = os.path.join(STYLE_IMGS_PATH, STYLE_IMG_NAME)

        #   trening parametri (najcesce je potrebnno eksperimentisati sa LR, CW i SW) batch size probati sto veci moguci
        #   broj epoha neka ostane na 2
        TRAINING_ARGS = {
            "learning_rate": 1e-4,
            "batch_size": 2,
            "epochs": 1,
            "content_w": 1e0,
            "style_w": 1e4
        }

        IMG_SIZE = 512

        MODEL_NAME = f'st_{STYLE_IMG_NAME.split(".")[0]}_tr_{DATASET_NAME}'
        MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
        os.makedirs(MODEL_PATH, exist_ok=True)

        train(training_imgs_path=TRAINING_IMGS_PATH, style_image_path=STYLE_IMG_PATH,
              training_args=TRAINING_ARGS, img_size=IMG_SIZE, model_path=MODEL_PATH, device=device)

    else:
        CONTENT_IMAGE_NAME = "amber.jpg"    # ovim parametrom navodimo koju sliku zelimo da stilizujemo
        MODEL_NAME = "coco10k_model"    # navodimo ime vec istreniranog modela, konfigurisati po zelji

        OUTPUT_IMG_NAME = f'c_{CONTENT_IMAGE_NAME.split(".")[0]}_m_{MODEL_NAME.split(".")[0]}.jpg'

        MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME, "params.model")
        CONTENT_IMG_PATH = os.path.join(CONTENT_IMGS_PATH, CONTENT_IMAGE_NAME)
        OUTPUT_IMG_PATH = os.path.join(OUTPUT_IMGS_PATH, OUTPUT_IMG_NAME)

        stylize_image(CONTENT_IMG_PATH, MODEL_PATH, OUTPUT_IMG_PATH, device=device)