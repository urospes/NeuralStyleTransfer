from typing import Dict, Union
import torch
from torch.optim import Adam
import utils.utils as utils
from neural_nets import transform_net, vgg
import matplotlib.pyplot as plt
import numpy as np


def train(training_imgs_path: str, style_image_path: str, img_size: int, training_args: Dict[str, Union[float, int]], model_name: str, device: torch.device):

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

    utils.save_model(transformer_net.eval().cpu(), model_name)

    plot_loss(total_losses, "total")
    plot_loss(content_losses, "content")
    plot_loss(style_losses, "style")


def plot_loss(losses, type):
    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.title(type)
    plt.savefig(f'./loss_graphs/{type}')


def stylize_image(content_image_name, model_name, device: torch.device):

    content_image = utils.prepare_image(f'images/content_images/{content_image_name}')
    content_image = content_image.unsqueeze(0).to(device)

    transformer_net = transform_net.ImageTransformNet()
    state_dict = torch.load(f'models/{model_name}.model')
    transformer_net.load_state_dict(state_dict, strict=True)
    transformer_net.to(device).eval()

    with torch.no_grad():
        stylized_image = transformer_net(content_image).cpu()
        utils.save_image(stylized_image[0], "test.jpg")


if __name__ == "__main__":

    TRAINING_IMGS_PATH = "./images/training/ms_coco_200"
    STYLE_IMG_PATH = "./images/style_images/mosaic.jpg"
    MODEL_NAME = "coco200"

    TRAINING_ARGS = {
        "learning_rate": 1e-5,
        "batch_size": 2,
        "epochs": 2,
        "content_w": 1e0,
        "style_w": 1e4
    }

    IMG_SIZE = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train(training_imgs_path=TRAINING_IMGS_PATH, style_image_path=STYLE_IMG_PATH, training_args=TRAINING_ARGS, img_size=IMG_SIZE, model_name=MODEL_NAME, device=device)
    # stylize_image("amber.jpg", MODEL_NAME, device=device)

    #img = utils.prepare_image("./images/style_images/mosaic.jpg")
    #img.show()
    #utils.save_image(img, "./images/output/asd.jpg")
