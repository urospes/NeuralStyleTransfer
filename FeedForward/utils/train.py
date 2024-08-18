from typing import Dict, Union
import os
import time
import torch
from torch.optim import Adam
import utils.utils as utils
from neural_nets import transform_net, vgg
from torch.utils.tensorboard import SummaryWriter


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
    total_losses = []
    content_losses = []
    style_losses = []
    tv_losses = []
    writer = SummaryWriter()

    start = time.time()
    for i in range(training_args["epochs"]):
        for batch_id, (batch, _) in enumerate(image_loader):

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

            tv_loss = training_args["tv_w"] * utils.total_variation(stylized_batch)

            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('Loss/content-loss', content_loss.item(), len(image_loader) * i + batch_id + 1)
            writer.add_scalar('Loss/style-loss', style_loss.item(), len(image_loader) * i + batch_id + 1)
            writer.add_scalar('Loss/tv-loss', tv_loss.item(), len(image_loader) * i + batch_id + 1)
            writer.add_scalar('Loss/Total-loss', total_loss.item(), len(image_loader) * i + batch_id + 1)

            total_losses.append(total_loss.item())
            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())
            tv_losses.append(tv_loss.item())

            if batch_id % 20 == 0:
                print(f'Batch {batch_id}')
                print(f'Total loss: {total_loss} |||||| Content loss: {content_loss} |||||| Style loss: {style_loss} |||||| TV Loss: {tv_loss}')

    torch.save(transformer_net.state_dict(), os.path.join(model_path, "params.pth"))

    end = time.time()
    writer.close()
    utils.save_training_info(training_args=training_args, training_time=end-start, total_losses=total_losses,
                             content_losses=content_losses, style_losses=style_losses, tv_losses=tv_losses, model_path=model_path)