from typing import Dict, Union
import os
import time
import torch
from torch.optim import Adam
import utils.utils as utils
from neural_nets import transform_net, vgg
from torch.utils.tensorboard import SummaryWriter
import cv2


def train(training_imgs_path: str, style_image_path: str, img_size: int, training_args: Dict[str, Union[float, int]], model_path: str, device: torch.device, use_temporal_loss: bool = False):

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

    # video training with temporal loss
    temporal_losses = [] if use_temporal_loss else None
    deep_flow = cv2.optflow.createOptFlow_DeepFlow() if use_temporal_loss else None

    writer = SummaryWriter()

    start = time.time()
    for i in range(training_args["epochs"]):
        for (batch_id, (batch, _)) in enumerate(image_loader):
            if batch.shape[0] < training_args["batch_size"]:
                continue

            batch = batch.to(device)
            stylized_batch = transformer_net(batch)

            if use_temporal_loss:
                temp_loss = 0
                for k in range(batch.shape[0] - 1):
                    prev_frame = stylized_batch[k]
                    next_frame = stylized_batch[k + 1]
                    f_t, f_t1_w = utils.next_and_prev_warped(deep_flow, prev_frame, next_frame)
                    temp_loss += torch.nn.MSELoss()(f_t, f_t1_w)
                temp_loss *= training_args["temp_w"] * temp_loss
                temporal_losses.append(temp_loss)

            features_batch = loss_net(batch)
            features_stylized_batch = loss_net(stylized_batch)

            content_loss = training_args["content_w"] * torch.nn.MSELoss()(features_batch.relu_2_2, features_stylized_batch.relu_2_2)

            style_loss = 0.
            features_style_input = [utils.gram_matrix(x) for x in features_stylized_batch]
            for gram_target, gram_current in zip(features_style, features_style_input):
                style_loss += torch.nn.MSELoss()(gram_target, gram_current)
            style_loss = training_args["style_w"] * style_loss

            tv_loss = training_args["tv_w"] * utils.total_variation(stylized_batch)

            total_loss = 0#content_loss + style_loss + tv_loss
            if use_temporal_loss:
                total_loss += temp_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('Loss/content-loss', content_loss.item(), len(image_loader) * i + batch_id + 1)
            writer.add_scalar('Loss/style-loss', style_loss.item(), len(image_loader) * i + batch_id + 1)
            writer.add_scalar('Loss/temp-loss', temp_loss.item(), len(image_loader) * i + batch_id + 1)
            writer.add_scalar('Loss/tv-loss', tv_loss.item(), len(image_loader) * i + batch_id + 1)
            writer.add_scalar('Loss/Total-loss', total_loss.item(), len(image_loader) * i + batch_id + 1)

            total_losses.append(total_loss.item())
            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())
            tv_losses.append(tv_loss.item())
            temporal_losses.append(temp_loss.item())

            if batch_id % 20 == 0:
                print(f'Batch {batch_id}')
                print(f'Total loss: {total_loss} |||||| Content loss: {content_loss} |||||| Style loss: {style_loss} |||||| TV Loss: {tv_loss} |||||| Temporal loss: {temp_loss}')

    torch.save(transformer_net.state_dict(), os.path.join(model_path, "params.pth"))

    end = time.time()
    writer.close()
    utils.save_training_info(training_args=training_args, training_time=end-start, total_losses=total_losses,
                             content_losses=content_losses, style_losses=style_losses, tv_losses=tv_losses,
                             temp_losses=temporal_losses, model_path=model_path)

def train_video(video_dataset_dir: str):
    data_loader = utils.get_video_data_loader(video_dir=video_dataset_dir, transformer=None, batch_size=2)
    for i, batch in enumerate(data_loader):
        a = batch
