import numpy as np
import torch
from utils import utils


def stylize_image(content_image_path: str, model_path: str, output_image_path: str, device: torch.device, save_image: bool = True):

    content_image = utils.prepare_image(content_image_path)
    content_image = content_image.unsqueeze(0).to(device)

    transformer_net = utils.load_model(model_path=model_path, device=device)

    with torch.no_grad():
        stylized_image = transformer_net(content_image)
        stylized_image = utils.post_process(stylized_image.cpu().numpy()[0])
        if save_image:
            utils.save_image(stylized_image, output_image_path)

    return stylized_image


def stylize_frame(frame: np.ndarray, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    frame = utils.prepare_frame(frame, device)
    with torch.no_grad():
        stylized_frame = model(frame)
        stylized_frame = utils.post_process(stylized_frame.cpu().numpy()[0])

    return stylized_frame
