import os
import json
from typing import Tuple, List, Dict, Union
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from neural_nets import transform_net
import utils.VideoDataset as vdset
import cv2
from time import time_ns


def get_training_transformer(img_size: int) -> transforms.Compose:
    transform_steps = [
        transforms.Resize(size=img_size),
        transforms.CenterCrop(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_steps)


def get_data_loader(folder_path: str, transformer: transforms.Compose, batch_size: int) -> DataLoader:
    image_folder = datasets.ImageFolder(root=folder_path, transform=transformer)
    return DataLoader(dataset=image_folder, batch_size=batch_size)


def load_image(img_path: str, img_size: int = None) -> Image:
    img = Image.open(img_path)
    if img_size:
        img = img.resize((img_size, img_size), resample=Image.Resampling.LANCZOS)
    return img


def prepare_image(img_path: str, img_size: int = None) -> torch.Tensor:
    img = load_image(img_path=img_path, img_size=img_size)
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transformer(img)


def prepare_frame(frame: np.ndarray, device: torch.device) -> np.ndarray:
    # NST model je istreniran nad slikama koje su transformisane u tenzore sa vrednostima u
    # opsegu [0, 1], dok su ucitani frejmovi u opsegu [0, 255]. Takodje, Pillow biblioteka
    # koja je koriscena za rad sa slikama ucitanu sliku predstavlja kao C x H x W, dok
    # je frejm predstavljen numpy nizom u formi H x W x C (jer opencv tako funkcionise)

    transform_steps = transforms.Compose([
        transforms.Lambda(lambda x: x / 255),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame = torch.permute(torch.from_numpy(frame), (2, 0, 1))
    transformed_frame = transform_steps(frame)
    return transformed_frame.unsqueeze(0).to(device)


# def normalize_batch(batch):
#     # normalize using imagenet mean and std
#     mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
#     std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
#     return (batch - mean) / std


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def total_variation(img_batch: torch.Tensor) -> float:
    batch_size = img_batch.shape[0]
    return (torch.sum(torch.abs(img_batch[:, :, :, :-1] - img_batch[:, :, :, 1:])) +
            torch.sum(torch.abs(img_batch[:, :, :-1, :] - img_batch[:, :, 1:, :]))) / batch_size


def post_process(img: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    output_img = (img * std) + mean  # de-normalize
    output_img = (np.clip(output_img, 0., 1.) * 255).astype(np.uint8)
    return output_img.transpose(1, 2, 0)


def save_image(img: np.ndarray, path: str) -> None:
    img = Image.fromarray(img)
    img.save(path)


def plot_loss(losses: List[float], path: str, loss_type: str):
    graphs_path = os.path.join(path, "loss_graphs")
    os.makedirs(graphs_path, exist_ok=True)
    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.title(loss_type)
    plt.savefig(os.path.join(graphs_path, loss_type))


def save_training_info(training_args: Dict[str, Union[float, int]], training_time: float, total_losses: List[float],
                       content_losses: List[float], style_losses: List[float], tv_losses: List[float],
                       temp_losses: List[float], model_path: str):
    training_info = {
        **training_args,
        "training_time": training_time
    }
    with open(os.path.join(model_path, "training_info.json"), 'w') as f:
        json.dump(training_info, f)

    plot_loss(losses=total_losses, path=model_path, loss_type="Total Loss")
    plot_loss(losses=content_losses, path=model_path, loss_type="Content Loss")
    plot_loss(losses=style_losses, path=model_path, loss_type="Style Loss")
    plot_loss(losses=tv_losses, path=model_path, loss_type="Total variation Loss")
    plot_loss(losses=temp_losses, path=model_path, loss_type="Temporal Loss")


def mkdirs(dirs: Tuple[str, ...]) -> None:
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    transformer_net = transform_net.ImageTransformNet()
    state_dict = torch.load(model_path)
    transformer_net.load_state_dict(state_dict, strict=True)
    return transformer_net.to(device).eval()


# --- VIDEO RELATED ---

def get_video_data_loader(video_dir: str, transformer: transforms.Compose, batch_size: int) -> DataLoader:
    dataset = vdset.VideoDataset(video_dir=video_dir)
    return DataLoader(dataset=dataset, batch_size=batch_size)


def parse_and_save_video_frames(video_path: str, dst_path: str) -> None:
    frames = []

    v_rd = cv2.VideoCapture(video_path)
    ret, frame = v_rd.read()
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        ret, frame = v_rd.read()

    print(f'Successfully parsed {len(frames)} video frames.')
    print("Saving frames to dataset folder...")

    video_name = video_path.split(os.sep)[-1].split(".")[0]
    timestamp = time_ns()
    for i, frame in enumerate(frames):
        frame_img = Image.fromarray(frame)
        frame_img.save(os.path.join(dst_path, f'{timestamp + i}_{video_name}_frame_{i}.png'))


def prepare_video_dataset(videos_folder_path: str, dst_path: str) -> None:
    os.makedirs(dst_path, exist_ok=True)
    videos_list = os.listdir(videos_folder_path)
    for video in videos_list:
        if not os.path.isfile(os.path.join(videos_folder_path, video)):
            continue
        print(f'\n\nParsing video: {video}\n')
        parse_and_save_video_frames(video_path=os.path.join(videos_folder_path, video),
                                    dst_path=dst_path)
    print(f'Parsing successful.\nTotal number of frames parsed: {len(os.listdir(dst_path))}')


def next_and_prev_warped(deepflow, prev_frame, next_frame):
    prev_frame = cv2.cvtColor(prev_frame.cpu().detach().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    next_frame = cv2.cvtColor(next_frame.cpu().detach().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    opt_flow = deepflow.calc(prev_frame, next_frame, None)

    h, w = prev_frame.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    flow_x = opt_flow[..., 0]
    flow_y = opt_flow[..., 1]

    # Compute the new pixel coordinates by adding the flow to the original coordinates
    map_x = grid_x + flow_x
    map_y = grid_y + flow_y

    # Warp the previous frame to the new coordinates using cv2.remap
    warped_frame = cv2.remap(prev_frame, map_x.astype(np.float32), map_y.astype(np.float32),
                             interpolation=cv2.INTER_LINEAR)

    return torch.from_numpy(next_frame), torch.from_numpy(warped_frame)
