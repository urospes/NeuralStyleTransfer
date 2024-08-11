import os
import numpy as np
import torch
import torchvision.utils
from PIL import Image
from neural_nets.transform_net import ImageTransformNet
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_training_transformer(img_size: int) -> transforms.Compose:
    transform_steps = [
        transforms.Resize(size=img_size),
        transforms.CenterCrop(size=img_size),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.mul(255))
    ]
    return transforms.Compose(transform_steps)


def get_data_loader(folder_path: str, transformer: transforms.Compose, batch_size: int) -> DataLoader:
    image_folder = datasets.ImageFolder(root=folder_path, transform=transformer)
    return DataLoader(dataset=image_folder, batch_size=batch_size)


def load_image(img_path: str, img_size: int = None):
    img = Image.open(img_path)
    if img_size:
        img = img.resize((img_size, img_size), resample=Image.Resampling.LANCZOS)
    return img


def prepare_image(img_path: str, img_size: int = None):
    img = load_image(img_path=img_path, img_size=img_size)
    transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.mul(255))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transformer(img)


# def normalize_batch(batch):
#     # normalize using imagenet mean and std
#     mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
#     std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
#     return (batch - mean) / std


def gram_matrix(x: torch.Tensor):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def save_model(transformer_net: ImageTransformNet, file_name: str):
    os.makedirs("models/", exist_ok=True)
    torch.save(transformer_net.state_dict(), f'models/{file_name}.model')


def post_process_image(img: np.ndarray):
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    dump_img = (img * std) + mean  # de-normalize
    dump_img = (np.clip(dump_img, 0., 1.) * 255).astype(np.uint8)
    dump_img = dump_img.transpose(1, 2, 0)
    return dump_img


def save_image(img: np.ndarray, path: str):
    img = Image.fromarray(img)
    img.save(f'./images/output/{path}')
