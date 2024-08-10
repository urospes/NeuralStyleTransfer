from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt

VGG19_Norm_Mean = [0.485, 0.456, 0.406]
VGG19_Norm_Std = [0.229, 0.224, 0.225]
DEFAULT_IMG_SIZE = 512

img_transformer = transforms.Compose([
    transforms.Resize(DEFAULT_IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=VGG19_Norm_Mean, std=VGG19_Norm_Std)
])

img_inverse_transformer = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=list(map(lambda std: 1. / std, VGG19_Norm_Std))),
    transforms.Normalize(mean=list(map(lambda mean: -mean, VGG19_Norm_Mean)), std=[1., 1., 1.]),
    transforms.ToPILImage()
])


def load_image_on_device(img_path, device):
    img = Image.open(img_path)
    img = img_transformer(img).unsqueeze(0)
    return img.to(device, torch.float)


def display_image(img_tensor, img_title):
    img = img_tensor.cpu().clone()
    img = img.squeeze(0)
    img = img_inverse_transformer(img)
    plt.imshow(img)
    plt.title(img_title)
    plt.show()
