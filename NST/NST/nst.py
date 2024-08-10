import torch
import utils.image_utils as img_utils
import utils.nst_utils as nst_utils
import VGG.vgg as vgg
import numpy as np

content_image_path = "../images/dancing.jpg"
style_image_path = "../images/picasso.jpg"

content_layer = "conv_4_2"
style_layers = ["conv_1_1", "conv_2_1", "conv_3_1", "conv_4_1", "conv_5_1"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Pytorch is running on {device} device')
torch.set_default_device(device)

content_image = img_utils.load_image_on_device(content_image_path, device)
#img_utils.display_image(content_image, "Content Image")
style_image = img_utils.load_image_on_device(style_image_path, device)
#img_utils.display_image(style_image, "Style Image")

cnn = vgg.Vgg19_NST(content_layer, style_layers).to(device)

content_img_feat_maps = cnn(content_image)
style_img_feat_maps = cnn(style_image)

content_img_content_features = content_img_feat_maps[content_layer]
style_img_style_features = [style_img_feat_maps[style_layer] for style_layer in style_layers]

content_repr = content_img_content_features.squeeze(0).detach()
style_repr = [nst_utils.gram_matrix(style_f_map).detach() for style_f_map in style_img_style_features]

#opt_img = img_utils.load_image_on_device(content_image_path, device)
gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_image.shape).astype(np.float32)
opt_img = torch.from_numpy(gaussian_noise_img).float().to(device)
img_utils.display_image(opt_img, "INIT")
output = nst_utils.run_nst(cnn, opt_img, content_repr, style_repr, content_layer, style_layers)
img_utils.display_image(output, "Final")
