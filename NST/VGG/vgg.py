import torch
from torchvision.models import vgg19, VGG19_Weights


# class ContentLoss(torch.nn.Module):
#     def __init__(self, content_target):
#         super().__init__()
#         self.content_target = content_target.detach()
#         self.loss = 0
#
#     def forward(self, input):
#         self.loss = torch.nn.functional.mse_loss(input, self.content_target)
#         return input
#
#
# class StyleLoss(torch.nn.Module):
#     def __init__(self, style_target):
#         super().__init__()
#         self.style_target = gram_matrix(style_target).detach()
#         self.loss = 0
#
#     def forward(self, input):
#         gram_input = gram_matrix(input)
#         self.loss = torch.nn.functional.mse_loss(gram_input, self.style_target)
#         return input


class Vgg19_NST(torch.nn.Module):
    def __init__(self, content_layer, style_layers):
        super().__init__()
        self.content_layer = content_layer
        self.style_layers = style_layers
        self.all_layers = style_layers + [content_layer]
        self.all_layers.sort()
        self.content_layer_index = self.all_layers.index(content_layer)

        vgg19_pretrained = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

        # delimo celu VGG cnn na manje podblokove, izlaz iz svakog podbloka su feature mape koje koristimo za racunanje content i style loss-a
        self.blocks = [torch.nn.Sequential() for _ in range(len(style_layers) + 2)]

        block_index = 0
        conv_index = [1, 1]
        for i, layer in enumerate(vgg19_pretrained.children()):
            self.blocks[block_index].add_module(str(i), layer)

            if isinstance(layer, torch.nn.Conv2d):
                layer_name = "conv_{}_{}".format(conv_index[0], conv_index[1])
                if layer_name in self.all_layers:
                    block_index += 1
                    conv_index[1] += 1
            elif isinstance(layer, torch.nn.MaxPool2d):
                conv_index[0] += 1
                conv_index[1] = 1

        # poslenji blok nam nije potreban, jer slojevi nakon poslednjeg cije su nam feature mape u interesu, nisu potrebni
        self.blocks = self.blocks[:-1]
        for param in self.parameters():
            param.requires_grad = False

        for i, blk in enumerate(self.blocks):
            print(f'sub-block {i}', blk)


    def forward(self, input):
        vgg_output = dict()
        for i, block in enumerate(self.blocks):
            input = block(input)
            if i == self.content_layer_index:
                vgg_output[self.content_layer] = input
            else:
                vgg_output[self.all_layers[i]] = input
        return vgg_output
