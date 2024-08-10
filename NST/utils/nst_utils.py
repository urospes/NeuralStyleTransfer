import torch


def gram_matrix(style_target):
    b_size, n_maps, w, h = style_target.size()
    style_target = style_target.view(b_size * n_maps, w * h)
    gram = torch.mm(style_target, style_target.t())
    return gram.div(b_size * n_maps * w * h)


def get_optimizer(input_img):
    optimizer = torch.optim.LBFGS([input_img])
    return optimizer


def get_losses(curr_content_repr, curr_style_repr, target_content_repr, target_style_repr, content_w, style_w):
    content_loss = torch.nn.MSELoss(reduction='mean')(curr_content_repr, target_content_repr)

    style_loss = torch.zeros(1, 1)
    for curr_style, t_style in zip(curr_style_repr, target_style_repr):
        style_loss += torch.nn.MSELoss(reduction='mean')(curr_style, t_style)

    total_loss = content_w * content_loss + style_w * style_loss
    return total_loss, content_loss, style_loss


def run_nst(cnn, opt_img, content_repr, style_repr, content_layer, style_layers, content_w=1, style_w=1, max_iter=200):
    opt_img.requires_grad_(True)

    optimizer = get_optimizer(opt_img)
    print("Optimizing...")
    i = 0
    while i <= max_iter:
        def closure():
            with torch.no_grad():
                opt_img.clamp_(0, 1)
            nonlocal i
            optimizer.zero_grad()
            opt_img_feat_maps = cnn(opt_img)
            opt_img_content_features = opt_img_feat_maps[content_layer]
            opt_img_style_features = [opt_img_feat_maps[style_layer] for style_layer in style_layers]

            opt_img_content_repr = opt_img_content_features.squeeze(0)
            opt_img_style_repr = [gram_matrix(style_f_map) for style_f_map in opt_img_style_features]

            total_loss, content_loss, style_loss = get_losses(opt_img_content_repr, opt_img_style_repr, content_repr,
                                                              style_repr, content_w, style_w)

            total_loss.backward()
            if i % 50 == 0:
                print(f'ITERATION {i}')
                print(f'Total Loss => {total_loss}')
                print(f'Content Loss => {content_loss}')
                print(f'Style Loss => {style_loss}')
            i += 1
            return total_loss

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        opt_img.clamp_(0, 1)
    return opt_img
