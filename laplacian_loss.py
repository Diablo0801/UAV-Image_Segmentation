import torch
import torch.nn.functional as F

def gaussian_kernel(size=5, sigma=1.0, channels=1):
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size, size)
    return kernel.expand(channels, 1, size, size)

def laplacian_pyramid_loss(y_true, y_pred, levels=3, kernel_size=5, sigma=1.0):
    loss = 0.0
    channels = 1
    kernel = gaussian_kernel(kernel_size, sigma, channels).to(y_pred.device)
    
    for _ in range(levels):
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)

        y_true_blur = F.conv2d(y_true, kernel, padding='same', groups=channels)
        y_pred_blur = F.conv2d(y_pred, kernel, padding='same', groups=channels)

        y_true_lap = y_true - y_true_blur
        y_pred_lap = y_pred - y_pred_blur

        loss += F.l1_loss(y_true_lap, y_pred_lap)

        y_true = F.interpolate(y_true, scale_factor=0.5, mode='bilinear')
        y_pred = F.interpolate(y_pred, scale_factor=0.5, mode='bilinear')

    return loss