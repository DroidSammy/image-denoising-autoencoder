import torch
import torch.nn.functional as F
from math import log10
from skimage.metrics import structural_similarity as ssim

def psnr_metric(output, target):
    mse = F.mse_loss(output, target).item()
    if mse == 0:
        return 100
    return 20 * log10(1.0 / mse)

def ssim_metric(output, target):
    o = output.detach().cpu().numpy()[0].transpose(1,2,0)
    t = target.detach().cpu().numpy()[0].transpose(1,2,0)
    return ssim(o, t, channel_axis=2, data_range=1.0)
