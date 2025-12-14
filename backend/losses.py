import torch
import torch.nn.functional as F

def sobel_loss(pred, target):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).to(pred.device)
    ky = kx.t()

    kx = kx.view(1,1,3,3)
    ky = ky.view(1,1,3,3)

    def edges(x):
        x = x.mean(1, keepdim=True)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    return F.l1_loss(edges(pred), edges(target))
