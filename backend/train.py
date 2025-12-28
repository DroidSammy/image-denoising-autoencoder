import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from model import UNet
from dataset import SIDD_Dataset
from losses import sobel_loss
from metrics import psnr_metric, ssim_metric
import os

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Device:", device)

dataset = SIDD_Dataset("../data/SIDD/SIDD_medium/paired_sidd")

loader = DataLoader(
    dataset, batch_size=8, shuffle=True,
    num_workers=0, pin_memory=True
)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler("cuda")
l1 = nn.L1Loss()

os.makedirs("models", exist_ok=True)

for epoch in range(30):
    model.train()
    total_loss = 0
    total_psnr, total_ssim = 0, 0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        with autocast("cuda"):
            out = model(noisy)
            loss = l1(out, clean) + 0.1 * sobel_loss(out, clean)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # 🔥 Evaluation metrics per batch
        total_psnr += psnr_metric(out, clean)
        total_ssim += ssim_metric(out, clean)

    print(f"📌 Epoch {epoch+1}/30 | Loss: {total_loss/len(loader):.4f} | "
          f"PSNR: {total_psnr/len(loader):.2f} | SSIM: {total_ssim/len(loader):.3f}")

    torch.save(model.state_dict(), f"models/denoise_unet.pt")
