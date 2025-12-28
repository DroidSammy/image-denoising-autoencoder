from flask import Flask, request, jsonify
import cv2, numpy as np, base64, torch
from io import BytesIO
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from torchvision import transforms
import torch.nn as nn

app = Flask(__name__)

# ============================================================
# 1️⃣ EXACT UNET YOU TRAINED (MATCHES YOUR .pt WEIGHTS)
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- ENCODER BLOCKS (d1,d2,d3,d4) ----
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # ---- BOTTLENECK ----
        self.bottleneck = DoubleConv(512, 1024)

        # ---- DECODER BLOCKS (u4,c4 -> u1,c1) ----
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.c4 = DoubleConv(1024, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c3 = DoubleConv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c2 = DoubleConv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c1 = DoubleConv(128, 64)

        # ---- FINAL CONV LAYER (out.weight) ----
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, inp):
        d1 = self.d1(inp)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        b  = self.bottleneck(self.pool(d4))

        x = self.c4(torch.cat([self.u4(b), d4], 1))
        x = self.c3(torch.cat([self.u3(x), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))

        noise = self.out(x)
        return torch.clamp(inp - noise, 0.0, 1.0)  # ⭐ RESIDUAL CLEANING


# ============================================================
# 2️⃣ LOAD MODEL
# ============================================================

MODEL_PATH = "models/denoise_unet.pt"  # your file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Loading Residual UNet model...")

unet = UNet().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)

# strict=False allows BN + unexpected keys without crashing
unet.load_state_dict(state_dict, strict=False)

unet.eval()
print("✅ Model Loaded Successfully on:", device)


# ============================================================
# 3️⃣ IMAGE UTILS
# ============================================================

transform = transforms.ToTensor()

def decode_image(b64):
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def encode_image(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


# ============================================================
# 4️⃣ INFERENCE
# ============================================================

def run_model(img, strength=1.0, sharpen=False):
    inp = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = unet(inp).cpu().squeeze(0).numpy()

    clean = (np.transpose(pred, (1,2,0)) * 255).astype(np.uint8)

    clean = cv2.addWeighted(img, 1-strength, clean, strength, 0)

    if sharpen:
        blur = cv2.GaussianBlur(clean, (0,0), 3)
        clean = cv2.addWeighted(clean, 1.5, blur, -0.5, 0)

    return clean


# ============================================================
# 5️⃣ FLUTTER ENDPOINT
# ============================================================

@app.route('/denoise', methods=['POST'])
def denoise_api():
    data = request.json
    strength = float(data.get("strength", 1.0))
    sharpen = bool(data.get("sharpen", False))

    original = decode_image(data["image"])
    output   = run_model(original, strength, sharpen)

    og = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    dn = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    psnr = peak_signal_noise_ratio(og, dn)
    ssim = structural_similarity(og, dn)
    mse  = mean_squared_error(og, dn)

    return jsonify({
        "denoised_image": encode_image(output),
        "metrics": {"psnr": psnr, "ssim": ssim, "mse": mse}
    })


# ============================================================
# 6️⃣ RUN SERVER
# ============================================================

if __name__ == "__main__":
    print("🌍 Server Active → http://127.0.0.1:5000/denoise")
    app.run(host="0.0.0.0", port=5000)
