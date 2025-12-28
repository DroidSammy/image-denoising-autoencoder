from flask import Flask, request, jsonify
import cv2, numpy as np, base64, torch
from io import BytesIO
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from torchvision import transforms
import torch.nn as nn

app = Flask(__name__)

# ---------------------------------------------------
# UNet (Residual Noise Removal)
# ---------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, 2); self.c4 = DoubleConv(1024, 512)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2);  self.c3 = DoubleConv(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2);  self.c2 = DoubleConv(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2);   self.c1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, inp):
        d1 = self.d1(inp); d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2)); d4 = self.d4(self.pool(d3))
        b  = self.bottleneck(self.pool(d4))
        x  = self.c4(torch.cat([self.u4(b), d4], 1))
        x  = self.c3(torch.cat([self.u3(x), d3], 1))
        x  = self.c2(torch.cat([self.u2(x), d2], 1))
        x  = self.c1(torch.cat([self.u1(x), d1], 1))
        noise = self.out(x)
        return torch.clamp(inp - noise, 0.0, 1.0)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
MODEL_PATH = "models/denoise_unet.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNet().to(device)
unet.load_state_dict(torch.load(MODEL_PATH, map_location=device))
unet.eval()

transform = transforms.ToTensor()

# ---------------------------------------------------
def decode_image(b64):
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def encode_image(img):
    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')

def run_model(img, strength=1.0, sharpen=False):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad(): clean = unet(t).cpu().squeeze().numpy()
    clean = (np.transpose(clean, (1,2,0)) * 255).astype(np.uint8)
    clean = cv2.cvtColor(clean, cv2.COLOR_RGB2BGR)
    clean = cv2.addWeighted(img, 1-strength, clean, strength, 0)
    if sharpen:
        blur = cv2.GaussianBlur(clean,(0,0),3)
        clean = cv2.addWeighted(clean,1.2,blur,-0.2,0)
    return clean
# ---------------------------------------------------

@app.route("/denoise", methods=["POST"])
def denoise_api():
    data = request.json
    original = decode_image(data["image"])
    strength = float(data.get("strength", 1.0))
    sharpen  = bool(data.get("sharpen", False))
    output   = run_model(original, strength, sharpen)

    og = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    dn = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    return jsonify({
        "denoised_image": encode_image(output),
        "metrics": {
            "psnr": round(peak_signal_noise_ratio(og,dn), 3),
            "ssim": round(structural_similarity(og,dn), 3),
            "mse":  round(mean_squared_error(og,dn), 3),
        }
    })

if __name__ == "__main__":
    print("\n🌍 Running on http://127.0.0.1:5000/denoise\n")
    app.run(host="0.0.0.0", port=5000)
