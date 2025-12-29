from flask import Flask, request, jsonify
import torch, cv2, base64, numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import torch.nn as nn

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
#        MODEL
# ============================
class ConvBlock(nn.Module):
    def __init__(self, inc, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc, out, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out, out, 3, padding=1), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class DenoiseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.ModuleList([
            ConvBlock(3,64),ConvBlock(64,128),
            ConvBlock(128,256),ConvBlock(256,512)
        ])
        self.bottle = ConvBlock(512,512)
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(512,256,2,2), ConvBlock(256,256),
            nn.ConvTranspose2d(256,128,2,2), ConvBlock(128,128),
            nn.ConvTranspose2d(128,64,2,2),  ConvBlock(64,64),
            nn.ConvTranspose2d(64,64,2,2)
        ])
        self.final = nn.Conv2d(64,3,1)

    def forward(self, x):
        for l in self.down:
            x = l(x)
            x = nn.MaxPool2d(2)(x)
        x = self.bottle(x)
        for i in range(0,len(self.up)-1,2):
            x = self.up[i](x)
            x = self.up[i+1](x)
        return torch.sigmoid(self.final(x))

model = DenoiseAutoencoder().to(device)
model.eval()


# ============================
#  UTILITIES
# ============================
def decode(b64):
    data = base64.b64decode(b64)
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def encode(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


# ============================
#  MAIN LOGIC (Best Version)
# ============================
def infer(img, strength):
    h, w = img.shape[:2]

    resized = cv2.resize(img, (256,256))
    t = torch.from_numpy(resized/255.).float().permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        base = model(t).cpu().squeeze().permute(1,2,0).numpy()

    base = (base*255).clip(0,255).astype(np.uint8)
    base = cv2.resize(base, (w,h))

    # Non-linear curve (bigger impact at high strength)
    p = strength ** 2.3
    clean = cv2.addWeighted(img, 1-p, base, p, 0)

    # Second-stage denoise when strength is high
    if strength > 0.4:
        level = 8 + int(strength*15)
        clean = cv2.fastNlMeansDenoisingColored(clean, None, level, level, 7, 21)

    # Micro sharpen to restore clarity
    sharp = cv2.GaussianBlur(clean, (0,0), 1.3 + strength)
    clean = cv2.addWeighted(clean, 1.12, sharp, -0.12, 0)

    # Difference heatmap
    diff = cv2.absdiff(img, clean)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(diff_gray, cv2.COLORMAP_INFERNO)

    return clean, heat


# ============================
#  API
# ============================
@app.route("/denoise", methods=["POST"])
def api():
    data = request.json
    img = decode(data["image"])
    strength = float(data.get("strength", 0.1))

    enhanced, heatmap = infer(img, strength)

    og = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dn = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    return jsonify({
        "clean": encode(enhanced),
        "difference": encode(heatmap),
        "psnr": float(peak_signal_noise_ratio(og, dn)),
        "ssim": float(structural_similarity(og, dn)),
        "mse": float(mean_squared_error(og, dn))
    })


if __name__ == "__main__":
    print("🚀 Running at http://127.0.0.1:5000/denoise")
    app.run(host="0.0.0.0", port=5000)
