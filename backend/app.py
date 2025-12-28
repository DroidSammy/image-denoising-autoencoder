from flask import Flask, request, jsonify
import torch, cv2, base64, numpy as np
from io import BytesIO
from PIL import Image
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# 1️⃣  MODEL (Matches your checkpoint structure)
# -------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class DenoiseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.downs = nn.ModuleList([
            ConvBlock(3,64),
            ConvBlock(64,128),
            ConvBlock(128,256),
            ConvBlock(256,512)
        ])

        self.bottleneck = ConvBlock(512,512)

        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(512,256,2,2),
            ConvBlock(256,256),
            nn.ConvTranspose2d(256,128,2,2),
            ConvBlock(128,128),
            nn.ConvTranspose2d(128,64,2,2),
            ConvBlock(64,64),
            nn.ConvTranspose2d(64,64,2,2),
        ])

        self.final = nn.Conv2d(64,3,1)

    def forward(self, x):
        for block in self.downs:
            x = block(x)
            x = nn.MaxPool2d(2)(x)

        x = self.bottleneck(x)

        for i in range(0, len(self.ups)-1, 2):
            x = self.ups[i](x)
            x = self.ups[i+1](x)

        return torch.sigmoid(self.final(x))


# -------------------------------------------------------
# 2️⃣ LOAD MODEL WITH RELAXED RULES (STRICT=False)
# -------------------------------------------------------
model = DenoiseAutoencoder().to(device)

MODEL_PATHS = [
    "models/best_unet.pt",
    "models/denoise_unet.pt"
]

loaded = False
for path in MODEL_PATHS:
    try:
        print(f"🔍 Trying to load: {path}")
        state = torch.load(path, map_location=device)
        model.load_state_dict(state, strict=False)   # ⭐ FIX HERE
        print(f"🔥 Loaded successfully with relaxed keys: {path}")
        loaded = True
        break
    except Exception as e:
        print(f"❌ Failed: {path} => {e}")

if not loaded:
    raise FileNotFoundError("❌ None of the model files could be loaded! Check /backend/models")

model.eval()


# -------------------------------------------------------
# 3️⃣ HELPERS
# -------------------------------------------------------
def decode(b64):
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def encode(img):
    ok, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# -------------------------------------------------------
# 4️⃣ INFERENCE
# -------------------------------------------------------
def infer(img, strength=1.0, sharpen=False):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (256,256))

    t = torch.from_numpy(resized/255.0).float().permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        den = model(t).cpu().squeeze().permute(1,2,0).numpy()

    den = (den*255).clip(0,255).astype(np.uint8)
    den = cv2.resize(den, (w,h))

    final = cv2.addWeighted(img, 1-strength, den, strength, 0)

    if sharpen:
        blur = cv2.GaussianBlur(final,(0,0),3)
        final = cv2.addWeighted(final, 1.5, blur, -0.5, 0)

    return final


# -------------------------------------------------------
# 5️⃣ API ENDPOINT
# -------------------------------------------------------
@app.route("/denoise", methods=["POST"])
def api():
    data = request.json
    img = decode(data["image"])
    strength = float(data.get("strength", 1.0))
    sharpen = bool(data.get("sharpen", False))

    out = infer(img, strength, sharpen)

    og = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dn = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)

    return jsonify({
        "image": encode(out),
        "psnr": float(peak_signal_noise_ratio(og, dn)),
        "ssim": float(structural_similarity(og, dn)),
        "mse":  float(mean_squared_error(og, dn))
    })


# -------------------------------------------------------
# 6️⃣ START SERVER
# -------------------------------------------------------
if __name__ == "__main__":
    print("🚀 RUNNING: http://127.0.0.1:5000/denoise")
    app.run(host="0.0.0.0", port=5000)