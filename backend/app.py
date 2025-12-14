from flask import Flask, request
import torch
import cv2
import numpy as np
from model import UNet

app = Flask(__name__)

# -----------------------------
# Device & Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

model = UNet().to(device)
model.load_state_dict(torch.load("models/denoise_unet.pt", map_location=device))
model.eval()


def mild_sharpen(img):
    """
    Very mild unsharp masking
    Safe for denoised images
    """
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(img, 1.15, blur, -0.15, 0)
    return sharpened


@app.route("/denoise", methods=["POST"])
def denoise():
    file = request.files["image"]
    sharpen_flag = request.args.get("sharpen", "0") == "1"

    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    # Resize only for model
    img_small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # Normalize
    x = torch.from_numpy(img_small).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = x.to(device)

    # Inference
    with torch.no_grad():
        y_small = model(x)[0]

    # Back to image
    y_small = y_small.permute(1, 2, 0).cpu().numpy()
    y_small = np.clip(y_small, 0.0, 1.0)
    y_small = (y_small * 255).astype(np.uint8)

    # Resize back to original
    y = cv2.resize(y_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Optional sharpening
    if sharpen_flag:
        y = mild_sharpen(y)

    # Encode output
    _, buffer = cv2.imencode(".png", cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
    return buffer.tobytes(), 200, {"Content-Type": "image/png"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
