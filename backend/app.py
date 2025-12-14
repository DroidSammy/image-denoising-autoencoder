from flask import Flask, request
import torch
import cv2
import numpy as np
from model import UNet

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

model = UNet().to(device)
model.load_state_dict(torch.load("models/denoise_unet.pt", map_location=device))
model.eval()


def mild_sharpen(img):
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.15, blur, -0.15, 0)


def add_watermark(img, text="Denoised by UNet"):
    overlay = img.copy()
    h, w, _ = img.shape

    cv2.putText(
        overlay,
        text,
        (w - 320, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # transparency
    return cv2.addWeighted(overlay, 0.6, img, 0.4, 0)


@app.route("/denoise", methods=["POST"])
def denoise():
    file = request.files["image"]

    sharpen = request.args.get("sharpen", "0") == "1"
    imprint = request.args.get("imprint", "1") == "1"

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    img_small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(img_small).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = x.to(device)

    with torch.no_grad():
        y_small = model(x)[0]

    y_small = y_small.permute(1, 2, 0).cpu().numpy()
    y_small = np.clip(y_small, 0, 1)
    y_small = (y_small * 255).astype(np.uint8)

    y = cv2.resize(y_small, (w, h), interpolation=cv2.INTER_LINEAR)

    if sharpen:
        y = mild_sharpen(y)

    if imprint:
        y = add_watermark(y)

    _, buffer = cv2.imencode(".png", cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
    return buffer.tobytes(), 200, {"Content-Type": "image/png"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
