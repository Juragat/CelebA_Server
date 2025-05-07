import os
import io
import torch
import torch.nn as nn
import gdown
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from torchvision import transforms
from PIL import Image, ImageDraw
import uvicorn

# ---------- Config ----------
MODEL_PATH = "persist/best_model.pth"
GDRIVE_ID = "1rW6UfVvMkbAXOT9SNGLE6dFWVLLpbwfP"
os.makedirs("persist", exist_ok=True)

# ---------- Download model ----------
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Download complete.")
    else:
        print("Model already exists. Skipping download.")

# ---------- Model architecture ----------
class KeypointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 53 * 53, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.net(x)

# ---------- Init ----------
print("Setting device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Downloading model if needed...")
download_model()

print("Loading model...")
model = KeypointModel()
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()
print("Model loaded successfully.")

# ---------- Image transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- FastAPI ----------
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("Received file:", file.filename)
        image = Image.open(file.file).convert("RGB")
        original = image.resize((224, 224))
        tensor = transform(original).unsqueeze(0).to(device)
        print("Image transformed.")

        with torch.no_grad():
            print("Running model...")
            output = model(tensor).view(-1, 2).cpu().tolist()

        # Draw keypoints on the image
        draw = ImageDraw.Draw(original)
        for (x, y) in output:
            r = 3
            draw.ellipse((x - r, y - r, x + r, y + r), fill='red')

        print("Keypoints drawn. Returning image.")
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        print("Error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- Main ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
