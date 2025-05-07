import os
import torch
import torch.nn as nn
import gdown
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import uvicorn

# ---------- Config ----------
MODEL_PATH = "persist/best_model.pth"
GDRIVE_ID = "1rW6UfVvMkbAXOT9SNGLE6dFWVLLpbwfP"
os.makedirs("persist", exist_ok=True)

# ---------- Download model from Google Drive if not exists ----------
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ---------- Define the model ----------
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
download_model()
model = KeypointModel()
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

# ---------- Image transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- API ----------
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor).view(-1, 2).cpu().tolist()
        return {"keypoints": output}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- Main ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
