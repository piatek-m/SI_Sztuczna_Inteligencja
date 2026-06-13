import time
import torch
import cv2
from torchvision import transforms
from PIL import Image

from src.models.model import build_model
from src.data.transforms import test_transform
from src.data.dataset import get_classes
from src.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE,
    CROP,
    PROJECT_DIR,
    ASSETS_DIR,
    MODELS_DIR,
    DEVICE,
    INFERENCE_INTERVAL,
)

classes = get_classes()
num_classes = len(classes)

model = build_model(num_classes)
model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth", map_location=DEVICE))
model.eval()

inference_transform = test_transform


def predict(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    tensor = inference_transform(frame).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max()
    return classes[predicted.item()], confidence.item()


cap = cv2.VideoCapture(0)

label_text = "Waiting..."
last_inference = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if now - last_inference >= INFERENCE_INTERVAL:
        label, confidence = predict(frame)
        label_text = f"{label} ({confidence*100:.1f}%)"
        last_inference = now

    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
