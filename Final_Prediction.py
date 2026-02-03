# Prediction

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
sys.path.append("/content/HoVerTrans")
from hovertrans import create_model

# --------------------
# CONFIG
# --------------------
IMG_SIZE = 256
MODEL_PATH = "/content/FinalPaths(89.8).pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "benign",
    1: "malignant"
}

# --------------------
# IMAGE PREPROCESSING
# (same as test transform)
# --------------------
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# --------------------
# LOAD MODEL
# --------------------
def load_model():
    model = create_model(
        img_size=IMG_SIZE,
        num_classes=2,
        patch_size=[2, 2, 2, 2],
        dim=[4, 8, 16, 32],
        depth=[2, 4, 4, 2],
        num_heads=[2, 4, 8, 16],
        num_inner_head=[2, 4, 8, 16],
        drop_rate=0.0,
        attn_drop_rate=0.0
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --------------------
# PREDICT FUNCTION
# --------------------
def predict(image_path):
    model = load_model()

    img = Image.open(image_path)
    img = test_transform(img)

    # shape: [1, 1, H, W] → expand to 3 channels
    img = img.unsqueeze(0)
    img = img.expand(-1, 3, -1, -1)
    img = img.to(DEVICE)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    print(f"Prediction   : {CLASS_NAMES[pred_class]}")
    print(f"Confidence   : {confidence * 100:.2f}%")

    return CLASS_NAMES[pred_class], confidence


# --------------------
# RUN
# --------------------
if __name__ == "__main__":
    image_path = "/content/b1.png"  # change this
    predict(image_path)
