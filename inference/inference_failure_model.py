"""
3D Print Failure Detection - Inference Script
---------------------------------------------
This script loads a trained MobileNetV2 model and performs inference 
on a single image to classify it as 'Good Print' or 'Bad Print'.

Features:
- Loads YAML config for defaults (threshold, paths).
- Preprocesses input images with the same transforms as training.
- Loads model weights for evaluation.
- Runs prediction and outputs probabilities via CLI.

Author: Jason Menard Vasallo
Date: 2025-10-03
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os, argparse, yaml

# -----------------------------
# Load Configuration
# -----------------------------
# Loads training config YAML to retrieve decision threshold
# and ensure consistency between training and inference.
with open("training/config_training.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Image Transforms
# -----------------------------
# Same preprocessing as used during training/validation:
# - Resize to 224x224
# - Convert to tensor
# - Normalize with ImageNet mean/std
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Load Model Function
# -----------------------------
# Initializes MobileNetV2, replaces final classifier with binary head,
# loads trained weights, moves model to device, sets to eval mode.
def load_model(weights_path):
    print("[INFO] Loading MobileNetV2 for inference...")
    # Start with architecture only (no pretrained weights) because custom training weights will be loaded.
    model = models.mobilenet_v2(weights=None)  # no pretrain since we load weights
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# -----------------------------
# Inference Function
# -----------------------------
# Preprocesses input image, runs forward pass,
# computes softmax probabilities, and applies thresholding.
def predict(image_path, model, threshold=0.35):
    img = Image.open(image_path).convert("RGB")
    tensor = eval_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        prob_bad = probs[0,1].item()
        prob_good = probs[0,0].item()

    # Default: classify as Bad if probability exceeds threshold (default 0.35)
    pred_class = "Bad Print" if prob_bad > threshold else "Good Print"
    return pred_class, prob_good, prob_bad

# -----------------------------
# Command-Line Interface (CLI)
# -----------------------------
# Usage:
#   python inference.py --image path/to/img.jpg --weights outputs/best_model.pth
# Allows overriding threshold via argument.
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="3D Print Failure Detection Inference")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--weights", default="outputs/best_model.pth", help="Path to trained weights")
    parser.add_argument("--threshold", type=float, default=CONFIG["training"].get("decision_threshold", 0.35),
                        help="Decision threshold for 'bad' classification")
    args = parser.parse_args()

    # Load model
    model = load_model(args.weights)

    # Predict
    # Run prediction and display results
    pred_class, prob_good, prob_bad = predict(args.image, model, threshold=args.threshold)

    print(f"[RESULT] {pred_class}")
    print(f"         Good Probability: {prob_good:.4f}")
    print(f"         Bad Probability : {prob_bad:.4f}")
