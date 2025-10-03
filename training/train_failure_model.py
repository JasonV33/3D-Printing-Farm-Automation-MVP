"""
Training Script for Print Quality Classification
------------------------------------------------
This script trains and evaluates a binary classifier (good vs bad print) 
using MobileNetV2 on a COCO-style dataset of 3D print images.

Features:
- Config-driven training (YAML file).
- Data augmentation for robustness.
- Stratified train/val/test split.
- Class-weighted loss to handle imbalance.
- Early stopping based on validation loss.
- Logs training history and saves plots (loss/accuracy).
- Saves best model and evaluates on test set with metrics (Confusion Matrix, ROC, PR).

Author: Jason Menard Vasallo
Date: 2025-10-03
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import os, json, yaml
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             balanced_accuracy_score)

# LOAD CONFIG
with open("training/config_training.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

data_dir = CONFIG["dataset"]["images_dir"]
coco_json = CONFIG["dataset"]["coco_json"]
epochs = int(CONFIG["training"]["epochs"])
batch_size = int(CONFIG["training"]["batch_size"])
lr = float(CONFIG["training"]["learning_rate"])
threshold = float(CONFIG["training"].get("decision_threshold", 0.35))
patience = int(CONFIG["training"].get("patience", 5))
output_dir = CONFIG["project"]["output_dir"]
os.makedirs(output_dir, exist_ok=True)

# Use GPU if available, otherwise fallback to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# LOAD COCO ANNOTATIONS
with open(coco_json, "r") as f:
    coco = json.load(f)

labels_map = {img["file_name"]: ann["category_id"]
              for img, ann in zip(coco["images"], coco["annotations"])}

# -------------------------
# TRANSFORMS
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# DATASET CLASS
# -------------------------
class PrintDataset(torch.utils.data.Dataset):
    """
    Custom dataset for print classification.
    Args:
        fnames: list of image filenames
        labels: list of integer labels (0=good, 1=bad)
        root: base directory containing images
        transform: torchvision transforms to apply
    """

    def __init__(self, fnames, labels, root, transform=None):
        self.fnames, self.labels, self.root, self.transform = fnames, labels, root, transform
    def __len__(self): return len(self.fnames)
    def __getitem__(self, idx):
        # Load image and apply transform
        img = Image.open(os.path.join(self.root, self.fnames[idx])).convert("RGB")
        return self.transform(img), self.labels[idx]

# -------------------------
# STRATIFIED SPLIT
# -------------------------
# Stratified splitting ensures class distribution is preserved 
# across train, validation, and test sets.
fnames, labels = list(labels_map.keys()), [labels_map[f] for f in labels_map]

train_ratio, val_ratio, test_ratio = CONFIG["dataset"]["train_split"], CONFIG["dataset"]["val_split"], CONFIG["dataset"]["test_split"]

fn_tr, fn_temp, lb_tr, lb_temp = train_test_split(
    fnames, labels, test_size=(1-train_ratio), stratify=labels, random_state=CONFIG["training"]["seed"]
)
val_size_rel = val_ratio / (val_ratio + test_ratio)
fn_val, fn_te, lb_val, lb_te = train_test_split(
    fn_temp, lb_temp, test_size=(1-val_size_rel), stratify=lb_temp, random_state=CONFIG["training"]["seed"]
)

train_ds = PrintDataset(fn_tr, lb_tr, data_dir, train_transform)
val_ds   = PrintDataset(fn_val, lb_val, data_dir, eval_transform)
test_ds  = PrintDataset(fn_te, lb_te, data_dir, eval_transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size)
test_loader  = DataLoader(test_ds, batch_size=batch_size)

print(f"[INFO] Split: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

# -------------------------
# MODEL
# -------------------------
# Load MobileNetV2 pretrained on ImageNet
# Replace the classifier head with a new Linear layer for binary classification
print("[INFO] Using MobileNetV2 (fine-tuning all layers)")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)
model = model.to(device)

# -------------------------
# CLASS WEIGHTS FROM TRAIN SET
# -------------------------
# Compute class weights inversely proportional to class frequency
# Helps mitigate class imbalance by penalizing underrepresented class errors more.
counts = np.bincount(lb_tr)
weights = torch.tensor([len(lb_tr)/c if c > 0 else 0 for c in counts],
                       dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=lr)
print(f"[INFO] Class weights (train split): {weights.cpu().numpy()}")

# -------------------------
# LOGGING
# -------------------------
history = {"train_loss": [], "val_loss": [], "val_bal_acc": []}
log_path = os.path.join(output_dir, CONFIG["logging"]["log_file"])
plot_path = os.path.join(output_dir, CONFIG["logging"]["plot_file"])
best_model_path = os.path.join(output_dir,"best_model.pth")

best_val_loss, patience_counter = float("inf"), 0

# -------------------------
# TRAINING LOOP
# -------------------------
# For each epoch:
# 1. Train model on training split.
# 2. Validate on validation split.
# 3. Log loss and balanced accuracy.
# 4. Save plots of training history.
# 5. Apply early stopping if val loss does not improve.
for epoch in range(epochs):
    print(f"[INFO] Epoch {epoch+1}/{epochs}")

    # train
    model.train(); running_loss=0.0
    for imgs, lbls in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward(); optimizer.step()
        running_loss += loss.item()
    # Average training loss across all mini-batches
    train_loss = running_loss/len(train_loader)

    # val
    model.eval(); val_loss=0.0; preds=[]; gts=[]
    with torch.no_grad():
        # Validation loop (no gradient updates) to monitor generalization
        for imgs, lbls in tqdm(val_loader, desc=f"Val {epoch+1}", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs); loss = criterion(out, lbls)
            val_loss += loss.item()
            probs = torch.softmax(out,1)[:,1]
            pr = (probs > threshold).long()
            preds.extend(pr.cpu()); gts.extend(lbls.cpu())
    val_bal_acc = balanced_accuracy_score(gts, preds)

    # log
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss/len(val_loader))
    history["val_bal_acc"].append(val_bal_acc)

    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[INFO] Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val BalancedAcc={val_bal_acc:.4f}")

    # plot
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(history["val_bal_acc"], label="Val Balanced Acc")
    plt.xlabel("Epoch"); plt.ylabel("Balanced Accuracy"); plt.legend(); plt.title("Validation Metric")

    plt.tight_layout()
    plt.savefig(plot_path); plt.close()

    # early stopping
    # Save model if validation improves, else increment patience counter.
    # Stop training when patience is exceeded.
    if val_loss < best_val_loss:
        best_val_loss = val_loss; patience_counter=0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("[WARNING] Early stopping triggered"); break

# -------------------------
# TEST EVAL
# -------------------------
# Load best model (lowest validation loss).
# Evaluate on test split and generate:
# - Classification report (precision/recall/F1).
# - Confusion Matrix plot.
# - ROC curve + AUC.
# - Precision-Recall curve + Average Precision.
# Save summary metrics in JSON for downstream use.
print("[INFO] Evaluating best model on test set...")
model.load_state_dict(torch.load(best_model_path))
model.eval(); preds=[]; probs=[]; gts=[]
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model(imgs)
        ps = torch.softmax(out,1)[:,1]
        pr = (ps > threshold).long()
        preds.extend(pr.cpu()); probs.extend(ps.cpu()); gts.extend(lbls.cpu())

report = classification_report(gts, preds, target_names=["good","bad"], output_dict=True)
print("[INFO] Classification Report:"); print(json.dumps(report, indent=2))

# Confusion Matrix
# Confusion matrix shows counts of true vs predicted labels.
cm = confusion_matrix(gts, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["good","bad"])
disp.plot(cmap="Blues", values_format="d")
plt.savefig(os.path.join(output_dir,"confusion_matrix.png")); plt.close()

# ROC Curve
# ROC curve (TPR vs FPR) and AUC for binary classifier evaluation.
fpr, tpr, _ = roc_curve(gts, probs, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure(); plt.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],"--",color="gray"); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve"); plt.legend(); 
plt.savefig(os.path.join(output_dir,"roc_curve.png")); plt.close()

# Precision-Recall
# Precision-Recall curve useful for imbalanced datasets.
prec, rec, _ = precision_recall_curve(gts, probs, pos_label=1)
ap = average_precision_score(gts, probs)
plt.figure(); plt.plot(rec,prec,label=f"AP={ap:.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
plt.legend(); plt.savefig(os.path.join(output_dir,"precision_recall_curve.png")); plt.close()

# Summary
summary = {"accuracy": report["accuracy"],
           "balanced_accuracy": balanced_accuracy_score(gts,preds),
           "per_class": {"good": report["good"], "bad": report["bad"]},
           "roc_auc": roc_auc, "average_precision": ap}
with open(os.path.join(output_dir,"summary_metrics.json"),"w") as f:
    json.dump(summary,f,indent=2)
print("[SUCCESS] Evaluation complete, results saved")
