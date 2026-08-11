import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- STEP 1: PREP THE DATA ---
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_data_dir = "./data/CUB_200_2011/images" 

test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

# --- STEP 2: LOAD YOUR MODEL ---
import timm
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> Currently running on device: {device.type.upper()}")

model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in1k', pretrained=False, num_classes=200)
model.load_state_dict(torch.load("final_bird_weights.pth", map_location=device))
model.to(device)
model.eval()

# --- STEP 3: RUN THE TEST LOOP WITH PROGRESS TRACKER ---
all_preds = []
all_labels = []

print(f"Running model across {len(test_dataset)} test images ({len(test_loader)} total batches)...")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Prints live progress every batch
        print(f"Completed batch {batch_idx + 1} / {len(test_loader)}")

# --- STEP 4: PRINT THE SCORES ---
print("\n========== FINAL RESULTS ==========")
acc = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds))

# --- STEP 5: SHOW THE CONFUSION MATRIX ---
print("Generating Confusion Matrix window...")
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix")
plt.show()