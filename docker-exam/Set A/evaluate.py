import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import random
from sklearn.metrics import accuracy_score, f1_score, classification_report
from PIL import Image

DATA_DIR = "data/test/"
MODEL_PATH = "setA.pth"
BATCH_SIZE = 32
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
class_names = dataset.classes
print("Classes:", class_names)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded from", MODEL_PATH)

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

overall_acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")
print(f"F1 Score (macro): {macro_f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

random_image_path, _ = random.choice(dataset.samples)
image = Image.open(random_image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    output = model(image)
    probs = torch.softmax(output, dim=1)
    confidence, pred = torch.max(probs, 1)
print(f"\nRandom Image: {random_image_path}")
print(f"Predicted: {class_names[pred.item()]} ({confidence.item()*100:.2f}% confidence)")
