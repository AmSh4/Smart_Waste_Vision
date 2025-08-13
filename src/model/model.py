"""
model.py - simple helper to load the trained PyTorch model and run inference
"""
import torch, torch.nn.functional as F, os
from PIL import Image
from torchvision import transforms
import json

class SmallCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import torch.nn as nn
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*8*8,128)
        self.fc2 = nn.Linear(128,num_classes)
    def forward(self,x):
        import torch.nn.functional as F
        x = F.relu(self.conv1(x)); x=self.pool(x)
        x = F.relu(self.conv2(x)); x=self.pool(x)
        x = F.relu(self.conv3(x)); x=self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(path="src/model/waste_cnn.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    classes = checkpoint.get("classes", ["plastic","organic","metal"])
    model = SmallCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes, device

def predict_image(pil_image, model=None, classes=None, device=None):
    if model is None or classes is None or device is None:
        model, classes, device = load_model()
    transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    x = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = int(probs.argmax())
    return {"class": classes[top_idx], "confidence": float(probs[top_idx]), "all": dict(zip(classes, probs.tolist()))}
