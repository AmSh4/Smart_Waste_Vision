"""
train.py - train a tiny CNN on the synthetic dataset
Run: python src/model/train.py
Produces model saved as src/model/waste_cnn.pth
"""
import os, torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

DATA_DIR = "data/sample_images"
MODEL_OUT = "src/model/waste_cnn.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*8*8,128)
        self.fc2 = nn.Linear(128,num_classes)
    def forward(self,x):
        x = F.relu(self.conv1(x)); x=self.pool(x)
        x = F.relu(self.conv2(x)); x=self.pool(x)
        x = F.relu(self.conv3(x)); x=self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

num_classes = len(train_ds.classes)
model = SmallCNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 8
for epoch in range(epochs):
    model.train()
    running=0.0; total=0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        running += loss.item()*images.size(0); total += images.size(0)
    print(f"Epoch {epoch+1}/{epochs} - loss: {running/total:.4f}")
torch.save({"model_state":model.state_dict(),"classes":train_ds.classes}, MODEL_OUT)
print("Model saved to", MODEL_OUT)
