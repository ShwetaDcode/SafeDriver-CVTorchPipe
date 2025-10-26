import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Dataset and transforms
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
dataset = datasets.ImageFolder('data', transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# Model
model = mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# Quick training loop (synthetic demo)
for epoch in range(2):
    for imgs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# Save model
torch.save(model.state_dict(), 'drowsiness_quantized.pth')
print("Quantized model saved!")