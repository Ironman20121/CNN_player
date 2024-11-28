import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class RotationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.angles = [0, 90, 180, 270]

    def __len__(self):
        return len(self.dataset) * len(self.angles)

    def __getitem__(self, idx):
        original_idx = idx // len(self.angles)
        angle_idx = idx % len(self.angles)
        img, _ = self.dataset[original_idx]
        rotated_img = TF.rotate(img, self.angles[angle_idx])
        return rotated_img, angle_idx

def train_rotation_task(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

