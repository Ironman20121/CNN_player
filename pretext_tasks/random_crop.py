import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transforms = ['rotation', 'flip', 'color_jitter', 'crop']
        self.angles = [0, 90, 180, 270]

    def __len__(self):
        return len(self.dataset) * len(self.transforms)

    def __getitem__(self, idx):
        original_idx = idx // len(self.transforms)
        transform_idx = idx % len(self.transforms)
        img, _ = self.dataset[original_idx]

        if self.transforms[transform_idx] == 'rotation':
            angle = random.choice(self.angles)
            transformed_img = TF.rotate(img, angle)
            label = angle // 90  # Label for rotation task

        elif self.transforms[transform_idx] == 'flip':
            if random.random() > 0.5:
                transformed_img = TF.hflip(img)
                label = 1  # Label for horizontal flip
            else:
                transformed_img = TF.vflip(img)
                label = 0  # Label for vertical flip

        elif self.transforms[transform_idx] == 'color_jitter':
            brightness = random.uniform(0.5, 1.5)
            contrast = random.uniform(0.5, 1.5)
            saturation = random.uniform(0.5, 1.5)
            hue = random.uniform(-0.1, 0.1)
            transformed_img = TF.adjust_brightness(img, brightness)
            transformed_img = TF.adjust_contrast(transformed_img, contrast)
            transformed_img = TF.adjust_saturation(transformed_img, saturation)
            transformed_img = TF.adjust_hue(transformed_img, hue)
            label = 2  # Label for color jittering

        elif self.transforms[transform_idx] == 'crop':
            i, j, h, w = TF.RandomCrop.get_params(img, output_size=(img.size[1] // 2, img.size[0] // 2))
            transformed_img = TF.crop(img, i, j, h, w)
            label = 3  # Label for cropping

        return transformed_img, label

def train_augmented_task(model, train_loader, criterion, optimizer, device, num_epochs):
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

# Example usage:
# dataset = YourImageDataset()  # Replace with your dataset
# augmented_dataset = AugmentedDataset(dataset)
# train_aug_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
# train_augmented_task(model, train_loader, criterion, optimizer, device='cuda', num_epochs=10)
