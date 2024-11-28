import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from fastai.vision.all import *
import os 
import sys


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/newcombo.py')))

from model.newcombo import SimpleCNN
def find_optimal_lr():
    # Configuration
    data_dir = './data'
    batch_size = 64

    # Data Preprocessing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for fastai
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    dls = DataLoaders(train_loader, val_loader)

    # Model
    model = SimpleCNN(num_classes=10)  # Adjust for your model
    criterion = nn.CrossEntropyLoss()

    # Create a Learner
    learn = Learner(dls, model, loss_func=criterion, metrics=accuracy)

    # Find the optimal learning rate
    learn.lr_find()

    # Plot the results
    learn.recorder.plot_lr_find()

    # Save the plot
    plot_path = './output/lr_find_plot.png'  # Specify your desired path
    plt.savefig(plot_path)  # Save the current figure
    plt.close()  # Close the plot to free up memory

    # # Optionally, you can print the suggested learning rates
    # print(f"Suggested learning rate (min): {learn.recorder.lr_min}")
    # print(f"Suggested learning rate (steep): {learn.recorder.lr_steep}")
    # print(f"Learning rate plot saved to: {plot_path}")

if __name__ == "__main__":
    find_optimal_lr()
