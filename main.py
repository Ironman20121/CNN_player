import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tvt.train import train_model
from tvt.validate import validate_model
from tvt.evaluate import evaluate_model
from plots.plot_utils import plot_metrics, plot_confusion_matrix
from model.newcombo import SimpleCNN
from pretext_tasks.rotation_task import RotationDataset, train_rotation_task
from pretext_tasks.random_crop import AugmentedDataset, train_augmented_task
from torch.utils.data import DataLoader, random_split
def main():
    # Configuration
    data_dir = './data'
    batch_size = 128
    epochs = 2000
    learning_rate = 1e-3 
    # learning_rate-=1
    patience = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f"Configuration batch_size  : {batch_size} ,epochs : {epochs} ,learning_rate : {learning_rate} , patience : {patience} , device : {device}",sep="\n")
    # Configuration

    # Data Preprocessing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),      #he image by 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(dataset))  # 80
    val_size = len(dataset) - train_size   # 20


    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = datasets.CIFAR10(root=data_dir,train=False, download=True,transform =transform_test)


    # Model, Optimizer, and Loss Function no auto adjument lr fixed lr for now 
    model = SimpleCNN(num_classes=4).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,               # Initial learning rate
    momentum=0.9,          # Momentum to accelerate convergence
    weight_decay=5e-4      # Regularization to reduce overfitting
)
    criterion = nn.CrossEntropyLoss()
    
    
    pretext task 1
    rotation_dataset_train = RotationDataset(train_dataset)
    train_rotation_loader = DataLoader(rotation_dataset_train, batch_size=batch_size, shuffle=True)
    print("Training Rotation task")
    print(model)
    # train_rotation_task(model,train_rotation_loader,criterion,optimizer,device,epochs)
    history = train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_rotation_loader,
        val_loader=train_rotation_loader,
        epochs=200,
        patience=patience,
        device=device,
        model_name = "pretext"
    )


    print("Output layer is changing 4 to 10 ")
    ## changed layer for output
    model.fc = nn.Linear(256, 10)  # Adjust output layer for 10 classes
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,               # Initial learning rate
    momentum=0.9,          # Momentum to accelerate convergence
    weight_decay=5e-4      # Regularization to reduce overfitting
)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    
    print("Training")
    print(model)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Training and Validation
    history = train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        device=device,
        model_name="best_model",
        #resume="/home/kundan/Documents/project/best_model.pth"
    )

    print("Evaluation")
    # testing 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_accuracy, confusion_matrix = evaluate_model(model, test_loader, device)
    print(f"Test Acc :{test_accuracy}")
    # Plot Metrics
    print("Ploting")
    plot_metrics(history, save_path="./output/accuracy_loss.png")
    plot_confusion_matrix(confusion_matrix, classes=dataset.classes, save_path="./output/confusion_matrix.png")

if __name__ == "__main__":
    main()

