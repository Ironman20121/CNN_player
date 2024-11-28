import torch
from tvt.validate import validate_model  # Ensure this import exists

def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, patience, device ,model_name,resume=None):
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']  + 1
    else :
        start_epoch = 1

    for epoch in range(start_epoch, epochs + 1):
        # Training Phase
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validation Phase
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)  # Validate here

        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, f'{model_name}.pth')

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Logging
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.4f}")

    return history

