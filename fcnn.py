import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyper parameters
# ==========================================
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
# ==========================================
def get_dataloaders():
    # Pre-process with data augmentation(RandomCrop + RandomHorizontalFlip + RandomRotation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader, test_dataset.classes

# Define FCNN
# ==========================================
class SimpleFCNN(nn.Module):
    def __init__(self):
        super(SimpleFCNN, self).__init__()
        # Flatten image (CIFAR's default image size is 3x32x32)
        self.flatten = nn.Flatten()
        
        self.network = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits

# Training and validation loop
# ==========================================
def train_and_evaluate(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss() # Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
        epoch_test_loss = running_test_loss / total_test
        epoch_test_acc = 100 * correct_test / total_test
        test_losses.append(epoch_test_loss)
        test_accs.append(epoch_test_acc)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - LR: {current_lr:.6f} |"
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")

    return train_losses, test_losses, train_accs, test_accs

# Visualization
# ==========================================
def plot_curves(train_losses, test_losses, train_accs, test_accs, model_type):
    if model_type == 'FCNN':
        filename1 = 'fcnn_loss_curve.png'
        filename2 = 'fcnn_acc_curve.png'

    elif model_type == 'CNN':
        filename1 = 'cnn_loss_curve.png'
        filename2 = 'cnn_acc_curve.png'

    # Plot and save Loss and Acc curves
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', marker='s', color='orange')
    plt.title(str(model_type + 'Train and Test Losses'))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='o', color='green')
    plt.plot(epochs, test_accs, label='Test Accuracy', marker='s', color='red')
    plt.title(str(model_type + 'Train and Test Accuracies'))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    plt.close()

    print('Succesfully plot and save curves')

  

def visualize_predictions(model, test_loader, classes, model_type):
    if model_type == 'FCNN':
        filename = 'fcnn_predictions.png'

    elif model_type == 'CNN':
        filename = 'cnn_predictions.png'

    # Visualization of 5 random images with their ground truth and predicted labels in captions
    model.eval()
    
    # get first batch fromn test loader
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    
    images = images[:5]
    labels = labels[:5]
    
    
    device = next(model.parameters()).device
    images_dev = images.to(device)
    
    with torch.no_grad():
        outputs = model(images_dev)
        _, predicted = torch.max(outputs, 1)
        
  
    fig = plt.figure(figsize=(15, 4))
    
    # CIFAR-100 mean and standard deviation
    means = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    stds = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
    
    for idx in range(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        
        img = images[idx].cpu().clone()
        
        # Un-normalize: img = img * std + mean
        img = img * stds + means
        
        # Transpose iamge from (C, H, W) to (H, W, C) for matplotlib
        npimg = img.numpy()
        npimg = np.clip(npimg, 0, 1)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
    
        gt_label = classes[labels[idx]]
        pred_label = classes[predicted[idx]]
        
        # True in green, false in red
        color = "green" if gt_label == pred_label else "red"
        ax.set_title(f"GT: {gt_label}\nPred: {pred_label}", color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(" Predictions image saved")

# Main
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train_loader, test_loader, classes = get_dataloaders()
    
    model = SimpleFCNN().to(DEVICE)
    train_losses, test_losses, train_accs, test_accs = train_and_evaluate(model, train_loader, test_loader)

    model_type = 'FCNN'
    plot_curves(train_losses, test_losses, train_accs, test_accs, model_type)
    visualize_predictions(model, test_loader, classes, model_type)
    
    # Save model
    torch.save(model.state_dict(), "fcnn_model.pth")
    print("Training complete and model saved.")