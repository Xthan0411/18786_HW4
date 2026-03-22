import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from fcnn import get_dataloaders, train_and_evaluate

# Residual block
class BasicBlock(nn.Module):
    expansion = 1 # outchannel/inchannel

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Residual
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion * planes:
            # use a 1x1 kernel to modify dimension and size of x
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Conv -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Conv -> BN
        out = self.bn2(self.conv2(out))
        
        # add residual
        out += self.shortcut(x)
        
        # ReLU
        out = F.relu(out)
        return out
    

# ResNet backbone
class CIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(CIFAR_ResNet, self).__init__()
        self.in_planes = 64

        # 3x3 conv, stride 1, no MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual backbone, layer by layer downsampling and double channel
        # 32x32, 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # 32x32 -> 16x16，128
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # 16x16 -> 8x8，256
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # 8x8 -> 4x4，512
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Classifier
        # 512x4x4 -> 512x1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # planes = expected out_channel within the layer
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            # downsampling only for first block of layer
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 32x32
        out = F.relu(self.bn1(self.conv1(x)))
        # 32 -> 16 -> 8 -> 4
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Initialize model
def ResNet18_CIFAR100():
    
    return CIFAR_ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)

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
# ==============================================================================
def plot_curves(train_losses, test_losses, train_accs, test_accs):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('ResNet18 - Loss Curve')
    plt.legend()
    plt.savefig('resnet_loss_curve.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('ResNet18 - Accuracy Curve')
    plt.legend()
    plt.savefig('resnet_acc_curve.png')
    plt.close()

def visualize_predictions(model, test_loader, classes):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    images = images[:10].cpu()
    labels = labels[:10].cpu()
    predicted = predicted[:10].cpu()

    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    axes = axes.flatten() 
    
    for i in range(10):
        img = images[i] * std + mean
        img = np.clip(img.numpy().transpose((1, 2, 0)), 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
        
        gt_class = classes[labels[i]]
        pred_class = classes[predicted[i]]
        color = 'green' if gt_class == pred_class else 'red'
        axes[i].set_title(f'GT: {gt_class}\nPred: {pred_class}', color=color)
        
    plt.tight_layout()
    plt.savefig('resnet_predictions.png')
    plt.close()

# Hyper-param
# ==============================================================================
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 60  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main
# ==============================================================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    

    train_loader, test_loader, classes = get_dataloaders()
    
    model = ResNet18_CIFAR100().to(DEVICE)
    
    train_losses, test_losses, train_accs, test_accs = train_and_evaluate(model, train_loader, test_loader)
    
    plot_curves(train_losses, test_losses, train_accs, test_accs)
    visualize_predictions(model, test_loader, classes)
    
    # Save model
    torch.save(model.state_dict(), "resnet18_model.pth")
    print("Training complete and model saved.")