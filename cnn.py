import torch
import torch.nn as nn
import torch.optim as optim


from mytorch import MyConv2D, MyMaxPool2D
from fcnn import get_dataloaders, train_and_evaluate, plot_curves, visualize_predictions

# Define CNN


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Feature Extractor
        self.features = nn.Sequential(
            # Input size: (Batch, 3, 32, 32)
            # Convolution: channel 3->32
            MyConv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            
            # Maxpoooling: size 32x32->16x16
            MyMaxPool2D(kernel_size=2, stride=2),
            
            # Convolution: channel 32->64
            MyConv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            
            # Maxpoooling: size 16x16->8x8
            MyMaxPool2D(kernel_size=2, stride=2),
        )
        
        self.flatten = nn.Flatten()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits

# main
if __name__ == "__main__":
    # Hyper parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    

    train_loader, test_loader, classes = get_dataloaders()
    
    model = SimpleCNN().to(DEVICE)
    
    train_losses, test_losses, train_accs, test_accs = train_and_evaluate(model, train_loader, test_loader)
    
    model_type = 'CNN'
    plot_curves(train_losses, test_losses, train_accs, test_accs, model_type)
    visualize_predictions(model, test_loader, classes, model_type)
    
    # Save model
    torch.save(model.state_dict(), "cnn_model.pth")
    print("Training complete and model saved.")
