import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Load pretrained ResNet50
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.to(device)
model.eval()

weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
categories = weights.meta["categories"]


# Image preprocessing
# ==============================================================================
def get_baseline_detector(img_path, threshold=0.5):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    # Baseline: 5x5 non-overlapping slice
    # ==============================================================================
    patch_w = W // 5
    patch_h = H // 5
    
    found_objects = []

    
    for i in range(5): # row
        for j in range(5): # col
            # position of current window (left, top, right, bottom)
            left = j * patch_w
            top = i * patch_h
            right = left + patch_w
            bottom = top + patch_h
            
            patch = img.crop((left, top, right, bottom))
            input_tensor = transform(patch).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                score, class_id = torch.max(probs, dim=0)
            
            # valid box
            if score > threshold:
                label = categories[class_id]
                found_objects.append({
                    'box': [left, top, patch_w, patch_h],
                    'label': label,
                    'score': score.item()
                })
                

                rect = patches.Rectangle((left, top), patch_w, patch_h, 
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(left, top, f"{label}: {score:.2f}", 
                         color='white', verticalalignment='top', 
                         bbox={'color': 'red', 'alpha': 0.5, 'pad': 0})

    plt.axis('off')
    plt.title(f"Baseline 5x5 Detection on {img_path}")
    plt.savefig(f"baseline_{img_path.split('/')[-1]}")
    plt.show()

# main
# ==============================================================================
if __name__ == "__main__":
    get_baseline_detector("cats1.jpg", threshold=0.3) 
    get_baseline_detector("dogs1.jpg", threshold=0.3)
 