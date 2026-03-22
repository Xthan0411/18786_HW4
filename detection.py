import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


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



# Improvement: NMS
def nms(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0: return []
    
    # [x1, y1, x2, y2]
    boxes = np.array(boxes)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    scores = np.array(scores)

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


# Improved detector: Sliding window
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
model = torchvision.models.resnet50(weights=weights).to(device).eval()
categories = weights.meta["categories"]

def improved_detection(img_path, threshold=0.7):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Multi-size sliding window
    window_sizes = [W//3, W//4]
    stride = 30
    
    candidate_boxes = []
    candidate_scores = []
    candidate_labels = []

    print(f"Begin scanning: {img_path}...")
    for win_size in window_sizes:
        for y in range(0, H - win_size, stride):
            for x in range(0, W - win_size, stride):
                patch = img.crop((x, y, x + win_size, y + win_size))
                input_tensor = transform(patch).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.nn.functional.softmax(output[0], dim=0)
                    score, class_id = torch.max(probs, dim=0)
                
                label = categories[class_id].lower()
                # only focus on cats and dogs

                if score > threshold:
                    is_dog = (151 <= class_id <= 268)
                    is_cat = (281 <= class_id <= 285)
                    if is_dog or is_cat:
                        final_label = "Dog" if is_dog else "Cat"
                        candidate_boxes.append([x, y, win_size, win_size])
                        candidate_scores.append(score.item())
                        candidate_labels.append(f"{final_label} ({categories[class_id]})")

    final_indices = nms(candidate_boxes, candidate_scores, iou_threshold=0.2)
    
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)
    
    for i in final_indices:
        box = candidate_boxes[i]
        label = candidate_labels[i]
        score = candidate_scores[i]
        
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], 
                                 linewidth=3, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1], f"{label} {score:.2f}", color='black', 
                bbox=dict(facecolor='lime', alpha=0.7))

    plt.axis('off')
    plt.title(f"Improved Detection (Sliding Window + NMS) on {img_path}")
    plt.savefig(f"improved_{img_path.split('/')[-1]}")
    plt.show()

# main
# ==============================================================================
if __name__ == "__main__":
    #get_baseline_detector("cats1.jpg", threshold=0.3) 
    #get_baseline_detector("dogs1.jpg", threshold=0.3)
    improved_detection("cats1.jpg", threshold=0.6)
    improved_detection("dogs1.jpg", threshold=0.6)

 