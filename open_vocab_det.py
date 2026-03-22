import os
import json
import time
import torch
import torchvision
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLOWorld
from torchvision.datasets import CocoDetection
from PIL import Image

# Paths and Constants Configuration
# ==============================================================================
COCO_ROOT = "./coco"
COCO_VAL_IMG_DIR = os.path.join(COCO_ROOT, "val2017")
COCO_ANN_FILE = os.path.join(COCO_ROOT, "annotations/instances_val2017.json")


TEST_IMAGE1_PATH = "cats1.jpg"
TEST_IMAGE2_PATH = "dogs1.jpg"


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# COCO's 80 catagory test, for Open Vocabulary prompt
COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kites", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 
    78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]


# Helper Functions for Manual mAP Calculation )
# ==============================================================================

def calculate_iou(box1, box2):
    """
    Intersection over Union (IoU) is calculated between two bounding boxes.
    Format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def compute_ap(detections, ground_truths, iou_threshold=0.5):
    """
    Average Precision (AP) for a single class is computed based on precision-recall curve interpolation.
    """
    # Detections are sorted by confidence score in descending order
    detections.sort(key=lambda x: x[0], reverse=True)
    
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    total_gts = len(ground_truths)
    
    if total_gts == 0:
        return 0

    # Ground truth matching status is initialized
    gt_matched = {img_id: np.zeros(sum(1 for gt in ground_truths if gt[1] == img_id)) 
                  for _, img_id, _ in ground_truths}
    
    # GTs are grouped by Image ID to optimize matching speed
    gt_by_img = {}
    for i, (box, img_id, _) in enumerate(ground_truths):
        if img_id not in gt_by_img: gt_by_img[img_id] = []
        gt_by_img[img_id].append([box, i])

    # Iteration is performed over all detections
    for i, (score, det_box, img_id) in enumerate(detections):
        best_iou = 0
        best_gt_idx = -1
        
        # The best matching GT within the same image is sought
        if img_id in gt_by_img:
            for j, (gt_box, original_idx) in enumerate(gt_by_img[img_id]):
                iou = calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        # IoU threshold check is performed
        if best_iou >= iou_threshold:
            # Check if this GT has already been assigned to a higher-confidence detection
            if not gt_matched[img_id][best_gt_idx]:
                tp[i] = 1
                gt_matched[img_id][best_gt_idx] = 1 # Assignment is finalized
            else:
                fp[i] = 1 # False Positive due to redundant detection
        else:
            fp[i] = 1 # False Positive due to low IoU

    # Cumulative TP and FP are calculated
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Precisions and Recalls are derived
    recalls = tp_cumsum / total_gts
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    # All-point interpolation method (AUC) is applied for AP calculation
    m_rec = np.concatenate(([0.0], recalls, [1.0]))
    m_pre = np.concatenate(([1.0], precisions, [0.0]))
    
    # Monotonicity is ensured on the precision curve
    for i in range(len(m_pre) - 2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    
    # Area under the curve is calculated
    indices = np.where(m_rec[1:] != m_rec[:-1])[0]
    ap = np.sum((m_rec[indices + 1] - m_rec[indices]) * m_pre[indices + 1])
    return ap


# Main
# ==============================================================================

if __name__ == "__main__":
    print("\n[Start] Task 5: Modern Open Vocabulary Object Detection\n" + "="*60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluation is running on: {device.upper()}")

    # YOLOv8s-world COCO Evaluation ---
    print("\n[PART 1/2] Task 5.1: mAP50 Evaluation on COCO Dataset")
    print("-" * 55)
    
    if not (os.path.exists(COCO_VAL_IMG_DIR) and os.path.exists(COCO_ANN_FILE)):
        print("Error: COCO dataset paths not found. Please ensure data is retrieved as in Deliverable 4.")
    else:
        print("Initializing COCO dataset for PART 1...")
        dataset = CocoDetection(root=COCO_VAL_IMG_DIR, annFile=COCO_ANN_FILE, transform=transform)
        
        indices = np.random.choice(len(dataset), 100, replace=False).tolist()
        print(f"{len(indices)} images are randomly selected for inference.")

        model = YOLOWorld('yolov8s-world.pt')
        model.to(device)
        
        print("Setting offline vocabulary to COCO 80 classes...")
        model.set_classes(COCO_80_CLASSES)

        all_detections = {cat_id: [] for cat_id in range(101)} 
        all_gts = {cat_id: [] for cat_id in range(101)}
        

        print("Running YOLOv8s-world Inference on selected COCO subset...")
        for idx in tqdm(indices):
            img_tensor, targets = dataset[idx]
            img_pil = torchvision.transforms.ToPILImage()(img_tensor)
            
            results = model.predict(img_pil, conf=0.001, device=device, verbose=False)
            
            for box in results[0].boxes:
                cls_idx = int(box.cls[0])
                cls = COCO80_TO_COCO91[cls_idx] 
                
                score = float(box.conf[0])
                # [x, y, x, y]（xyxy）
                coords = box.xyxy[0].cpu().numpy().tolist()
                all_detections[cls].append([score, coords, idx])
                
            # Ground Truth
            for target in targets:
                cls = target['category_id']
                x, y, w, h = target['bbox']
                # [x, y, w, h] -> [x1, y1, x2, y2]
                gt_box = [x, y, x + w, y + h]
                all_gts[cls].append([gt_box, idx, 0])

        print("Computing mAP50 via manual AP implementation...")
        aps = []
        unique_categories = set(all_gts.keys())
        for cat_id in unique_categories:
            det = all_detections.get(cat_id, [])
            gt = all_gts.get(cat_id, [])
            if len(gt) > 0:
                # IoU thres = 0.5
                ap = compute_ap(det, gt, iou_threshold=0.5)
                aps.append(ap)
        
        yolo_world_map50 = np.mean(aps) if aps else 0
        print(f"\nFinal Result: YOLOv8s-world mAP50 (on COCO val 100samples) = {yolo_world_map50:.4f}")

    torch.cuda.empty_cache()


    # Cat & Dog Visualization
    print("\n[PART 2/2] Task 5.2: Open Vocabulary Prediction on Attached Images")
    print("-" * 55)
    
    if not (os.path.exists(TEST_IMAGE1_PATH) and os.path.exists(TEST_IMAGE2_PATH)):
        print(f"Error: Target images not found. Please upload '{TEST_IMAGE1_PATH}' and '{TEST_IMAGE2_PATH}' to current directory.")
    else:
        print("Model vocab being updated with 'cat' and 'dog' as dynamic prompt...")
        model.set_classes(["cat", "dog"])
        
        # Autosave to runs/detect/predict/
        print(f"Inference is performed on {TEST_IMAGE1_PATH}...")
        results1 = model.predict(TEST_IMAGE1_PATH, save=True, conf=0.3, device=device)
        print(f"Result visualization is saved at runs/detect/predict/")
        
        print(f"Inference is performed on {TEST_IMAGE2_PATH}...")
        results2 = model.predict(TEST_IMAGE2_PATH, save=True, conf=0.3, device=device)
        print(f"Result visualization is saved at runs/detect/predict/")


    print("\n[Completed] Task 5 script execution finished.\n" + "="*60)