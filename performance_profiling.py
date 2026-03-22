import time
import os
import torch
import torch
import torchvision
from torchvision.datasets import CocoDetection
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

COCO_VAL_PATH = "/content/coco/val2017"
COCO_ANN_PATH = "/content/coco/annotations/instances_val2017.json"

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# Load dataset
coco_dataset = CocoDetection(root=COCO_VAL_PATH, annFile=COCO_ANN_PATH, transform=transform)

# Randomly pick 100 images for latency 
indices = np.random.choice(len(coco_dataset), 100, replace=False)
subset_indices = indices.tolist()


def measure_latency(model, device, input_size=(3, 640, 640)):
    model.eval()
    model.to(device)
    
    # random input
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # measure 100 times
    start_time = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy_input)
    end_time = time.perf_counter()
    
    avg_latency = (end_time - start_time) / 100
    print(f"Average Latency: {avg_latency:.4f} seconds")
    return avg_latency



# Helper functions of mAP
# ==============================================================================

def calculate_iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
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
    detections: lists with elements in [score, box, image_id]
    ground_truths: lists with elements in [box, image_id, matched_flag]
    """
    
    detections.sort(key=lambda x: x[0], reverse=True)
    
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    total_gts = len(ground_truths)
    
    if total_gts == 0:
        return 0

    gt_matched = {img_id: np.zeros(sum(1 for gt in ground_truths if gt[1] == img_id)) 
                  for _, img_id, _ in ground_truths}
    
    # Group GT by ID
    gt_by_img = {}
    for i, (box, img_id, _) in enumerate(ground_truths):
        if img_id not in gt_by_img: gt_by_img[img_id] = []
        gt_by_img[img_id].append([box, i])

    for i, (score, det_box, img_id) in enumerate(detections):
        best_iou = 0
        best_gt_idx = -1
        
        # Seek best fit GT in this image
        if img_id in gt_by_img:
            for j, (gt_box, original_idx) in enumerate(gt_by_img[img_id]):
                iou = calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_iou >= iou_threshold:
            if not gt_matched[img_id][best_gt_idx]:
                tp[i] = 1
                gt_matched[img_id][best_gt_idx] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

  
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / total_gts
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    # AUC
    m_rec = np.concatenate(([0.0], recalls, [1.0]))
    m_pre = np.concatenate(([1.0], precisions, [0.0]))
    
    # Precision curve decrease
    for i in range(len(m_pre) - 2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    
    # calculate area
    indices = np.where(m_rec[1:] != m_rec[:-1])[0]
    ap = np.sum((m_rec[indices + 1] - m_rec[indices]) * m_pre[indices + 1])
    return ap

# ==============================================================================


def run_yolo_pipeline(dataset, indices):
    model = YOLO('yolov8n.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Measure latency
    latency = measure_latency(model, device)

    # Define mapping list to solve imcampatible ID between YOLO and COCO
    coco80_to_coco91 = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
        27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 
        78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]


    all_detections = {cat_id: [] for cat_id in range(100)} # differ in catagory
    all_gts = {cat_id: [] for cat_id in range(100)}
    
    print("Running YOLOv8n Inference...")
    for idx in tqdm(indices):
        img_tensor, targets = dataset[idx]
        img_pil = torchvision.transforms.ToPILImage()(img_tensor)
        
        
        results = model.predict(img_pil, conf=0.001, device=device, verbose=False)
        
        # YOLOv8 output: [x1, y1, x2, y2]
        for box in results[0].boxes:
            cls_idx = int(box.cls[0])
            cls = coco80_to_coco91[cls_idx]
            score = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy().tolist()
            all_detections[cls].append([score, coords, idx])
            
        # Ground Truth (COCO in [x, y, w, h])
        for target in targets:
            cls = target['category_id']
            # [x1, y1, x2, y2]
            x, y, w, h = target['bbox']
            gt_box = [x, y, x + w, y + h]
            all_gts[cls].append([gt_box, idx, 0])

    return all_detections, all_gts, latency


def run_faster_rcnn_pipeline(dataset, indices):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
 
    latency = measure_latency(model, device, input_size=(3, 800, 800))

    all_detections = {}
    all_gts = {}
    
    print("Running Faster R-CNN Inference...")
    for idx in tqdm(indices):
        img_tensor, targets = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
 
        with torch.no_grad():
            preds = model(img_tensor)[0]
        
        # Faster R-CNN output: [x1, y1, x2, y2])
        for i in range(len(preds['boxes'])):
            cls = int(preds['labels'][i])
            score = float(preds['scores'][i])
            box = preds['boxes'][i].cpu().numpy().tolist()
            if cls not in all_detections: all_detections[cls] = []
            all_detections[cls].append([score, box, idx])
            
   
        for target in targets:
            cls = target['category_id']
            x, y, w, h = target['bbox']
            gt_box = [x, y, x + w, y + h]
            if cls not in all_gts: all_gts[cls] = []
            all_gts[cls].append([gt_box, idx, 0])

    return all_detections, all_gts, latency


def evaluate_model(all_detections, all_gts):
    aps = []
    # traverse every catgory ID
    unique_categories = set(all_gts.keys())
    for cat_id in unique_categories:
        det = all_detections.get(cat_id, [])
        gt = all_gts.get(cat_id, [])
        if len(gt) > 0:
            ap = compute_ap(det, gt, iou_threshold=0.5)
            aps.append(ap)
    
    return np.mean(aps) if aps else 0


# ==============================================================================

if __name__ == "__main__":
    # Download repo
    coco_root = "./coco"
    val_img_dir = os.path.join(coco_root, "val2017")
    ann_file = os.path.join(coco_root, "annotations/instances_val2017.json")

   
    if not os.path.exists(ann_file):
        print("Dataset not found. Commencing automated retrieval via wget...")
        
    
        os.makedirs(coco_root, exist_ok=True)
        
        # Annotations
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        os.system(f"wget -c {ann_url}")
        os.system(f"unzip -q annotations_trainval2017.zip -d {coco_root}")
        os.system("rm annotations_trainval2017.zip")

        # Validation Images
        img_url = "http://images.cocodataset.org/zips/val2017.zip"
        os.system(f"wget -c {img_url}")
        os.system(f"unzip -q val2017.zip -d {coco_root}")
        os.system("rm val2017.zip")
        print("Data retrieval and extraction completed.")
    else:
        print("Dataset already exists. Skipping download.")

    
    print("Initializing COCO dataset...")
    dataset = CocoDetection(root=val_img_dir, annFile=ann_file, transform=transform)
    
    
    indices = np.random.choice(len(dataset), 100, replace=False).tolist()
    results_summary = {}

    # YOLOv8n
    print("\n[Stage 1/2] Evaluating YOLOv8n...")
    yolo_det, yolo_gt, yolo_lat = run_yolo_pipeline(dataset, indices)
    yolo_map = evaluate_model(yolo_det, yolo_gt)
    results_summary['YOLOv8n'] = {'mAP50': yolo_map, 'Latency': yolo_lat}


    torch.cuda.empty_cache()

    # Faster R-CNN
    print("\n[Stage 2/2] Evaluating Faster R-CNN...")
    frcnn_det, frcnn_gt, frcnn_lat = run_faster_rcnn_pipeline(dataset, indices)
    frcnn_map = evaluate_model(frcnn_det, frcnn_gt)
    results_summary['Faster R-CNN'] = {'mAP50': frcnn_map, 'Latency': frcnn_lat}

    # Final Summary
    print("\n" + "="*55)
    print(f"{'Model Arch':<18} | {'mAP50':<10} | {'Avg Latency (s)':<15}")
    print("-" * 55)
    for model, metrics in results_summary.items():
        print(f"{model:<18} | {metrics['mAP50']:<10.4f} | {metrics['Latency']:<15.4f}")
    print("="*55)