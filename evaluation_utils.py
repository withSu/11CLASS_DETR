import os
import json
import cv2
import numpy as np
from collections import defaultdict

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_iou(box1, box2):
    x1, y1, w1, h1 = map(float, box1)
    x2, y2, w2, h2 = map(float, box2)
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = (w1 * h1) + (w2 * h2) - intersection_area
    if union_area < 1e-7:
        return 0.0
    
    iou = intersection_area / union_area
    return min(iou, 1.0)

def match_boxes(gt_list, pred_list, iou_threshold=0.75):
    matched_gt, matched_pred = set(), set()
    for gt_idx, gt in enumerate(gt_list):
        best_iou = 0.0
        best_pred_idx = -1
        for pred_idx, pred in enumerate(pred_list):
            if pred_idx in matched_pred:
                continue
            iou_val = compute_iou(gt["bbox"], pred["bbox"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_pred_idx = pred_idx
        if best_iou >= iou_threshold and best_pred_idx != -1:
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred_idx)
    return matched_gt, matched_pred

def draw_boxes(image, gt_list, pred_list, matched_gt, matched_pred):
    for gt_idx, gt in enumerate(gt_list):
        x, y, w, h = map(int, gt["bbox"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
    for pred_idx, pred in enumerate(pred_list):
        x, y, w, h = map(int, pred["bbox"])
        color = (0, 255, 0) if pred_idx in matched_pred else (0, 255, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
    for gt_idx, gt in enumerate(gt_list):
        if gt_idx not in matched_gt:
            x, y, w, h = map(int, gt["bbox"])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return image

def evaluate_map(gt_json_path, pred_json_path, images_dir, output_dir, iou_threshold=0.75):
    category_map = {
        'Chip': 0,
        'CSolder': 1,
        '2sideIC': 2,
        'SOD': 3,
        'Circle': 4,
        '4sideIC': 5,
        'Tantalum': 6,
        'BGA': 7,
        'MELF': 8,
        'Crystal': 9,
        'Array': 10
    }
    category_id_to_name = {v: k for k, v in category_map.items()}
    
    gt_data = load_json(gt_json_path)
    pred_data = load_json(pred_json_path)
    
    image_id_to_file = {img["id"]: img["file_name"] for img in gt_data["images"]}

    class_total = defaultdict(int)
    class_matched = defaultdict(int)
    
    region_total = {"small": 0, "medium": 0, "large": 0}
    region_matched = {"small": 0, "medium": 0, "large": 0}
    
    total_gt_boxes = 0
    total_missed_boxes = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    for image_id, file_name in image_id_to_file.items():
        image_path = os.path.join(images_dir, file_name)
        if not os.path.exists(image_path):
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        gt_list = [ann for ann in gt_data["annotations"] if ann["image_id"] == image_id]
        pred_list = [ann for ann in pred_data if ann["image_id"] == image_id]
        
        matched_gt, matched_pred = match_boxes(gt_list, pred_list, iou_threshold)
        
        vis_image = draw_boxes(image, gt_list, pred_list, matched_gt, matched_pred)
        #cv2.imwrite(os.path.join(output_dir, file_name), vis_image)
        
        missed_count = len(gt_list) - len(matched_gt)
        total_gt_boxes += len(gt_list)
        total_missed_boxes += missed_count
        
        for gt_idx, gt in enumerate(gt_list):
            cat_id = gt["category_id"]
            if cat_id not in category_id_to_name:
                continue
            cat_name = category_id_to_name[cat_id]
            
            class_total[cat_name] += 1
            
            x, y, w, h = gt["bbox"]
            area = w * h
            if area < 1024:
                region = "small"
            elif area < 9216:
                region = "medium"
            else:
                region = "large"
            region_total[region] += 1
            
            if gt_idx in matched_gt:
                class_matched[cat_name] += 1
                region_matched[region] += 1
    
    print("==== 평가 결과 ====")
    print(f"총 GT 박스: {total_gt_boxes}")
    print(f"누락된 박스: {total_missed_boxes}")
    if total_gt_boxes > 0:
        miss_ratio = (total_missed_boxes / total_gt_boxes) * 100
        print(f"누락 비율: {miss_ratio:.2f}%")
    
    print("\n==== 영역별 평가 결과 ====")
    for region in ["small", "medium", "large"]:
        total = region_total[region]
        matched = region_matched[region]
        ratio = (matched / total * 100) if total > 0 else 0
        print(f"{region.capitalize()} 영역 - 총: {total}, 검출: {matched}, 검출율: {ratio:.2f}%")
    
    print("\n==== 클래스별 평가 결과 ====")
    for cat_name in category_map.keys():
        total_c = class_total[cat_name]
        matched_c = class_matched[cat_name]
        ratio_c = (matched_c / total_c * 100) if total_c > 0 else 0
        print(f"{cat_name} - 총: {total_c}, 검출: {matched_c}, 검출율: {ratio_c:.2f}%")
