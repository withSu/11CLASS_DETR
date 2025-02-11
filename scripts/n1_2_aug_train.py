import os
import cv2
import json
import albumentations as A
from glob import glob

print("Initializing Data Augmentation for DETR JSON...")

# 원본 이미지 및 JSON(라벨) 경로 설정
input_images = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/2_3_train_image"
input_json   = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/2_1_train_json"
# 증강 결과를 저장할 출력 폴더 설정
output_dir = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/3_1_aug_train"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# 다양한 증강 기법을 적용하는 함수 (bbox는 pascal_voc 형식)
def get_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15,
                           interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomSizedBBoxSafeCrop(height=640, width=640, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# 한 이미지에 대해 증강을 적용하고, 증강된 이미지와 JSON 파일을 저장하는 함수
def augment_data(image_path, json_path, output_dir, num_augmentations=12):
    print(f"Processing: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    print(f"Loaded image: {image_path} with shape {image.shape}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JSON에서 "shapes"를 읽어 bbox와 label 목록 생성 (pascal_voc 형식)
    bboxes = []
    labels = []
    if "shapes" in data:
        for shape in data["shapes"]:
            if "points" not in shape or len(shape["points"]) < 2:
                print(f"Warning: Invalid shape in {json_path}")
                continue
            pts = shape["points"]
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(shape.get("label", "unknown"))
    else:
        print(f"Warning: No shapes found in {json_path}")
        return
    
    if not bboxes:
        print(f"Warning: No valid bounding boxes found in {json_path}")
        return
        
    print(f"Applying augmentations to {image_path}")
    successful = 0
    attempts = 0
    max_attempts = num_augmentations * 3  # 최대 재시도 횟수 설정
    while successful < num_augmentations and attempts < max_attempts:
        attempts += 1
        transform = get_augmentation()
        try:
            augmented = transform(image=image, bboxes=bboxes, labels=labels)
        except ValueError as e:
            print(f"Warning: Augmentation failed with error: {e}. Retrying...")
            continue
        
        aug_image = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["labels"]
        
        # 원본 JSON 데이터를 복사한 후, "shapes" 항목을 업데이트
        aug_data = data.copy()
        new_shapes = []
        for bbox, lbl in zip(aug_bboxes, aug_labels):
            # bbox는 [x_min, y_min, x_max, y_max]
            new_shape = {
                "label": lbl,
                "points": [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            new_shapes.append(new_shape)
        aug_data["shapes"] = new_shapes
        
        # 증강된 파일 저장 (파일명 앞에 "aug_{번호}_" 접두어 추가)
        aug_image_filename = f"aug_{successful+1}_" + os.path.basename(image_path)
        aug_json_filename = f"aug_{successful+1}_" + os.path.basename(json_path)
        aug_image_path = os.path.join(output_dir, aug_image_filename)
        aug_json_path = os.path.join(output_dir, aug_json_filename)
        
        cv2.imwrite(aug_image_path, aug_image)
        with open(aug_json_path, 'w', encoding='utf-8') as f:
            json.dump(aug_data, f, indent=4)
        
        print(f"Saved: {aug_image_path} and {aug_json_path}")
        successful += 1
    if successful < num_augmentations:
        print(f"Warning: Only {successful} augmentations generated for {image_path} after {attempts} attempts.")

# 원본 이미지 목록을 glob로 가져옴 (확장자 .jpg 기준)
image_paths = glob(os.path.join(input_images, "*.jpg"))
print(f"Found {len(image_paths)} images")

for img_path in image_paths:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(input_json, base_name + ".json")
    if os.path.exists(json_path):
        augment_data(img_path, json_path, output_dir, num_augmentations=12)
    else:
        print(f"Warning: JSON file not found for {img_path}")
