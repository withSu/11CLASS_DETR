import os
import cv2
import json
import random
import numpy as np
import albumentations as A
from glob import glob

print("Initializing Enhanced Data Augmentation...")

# 원본 이미지 및 JSON 경로 설정
input_images = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/2_3_train_image"
input_json   = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/2_1_train_json"

# 증강 결과를 저장할 폴더
output_dir = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/3_1_aug_train_enhanced"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# 증강 횟수를 늘려서 대략 50배까지 생성
# 예: 원본 이미지가 100장이면, 1장당 50장씩 증강하여 총 5000장
num_augmentations = 50

# Albumentations 기반의 기본 증강 파이프라인
# 작은 객체에 도움이 되는 RandomScale 등 추가
def get_base_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1,
                           scale_limit=0.2,
                           rotate_limit=15,
                           interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_CONSTANT,
                           p=0.7),
        A.RandomSizedBBoxSafeCrop(height=640, width=640, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomScale(scale_limit=(0.5, 2.0), p=0.4)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.3))

# 작은 객체 주변만 확대하는 Selective Zoom 함수
def selective_zoom(image, bboxes, labels, zoom_factor=1.5):
    if not bboxes:
        return image, bboxes, labels
    # 작은 객체 임의 선택
    chosen_idx = random.randint(0, len(bboxes)-1)
    x_min, y_min, x_max, y_max = bboxes[chosen_idx]
    h, w = image.shape[:2]

    obj_width = x_max - x_min
    obj_height = y_max - y_min

    # 객체 주변에 여유를 두고 확대
    pad = 10
    x1 = max(0, int(x_min - pad))
    y1 = max(0, int(y_min - pad))
    x2 = min(w, int(x_max + pad))
    y2 = min(h, int(y_max + pad))

    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return image, bboxes, labels

    # 확대
    new_w = int(region.shape[1] * zoom_factor)
    new_h = int(region.shape[0] * zoom_factor)
    zoomed_region = cv2.resize(region, (new_w, new_h))

    # 확대 후, 원본 이미지 중앙 근처에 붙여넣기 (혹은 랜덤 위치)
    paste_x = random.randint(0, max(0, w - new_w))
    paste_y = random.randint(0, max(0, h - new_h))

    # 원본 이미지를 복사해서 그 위에 확대된 영역을 덮어쓰기
    out_image = image.copy()
    out_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = zoomed_region

    # 원본 bbox 좌표를 새 위치로 변환
    dx = paste_x - x1
    dy = paste_y - y1

    new_bboxes = []
    for i, (bx_min, by_min, bx_max, by_max) in enumerate(bboxes):
        if i == chosen_idx:
            bw = bx_max - bx_min
            bh = by_max - by_min
            scale_x = zoom_factor
            scale_y = zoom_factor

            new_xmin = paste_x
            new_ymin = paste_y
            new_xmax = paste_x + int(bw * scale_x)
            new_ymax = paste_y + int(bh * scale_y)
            new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
        else:
            # 다른 객체는 그대로
            new_bboxes.append([bx_min, by_min, bx_max, by_max])

    return out_image, new_bboxes, labels

# Mosaic 증강을 위한 함수
# 4장을 결합해 하나의 이미지로 만들고, bbox도 재조정
def mosaic_augmentation(image_paths, json_paths):
    selected_indices = random.sample(range(len(image_paths)), 4)
    imgs = []
    bboxes_list = []
    labels_list = []

    # 4장의 이미지와 bbox, label 불러오기
    for idx in selected_indices:
        img = cv2.imread(image_paths[idx])
        if img is None:
            continue
        with open(json_paths[idx], 'r', encoding='utf-8') as f:
            data = json.load(f)
        bboxes = []
        labels = []
        for shape in data.get("shapes", []):
            pts = shape["points"]
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            labels.append(shape.get("label", "unknown"))
        imgs.append(img)
        bboxes_list.append(bboxes)
        labels_list.append(labels)

    if len(imgs) < 4:
        return None, None, None

    # 각각의 이미지를 640x640으로 임시 리사이즈
    resized_imgs = []
    for img in imgs:
        resized_imgs.append(cv2.resize(img, (640, 640)))
    # 하나의 이미지로 붙이기 (2x2)
    # 최종 크기는 1280x1280
    final_img = np.zeros((1280, 1280, 3), dtype=np.uint8)

    # 좌상단, 우상단, 좌하단, 우하단 순서로 배치
    final_img[0:640, 0:640] = resized_imgs[0]
    final_img[0:640, 640:1280] = resized_imgs[1]
    final_img[640:1280, 0:640] = resized_imgs[2]
    final_img[640:1280, 640:1280] = resized_imgs[3]

    # bbox 좌표 보정
    final_bboxes = []
    final_labels = []
    offsets = [(0, 0), (640, 0), (0, 640), (640, 640)]
    for i, (bboxes, labels) in enumerate(zip(bboxes_list, labels_list)):
        ox, oy = offsets[i]
        for (xmin, ymin, xmax, ymax), label in zip(bboxes, labels):
            # 리사이즈되었으므로 비율 보정
            scale_x = 640.0 / imgs[i].shape[1]
            scale_y = 640.0 / imgs[i].shape[0]
            new_xmin = int(xmin * scale_x + ox)
            new_ymin = int(ymin * scale_y + oy)
            new_xmax = int(xmax * scale_x + ox)
            new_ymax = int(ymax * scale_y + oy)
            final_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
            final_labels.append(label)

    return final_img, final_bboxes, final_labels

# Copy-Paste 기법을 사용하여 작은 객체를 다른 위치에 복사해 붙이는 함수
def copy_paste_augmentation(image, bboxes, labels, max_paste=3):
    out_image = image.copy()
    out_bboxes = list(bboxes)
    out_labels = list(labels)

    if len(bboxes) == 0:
        return image, bboxes, labels

    h, w = image.shape[:2]
    num_paste = random.randint(1, max_paste)
    for _ in range(num_paste):
        idx = random.randint(0, len(bboxes)-1)
        x_min, y_min, x_max, y_max = bboxes[idx]

        # 정수 변환
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        obj = image[y_min:y_max, x_min:x_max].copy()
        obj_h, obj_w = obj.shape[:2]

        # 붙여넣을 위치
        paste_x = random.randint(0, max(0, w - obj_w))
        paste_y = random.randint(0, max(0, h - obj_h))

        out_image[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = obj

        # 새로운 bbox
        out_bboxes.append([paste_x, paste_y, paste_x + obj_w, paste_y + obj_h])
        out_labels.append(labels[idx])

    return out_image, out_bboxes, out_labels
def augment_data(image_path, json_path, output_dir, image_paths, json_paths):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bboxes = []
    labels = []
    for shape in data.get("shapes", []):
        pts = shape["points"]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)
        bboxes.append([x_min, y_min, x_max, y_max])
        labels.append(shape.get("label", "unknown"))

    if not bboxes:
        return

    # 50회 반복
    successful = 0
    attempts = 0
    max_attempts = num_augmentations * 3

    while successful < num_augmentations and attempts < max_attempts:
        attempts += 1

        # 1) Mosaic 사용 확률
        # 2) Copy-Paste 사용 확률
        # 3) Selective Zoom 사용 확률
        # 4) 기본 Augmentation
        # 무작위로 기법 하나 선택
        choice = random.random()

        if choice < 0.2:
            # Mosaic
            mosaic_img, mosaic_bboxes, mosaic_labels = mosaic_augmentation(image_paths, json_paths)
            if mosaic_img is None:
                continue
            # Mosaic 결과도 Albumentations 파이프라인 적용
            transform = get_base_augmentation()
            try:
                augmented = transform(image=mosaic_img, bboxes=mosaic_bboxes, labels=mosaic_labels)
            except ValueError:
                continue
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["labels"]
        elif choice < 0.4:
            # Copy-Paste
            cp_img, cp_bboxes, cp_labels = copy_paste_augmentation(image, bboxes, labels, max_paste=3)
            transform = get_base_augmentation()
            try:
                augmented = transform(image=cp_img, bboxes=cp_bboxes, labels=cp_labels)
            except ValueError:
                continue
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["labels"]
        elif choice < 0.6:
            # Selective Zoom
            zoom_img, zoom_bboxes, zoom_labels = selective_zoom(image, bboxes, labels, zoom_factor=1.5)
            transform = get_base_augmentation()
            try:
                augmented = transform(image=zoom_img, bboxes=zoom_bboxes, labels=zoom_labels)
            except ValueError:
                continue
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["labels"]
        else:
            # 기본 Albumentations
            transform = get_base_augmentation()
            try:
                augmented = transform(image=image, bboxes=bboxes, labels=labels)
            except ValueError:
                continue
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["labels"]

        # Augmented JSON 생성
        aug_data = data.copy()
        new_shapes = []
        for bbox, lbl in zip(aug_bboxes, aug_labels):
            new_shape = {
                "label": lbl,
                "points": [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            new_shapes.append(new_shape)
        aug_data["shapes"] = new_shapes

        # 파일 저장
        aug_image_filename = f"aug_{successful+1}_" + os.path.basename(image_path)
        aug_json_filename = f"aug_{successful+1}_" + os.path.basename(json_path)
        aug_image_path = os.path.join(output_dir, aug_image_filename)
        aug_json_path = os.path.join(output_dir, aug_json_filename)

        cv2.imwrite(aug_image_path, aug_image)
        with open(aug_json_path, 'w', encoding='utf-8') as f:
            json.dump(aug_data, f, indent=4)

        successful += 1

    if successful < num_augmentations:
        print(f"Warning: Only {successful} augmentations generated for {image_path}.")

# 메인 실행부
image_paths = glob(os.path.join(input_images, "*.jpg"))
json_files = []
for path in image_paths:
    base_name = os.path.splitext(os.path.basename(path))[0]
    json_path = os.path.join(input_json, base_name + ".json")
    json_files.append(json_path)

print(f"Found {len(image_paths)} images.")

for img_path, js_path in zip(image_paths, json_files):
    if os.path.exists(js_path):
        augment_data(img_path, js_path, output_dir, image_paths, json_files)
    else:
        print(f"Warning: JSON file not found for {img_path}")
