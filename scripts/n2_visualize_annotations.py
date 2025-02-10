import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

def get_coco_size_label(w, h):
    """COCO 기준에 따른 객체 크기 분류 (면적 기반)"""
    area = w * h
    if area < 32**2:  # 1024 미만
        return "Small"
    elif 32**2 <= area < 96**2:  # 1024 이상 9216 미만
        return "Medium"
    else:  # 9216 이상
        return "Large"

def visualize_coco_annotations(image_dir, json_file, num_images=5):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    # 이미지 ID와 파일 이름 매핑
    images = {img['id']: img['file_name'] for img in coco_data['images']}

    # 어노테이션 데이터 매핑
    annotations = coco_data['annotations']
    ann_by_image = {}
    for ann in annotations:
        img_id = ann['image_id']
        ann_by_image.setdefault(img_id, []).append(ann)

    # num_images개의 이미지를 순회하며 시각화
    for img_id, img_file in list(images.items())[:num_images]:
        img_path = Path(image_dir) / img_file
        if not img_path.exists():
            print(f"이미지 {img_path}를 찾을 수 없습니다.")
            continue

        # 이미지 로드
        img = Image.open(img_path)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)

        # 바운딩 박스 시각화
        if img_id in ann_by_image:
            for ann in ann_by_image[img_id]:
                x, y, w, h = ann['bbox']
                size_label = get_coco_size_label(w, h)
                category_id = ann['category_id']
                category_name = next((c['name'] for c in coco_data['categories'] if c['id'] == category_id), "Unknown")

                # 크기에 따른 색상 지정
                if size_label == "Small":
                    color = 'red'
                elif size_label == "Medium":
                    color = 'blue'
                else:
                    color = 'yellow'

                # 바운딩 박스 그리기
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                # 텍스트 추가 (폰트 두께 조정)
                text = f"{category_name} ({size_label})"
                ax.text(
                    x, y - 2, text,
                    color='white', fontsize=8, ha='left', va='bottom'
                )

        plt.axis('off')
        plt.show()

# 실행
image_dir = '/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/datasets/train_images'
json_file = '/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/datasets/annotations/train.json'
visualize_coco_annotations(image_dir, json_file, num_images=10)
