import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

def get_coco_size_label(w, h):
    """COCO 공식 기준에 따라 객체 크기를 분류 (면적 기반)"""
    area = w * h  # 면적 계산
    if area < 32**2:  # 1024 픽셀² 미만
        return "Small"
    elif 32**2 <= area < 96**2:  # 1024 픽셀² 이상, 9216 픽셀² 미만
        return "Medium"
    else:  # 9216 픽셀² 이상
        return "Large"
    
    
    
def visualize_coco_annotations(image_dir, json_file, num_images=5):
    # COCO 어노테이션 JSON 파일 로드
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # 이미지 ID와 파일 이름 매핑
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # 어노테이션 데이터 로드
    annotations = coco_data['annotations']
    
    # 이미지와 어노테이션 매칭
    ann_by_image = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)
    
    # 시각화할 이미지 선택
    for img_id, img_file in list(images.items())[:num_images]:
        img_path = Path(image_dir) / img_file
        if not img_path.exists():
            print(f"이미지 {img_path}를 찾을 수 없습니다.")
            continue
        
        # 이미지 로드
        img = Image.open(img_path)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        # 어노테이션 그리기
        if img_id in ann_by_image:
            for ann in ann_by_image[img_id]:
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                size_label = get_coco_size_label(w, h)
                
                # 바운딩 박스 그리기
                rect = patches.Rectangle(
                    (x, y), w, h, linewidth=1, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                
                # 텍스트 정보 (COCO 크기 분류 & W x H)
                text = f"{size_label}\n{int(w)} x {int(h)}"
                ax.text(
                    x, y, text, color='white', fontsize=6  # backgroundcolor 제거
                )

        plt.axis('off')
        plt.show()

# 이미지 디렉토리와 어노테이션 파일 경로
image_dir = '/home/a/A_2024_selfcode/PCB_proj_DETR/raw_datasets/1_images'
json_file = '/home/a/A_2024_selfcode/PCB_proj_DETR/datasets/annotations/test.json'

# 시각화 실행
visualize_coco_annotations(image_dir, json_file, num_images=5)
