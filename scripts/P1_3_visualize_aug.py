import os
import json
import cv2
import random
import matplotlib.pyplot as plt
from glob import glob

print("증강된 결과의 바운딩 박스 시각화 코드를 실행한다.")

# 증강된 이미지와 JSON 파일이 저장된 디렉토리
augmented_dir = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/3_1_aug_train_enhanced"

# augmented_dir 내의 증강된 이미지 파일(.jpg) 목록을 가져온다.
aug_image_files = glob(os.path.join(augmented_dir, "aug_*.jpg"))
print(f"총 {len(aug_image_files)}개의 증강 이미지가 있습니다.")

# 50장 무작위 샘플 선택 (50장 미만이면 모두 사용)
if len(aug_image_files) < 50:
    sample_files = aug_image_files
    print(f"주의: 50장보다 적은 이미지({len(aug_image_files)}개)를 사용합니다.")
else:
    sample_files = random.sample(aug_image_files, 50)

# matplotlib 서브플롯으로 5행 10열 그리드 구성
fig, axes = plt.subplots(5, 10, figsize=(20, 10))
axes = axes.flatten()

for ax, img_path in zip(axes, sample_files):
    # OpenCV로 이미지 읽기 및 BGR -> RGB 변환
    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지 파일명과 동일한 이름의 JSON 파일 찾기
    base_name = os.path.splitext(os.path.basename(img_path))[0]  # 예: "aug_1_filename"
    json_path = os.path.join(augmented_dir, base_name + ".json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSON 파일의 "shapes" 리스트에서 각 객체에 대해 바운딩 박스 그리기
        for shape in data.get("shapes", []):
            points = shape.get("points", [])
            if len(points) >= 2:
                # 두 점을 이용하여 좌상단, 우하단 좌표 계산
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_min = int(min(x1, x2))
                y_min = int(min(y1, y2))
                x_max = int(max(x1, x2))
                y_max = int(max(y1, y2))
                label = shape.get("label", "unknown")
                # 사각형 그리기 (녹색) 및 라벨 텍스트 표시
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, label, (x_min, max(y_min-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print(f"Warning: {base_name}.json 파일이 존재하지 않습니다.")
    
    ax.imshow(image)
    ax.axis("off")

plt.tight_layout()
plt.show()
