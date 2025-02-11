import os
import json
import random
import cv2

def visualize_train_bboxes(train_json_dir, train_image_dir, output_dir, sample_count=50):
    os.makedirs(output_dir, exist_ok=True)
    
    # 학습용 이미지 폴더에서 이미지 파일 목록 (jpg, png, jpeg)
    image_files = [f for f in os.listdir(train_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) < sample_count:
        sample_files = image_files
        print(f"주의: 이미지가 {sample_count}장보다 적습니다. 전체 {len(image_files)}장으로 진행합니다.")
    else:
        sample_files = random.sample(image_files, sample_count)
    
    for image_file in sample_files:
        base = os.path.splitext(image_file)[0]
        image_path = os.path.join(train_image_dir, image_file)
        json_path = os.path.join(train_json_dir, base + ".json")
        
        if not os.path.exists(json_path):
            print(f"Warning: {image_file}에 대응하는 JSON 파일이 존재하지 않습니다.")
            continue
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: 이미지를 읽을 수 없습니다: {image_path}")
            continue
        
        # JSON 파일 로드 (LabelMe 형식 가정)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # JSON의 "shapes" 리스트 내의 각 객체에 대해 바운딩 박스 그리기
        for shape in data.get("shapes", []):
            label = shape.get("label", "unknown")
            points = shape.get("points", [])
            if len(points) < 2:
                continue
            # points[0]과 points[1]을 이용하여 좌상단, 우하단 좌표 계산
            x1, y1 = points[0]
            x2, y2 = points[1]
            x_min = int(min(x1, x2))
            y_min = int(min(y1, y2))
            x_max = int(max(x1, x2))
            y_max = int(max(y1, y2))
            
            # 바운딩 박스 그리기 (녹색)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, label, (x_min, max(y_min - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        output_path = os.path.join(output_dir, base + "_visualized.jpg")
        cv2.imwrite(output_path, image)
        print(f"시각화된 이미지 저장 완료: {output_path}")

if __name__ == "__main__":
    # 분할된 학습용 데이터 폴더 (사용자가 지정한 경로로 변경)
    train_json_dir = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/2_1_train_json"
    train_image_dir = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/2_3_train_image"
    output_dir = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/2_5_visualized_before_aug"
    
    visualize_train_bboxes(train_json_dir, train_image_dir, output_dir, sample_count=50)
