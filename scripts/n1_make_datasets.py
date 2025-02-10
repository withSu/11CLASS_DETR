import os
import json
import random
from PIL import Image
from sklearn.model_selection import train_test_split  # 추가

def split_and_convert_to_coco(json_dir,
                              image_dir,
                              output_dir,
                              train_ratio=0.8,
                              original_width=3904,
                              original_height=3904,
                              target_width=800,
                              target_height=800):

    # 클래스 이름과 ID 매핑
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
        'Array': 10,
    }

    # COCO 포맷 초기화 함수
    def initialize_coco():
        return {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": cat_id, "name": name} for name, cat_id in category_map.items()
            ]
        }

    train_dir = os.path.join(output_dir, "train_images")
    val_dir = os.path.join(output_dir, "val_images")

    train_annotations_file = os.path.join(output_dir, "annotations", "train.json")
    val_annotations_file = os.path.join(output_dir, "annotations", "val.json")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # JSON 파일 목록
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    # scikit-learn의 train_test_split 사용
    train_files, val_files = train_test_split(
        json_files,
        test_size=1 - train_ratio,
        random_state=42,  # 원하는 정수값을 지정하면 매번 같은 결과를 얻을 수 있음
        shuffle=True
    )

    train_coco = initialize_coco()
    val_coco = initialize_coco()

    image_id = 1
    annotation_id = 1

    # 스케일 비율
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    def process_files(file_list, target_dir, coco_format):
        nonlocal image_id, annotation_id
        for json_filename in file_list:
            json_filepath = os.path.join(json_dir, json_filename)

            with open(json_filepath, 'r') as f:
                input_json = json.load(f)

            image_filename_base = json_filename.replace('.json', '')
            image_filename = None

            for ext in ['.jpg', '.png', '.jpeg']:
                candidate = os.path.join(image_dir, image_filename_base + ext)
                if os.path.isfile(candidate):
                    image_filename = image_filename_base + ext
                    break

            if not image_filename:
                print(f"Warning: No matching image found for {json_filename}")
                continue

            original_image_path = os.path.join(image_dir, image_filename)
            output_image_path = os.path.join(target_dir, image_filename)

            try:
                with Image.open(original_image_path) as img:
                    resized_img = img.resize((target_width, target_height), Image.LANCZOS)
                    resized_img.save(output_image_path)
            except Exception as e:
                print(f"Error resizing image {image_filename}: {e}")
                continue

            coco_format["images"].append({
                "id": image_id,
                "file_name": image_filename,
                "width": target_width,
                "height": target_height
            })

            for shape in input_json.get("shapes", []):
                label = shape.get("label", None)
                if label not in category_map:
                    print(f"Warning: Undefined label '{label}' in {json_filename}, skipping annotation.")
                    continue

                # points[0], points[1]에 바운딩 박스 좌표가 담긴 것으로 가정
                x1, y1 = shape["points"][0]
                x2, y2 = shape["points"][1]

                x1_resized = x1 * scale_x
                y1_resized = y1 * scale_y
                x2_resized = x2 * scale_x
                y2_resized = y2 * scale_y

                bbox = [
                    min(x1_resized, x2_resized),
                    min(y1_resized, y2_resized),
                    abs(x2_resized - x1_resized),
                    abs(y2_resized - y1_resized)
                ]

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_map[label],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1

            image_id += 1

    process_files(train_files, train_dir, train_coco)
    process_files(val_files, val_dir, val_coco)

    with open(train_annotations_file, 'w') as f:
        json.dump(train_coco, f, indent=4)
    with open(val_annotations_file, 'w') as f:
        json.dump(val_coco, f, indent=4)

    print(f"Train images: {len(train_files)}, Val images: {len(val_files)}")
    print("COCO format JSON files created with resized images and multi-class labels!")

if __name__ == '__main__':
    json_directory = '/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/2_raw_json'
    image_directory = '/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/1_2_800images'
    output_directory = '/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/datasets'

    split_and_convert_to_coco(
        json_directory,
        image_directory,
        output_directory,
        train_ratio=0.8,
        original_width=3904,
        original_height=3904,
        target_width=800,
        target_height=800
    )
