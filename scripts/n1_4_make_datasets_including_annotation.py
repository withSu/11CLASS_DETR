import os
import json
import shutil
from PIL import Image

def convert_labelme_to_coco(source_json_dir, source_image_dir, output_image_dir, output_annotation_file,
                              target_width=800, target_height=800, category_map=None):
    if category_map is None:
        category_map = {
            'Chip': 0,
            'Solder': 1,
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
    # COCO 포맷 초기화
    def initialize_coco():
        return {
            "images": [],
            "annotations": [],
            "categories": [{"id": cid, "name": name} for name, cid in category_map.items()]
        }
    
    coco_data = initialize_coco()
    image_id = 1
    annotation_id = 1
    # 이미지들은 이미 800×800이므로 scale factor = 1
    scale_x = 1.0
    scale_y = 1.0

    # source_json_dir 내의 모든 JSON 파일 목록 (확장자가 .json인 파일)
    json_files = [f for f in os.listdir(source_json_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {source_json_dir}")
    
    os.makedirs(output_image_dir, exist_ok=True)
    
    for json_filename in json_files:
        json_filepath = os.path.join(source_json_dir, json_filename)
        with open(json_filepath, 'r', encoding='utf-8') as f:
            input_json = json.load(f)
        
        image_filename_base = os.path.splitext(json_filename)[0]
        image_filename = None
        # 이미지 파일은 jpg, jpeg, png 중 하나로 가정
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = os.path.join(source_image_dir, image_filename_base + ext)
            if os.path.isfile(candidate):
                image_filename = image_filename_base + ext
                break
        if not image_filename:
            print(f"Warning: No matching image found for {json_filename}")
            continue
        
        src_image_path = os.path.join(source_image_dir, image_filename)
        dest_image_path = os.path.join(output_image_dir, image_filename)
        try:
            with Image.open(src_image_path) as img:
                # 이미지가 이미 target 크기라면 그대로 저장
                img.save(dest_image_path)
        except Exception as e:
            print(f"Error processing image {image_filename}: {e}")
            continue
        
        # COCO 이미지 정보 추가
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": target_width,
            "height": target_height
        })
        
        # JSON의 "shapes" 항목을 COCO annotation으로 변환
        for shape in input_json.get("shapes", []):
            label = shape.get("label", None)
            if label not in category_map:
                print(f"Warning: Undefined label '{label}' in {json_filename}, skipping annotation.")
                continue
            if "points" not in shape or len(shape["points"]) < 2:
                continue
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
            coco_data["annotations"].append(annotation)
            annotation_id += 1
        image_id += 1

    with open(output_annotation_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)
    print(f"COCO annotation file saved: {output_annotation_file}")

if __name__ == '__main__':
    # 학습용 (증강된 데이터는 train에만 넣음)
    train_source_json = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/3_1_aug_train"
    train_source_image = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/3_1_aug_train"
    output_train_images = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/datasets/train_images"
    output_train_annotation = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/datasets/annotations/train.json"
    
    # 검증용: 별도로 준비된 검증용 JSON 및 이미지
    val_source_json = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/2_2_val_json"
    val_source_image = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/2_4_val_image"
    output_val_images = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/datasets/val_images"
    output_val_annotation = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/datasets/annotations/val.json"
    
    # 출력 폴더 생성
    os.makedirs(output_train_images, exist_ok=True)
    os.makedirs(output_val_images, exist_ok=True)
    os.makedirs(os.path.join("/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/datasets", "annotations"), exist_ok=True)
    
    print("Processing training set...")
    convert_labelme_to_coco(train_source_json, train_source_image, output_train_images, output_train_annotation,
                              target_width=800, target_height=800)
    
    print("Processing validation set...")
    convert_labelme_to_coco(val_source_json, val_source_image, output_val_images, output_val_annotation,
                              target_width=800, target_height=800)
