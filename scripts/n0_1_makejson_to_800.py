import os
import json

def convert_json_to_800(source_dir, target_dir, original_width=3904, original_height=3904, target_width=800, target_height=800):
    # 대상 폴더 생성 (없으면 새로 생성)
    os.makedirs(target_dir, exist_ok=True)
    
    # 스케일 비율 계산
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    # source_dir 내의 모든 JSON 파일 목록 (확장자 .json 인 파일)
    json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")

    for json_file in json_files:
        source_path = os.path.join(source_dir, json_file)
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # "shapes" 키가 존재하고 각 shape에 "points"가 있다면 좌표 변환
        if "shapes" in data:
            for shape in data["shapes"]:
                if "points" in shape:
                    new_points = []
                    for point in shape["points"]:
                        x, y = point
                        new_x = x * scale_x
                        new_y = y * scale_y
                        new_points.append([new_x, new_y])
                    shape["points"] = new_points

        target_path = os.path.join(target_dir, json_file)
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"변환 완료: {json_file} -> {target_path}")

if __name__ == '__main__':
    # 원본 JSON 파일이 있는 폴더 (예: 0_원본json)
    source_json_dir = '/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/0_원본json'
    # 변환된 JSON 파일을 저장할 폴더 (원본과 다른 폴더로 지정)
    target_json_dir = '/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/1_3_json_800'
    
    convert_json_to_800(source_json_dir, target_json_dir,
                        original_width=3904, original_height=3904,
                        target_width=800, target_height=800)
