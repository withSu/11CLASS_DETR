import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(json_dir, image_dir, output_dir, train_ratio=0.8):
    # 출력 폴더 경로 설정
    train_json_dir = os.path.join(output_dir, "2_1_train_json")
    val_json_dir   = os.path.join(output_dir, "2_2_val_json")
    train_image_dir = os.path.join(output_dir, "2_3_train_image")
    val_image_dir   = os.path.join(output_dir, "2_4_val_image")

    # 출력 폴더 생성 (없으면 새로 생성)
    for folder in [train_json_dir, val_json_dir, train_image_dir, val_image_dir]:
        os.makedirs(folder, exist_ok=True)
        print(f"생성된 폴더: {folder}")

    # JSON 파일 목록 (확장자가 .json 인 파일들)
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    print(f"총 JSON 파일 개수: {len(json_files)}")

    # train_test_split을 사용하여 80:20으로 분할
    train_files, val_files = train_test_split(
        json_files, test_size=(1 - train_ratio), random_state=42, shuffle=True
    )
    print(f"학습용 JSON 파일: {len(train_files)}개, 검증용 JSON 파일: {len(val_files)}개")

    # 파일 복사 함수
    def copy_files(file_list, json_dest, image_dest):
        for json_file in file_list:
            base_name = os.path.splitext(json_file)[0]
            json_src_path = os.path.join(json_dir, json_file)
            # 이미지 파일은 .jpg, .jpeg, .png 중 하나라고 가정
            image_src_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = os.path.join(image_dir, base_name + ext)
                if os.path.exists(candidate):
                    image_src_path = candidate
                    break
            if image_src_path is None:
                print(f"Warning: {json_file} 에 대응하는 이미지가 존재하지 않습니다.")
                continue

            shutil.copy2(json_src_path, os.path.join(json_dest, json_file))
            shutil.copy2(image_src_path, os.path.join(image_dest, os.path.basename(image_src_path)))
            print(f"복사 완료: {json_file} 와 {os.path.basename(image_src_path)}")

    # 학습용 파일 복사
    copy_files(train_files, train_json_dir, train_image_dir)
    # 검증용 파일 복사
    copy_files(val_files, val_json_dir, val_image_dir)

    print("데이터셋 분할 및 복사 완료!")

if __name__ == '__main__':
    # 입력 경로 (원본 JSON 파일과 이미지 파일이 저장된 폴더)
    json_directory = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/1_3_json_800"
    image_directory = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/1_2_800images"
    # 출력 경로 (4개 폴더가 생성될 최상위 폴더)
    output_directory = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets"

    split_dataset(json_directory, image_directory, output_directory, train_ratio=0.8)
