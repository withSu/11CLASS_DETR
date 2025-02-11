import os
from PIL import Image

def resize_images(input_directory, output_directory, target_size=(800, 800)):
    # 출력 폴더가 없으면 생성
    os.makedirs(output_directory, exist_ok=True)
    
    files = os.listdir(input_directory)
    
    for file in files:
        file_path = os.path.join(input_directory, file)
        output_path = os.path.join(output_directory, file)
        
        # 파일이 이미지인지 확인
        if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            try:
                with Image.open(file_path) as img:
                    # 이미지 크기 조정
                    img_resized = img.resize(target_size, Image.LANCZOS)
                    
                    # 새로운 경로에 저장
                    img_resized.save(output_path)
                    print(f"Resized and saved: {file}")
            except Exception as e:
                print(f"Error resizing {file}: {e}")

# 실행
input_directory = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/0_원본이미지"
output_directory = "/home/a/A_2024_selfcode/CLASS-PCB_proj_DETR/raw_datasets/1_2_800images"
resize_images(input_directory, output_directory)
