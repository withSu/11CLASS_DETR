import os
import torch
import cv2

def get_coco_size_label(w, h):
    """COCO 공식 기준에 따라 객체 크기를 분류 (면적 기반)"""
    area = w * h  # 면적 계산
    if area < 32**2:  # 1024 픽셀² 미만
        return "Small"
    elif 32**2 <= area < 96**2:  # 1024 픽셀² 이상, 9216 픽셀² 미만
        return "Medium"
    else:  # 9216 픽셀² 이상
        return "Large"

def inference_and_visualize(model, postprocessors, data_loader, device, output_dir):
    """
    (1) data_loader 순회하며 모델 추론
    (2) 결과를 이미지에 시각화
    (3) output_dir/vis 에 저장
    """
    model.eval()
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)  # 디렉토리 생성 확인
    print(f"[INFO] Visualization directory: {vis_dir}")

    print("Starting final inference & visualization on the entire dataset...")

    for samples, targets in data_loader:
        samples = samples.to(device)
        with torch.no_grad():
            outputs = model(samples)

        # postprocess
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        for i, result in enumerate(results):
            image_id = targets[i]["image_id"].item()

            # COCO 정보에서 파일명 얻기 (dataset.coco.loadImgs)
            img_info = data_loader.dataset.coco.loadImgs([image_id])[0]
            file_name = img_info["file_name"]  # 예: "image1.jpg"
            image_path = os.path.join("./datasets/val_images", file_name)

            print(f"[INFO] Processing image: {image_path}")

            visualize_single_image(
                image_path,
                result,
                vis_dir,
                threshold=0.5
            )

def visualize_single_image(image_path, result, vis_dir, threshold=0.5):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to open image: {image_path}")
        return

    # 이미지 크기 계산
    img_height, img_width = img.shape[:2]
    img_size_text = f"Image Size: {img_width}x{img_height} (pixels)"

    boxes = result["boxes"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    labels = result["labels"].cpu().numpy()

    # 폰트 크기와 두께 조정
    font_scale = 0.3
    text_thickness = 1

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)
        w, h = x2 - x1, y2 - y1
        area = w * h
        size_label = get_coco_size_label(w, h)

        # 크기(Small/Medium/Large)에 따라 색상 결정
        if size_label == "Small":
            color = (255, 0, 0)   # 파란색
        elif size_label == "Medium":
            color = (0, 255, 0)   # 초록색
        else:  # Large
            color = (0, 0, 255)   # 빨간색

        thickness = 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 출력 텍스트 작성
        text = f"{label}:{score:.2f}, {w}x{h}, {size_label}, Area: {area}"

        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)

        # 테두리 먼저 그린 후, 글씨 표시
        cv2.putText(img, text, (x1, max(y1 - 15, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, text_thickness + 2)
        cv2.putText(img, text, (x1, max(y1 - 15, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

    # 이미지 크기를 왼쪽 상단에 표시
    text_position = (10, 60)
    outline_color = (0, 0, 0)
    text_color = (255, 255, 255)
    cv2.putText(img, img_size_text, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, text_thickness + 2)
    cv2.putText(img, img_size_text, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

    # 결과 저장
    base_name = os.path.basename(image_path)
    out_name = base_name.replace(".jpg", "_res.jpg")
    save_path = os.path.join(vis_dir, out_name)

    os.makedirs(vis_dir, exist_ok=True)
    print(f"[INFO] Saving visualization to: {save_path}")

    success = cv2.imwrite(save_path, img)
    if not success:
        print(f"[ERROR] Failed to save: {save_path}")