import torch
import easyocr
import cv2

# YOLOv5 모델 및 가중치 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/jihoon/automatic-number-plate-recognition/best.pt')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/jihoon/automatic-number-plate-recognition/best_test.pt')
model.eval()

# EasyOCR 설정
reader = easyocr.Reader(['ko'])  # or ['ko'] for Korean

# 동영상 파일 열기
cap = cv2.VideoCapture('C:/jihoon/carnumber/output3.mp4')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # 텍스트 출력값 설정
    confidence_threshold = 0.7

# 객체 탐지 수행
    results = model(img)
    preds = results.pred[0]

    for pred in preds:
        label, x1, y1, x2, y2, score = pred[-1].item(), int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3]), pred[4]

        if score >= confidence_threshold:
            # ROI(Region Of Interest)를 이미지에서 잘라냄
            roi = img[y1:y2, x1:x2]

            # 잘라낸 부분에서 텍스트 인식
            ocr_results = reader.readtext(roi)

            for (bbox, text, prob) in ocr_results:
                print(f"Detected Text: {text}, Confidence: {prob:.4f}")

    # 프레임 보여주기 (optional)
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()