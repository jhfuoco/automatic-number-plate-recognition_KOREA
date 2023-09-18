from cv2 import waitKey
import torch
import cv2
import numpy as np
import time


# Model
model_path = r"C:/jihoon/automatic-number-plate-recognition/best.pt"  #custom model path
# model_path = r"C:/jihoon/automatic-number-plate-recognition/best_test.pt"  #custom model path
# model_path = r"C:/jihoon/automatic-number-plate-recognition/best_test_pure.pt"  #custom model path
video_path = r"C:/jihoon/automatic-number-plate-recognition/sorento_video.mp4"  #input video path
# video_path = r"C:/jihoon/carnumber_dataset/Video/보행상장애검출/VIGI C320I 1.0_20230908170812_4923312.mp4"  #input video path
cpu_or_cuda = "cuda"  #choose device; "cpu" or "cuda"(if cuda is available)
device = torch.device(cpu_or_cuda)
model = torch.hub.load('ultralytics/yolov5', 'custom', path= model_path, force_reload=True)
model = model.to(device)
frame = cv2.VideoCapture(video_path)

frame_width = int(frame.get(3))
frame_height = int(frame.get(4))
size = (frame_width, frame_height)
writer = cv2.VideoWriter('output3.mp4',-1,8,size)

text_font = cv2.FONT_HERSHEY_PLAIN
color= (0,0,255)
text_font_scale = 1.25
prev_frame_time = 0
new_frame_time = 0

# Inference Loop
while True:
    ret, image = frame.read()
    if ret:
        output = model(image)
        result = np.array(output.pandas().xyxy[0])
        for i in result:
            p1 = (int(i[0]), int(i[1]))
            p2 = (int(i[2]), int(i[3]))
            text_origin = (int(i[0]), int(i[1]) - 5)
            cv2.rectangle(image, p1, p2, color=color, thickness=2)  #drawing bounding boxes
            cv2.putText(image, text=f"{i[-1]} {i[-3]:.2f}", org=text_origin,
                        fontFace=text_font, fontScale=text_font_scale,
                        color=color, thickness=2)  #class and confidence text

        new_frame_time = time.time()

        difference = new_frame_time - prev_frame_time
        if difference != 0:
            fps = 1 / difference
            fps = int(fps)
            fps = str(fps)
        else:
            fps = "N/A"

        prev_frame_time = new_frame_time
        cv2.putText(image, fps, (7, 70), text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        writer.write(image)
        cv2.imshow("image", image)
    else:
        break

    if waitKey(1) & 0xFF == ord('q'):
        break

frame.release()
cv2.destroyAllWindows()