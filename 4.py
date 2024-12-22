import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')

VIDEO_PATH = "./yourvideo path.mp4"
RESULT_PATH = "results.mp4"
if __name__ == '__main__':
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file")
        exit()

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 转换为整数
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 转换为整数

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码
    videoWriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (frame_width, frame_height))

    while True:
        success, frame = capture.read()
        if not success:
            print("读取帧失败")
            break

        # 使用YOLO进行跟踪
        results = model.track(frame, persist=True)

        # 获取带有检测框的帧
        a_frame = results[0].plot()  # 使用plot()方法

        # 显示处理过的帧
        cv2.imshow('yolo track', a_frame)

        # 写入视频文件
        videoWriter.write(a_frame)

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()
