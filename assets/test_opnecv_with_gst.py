import os
os.add_dll_directory(r'C:\opencv\build\install\x64\vc16\bin')
os.add_dll_directory(r'C:\gstreamer\1.0\msvc_x86_64\bin')

import cv2
print("OpenCV:", cv2.__version__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())

pipeline = (
    "filesrc location=smaple_test_video.mp4 ! "
    "decodebin ! "
    "videoconvert ! "
    "appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ Failed to open video")
    exit()

WIDTH, HEIGHT = 640, 360
gpu_frame = cv2.cuda_GpuMat()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Upload to GPU
    gpu_frame.upload(frame)

    # GPU resize
    gpu_small = cv2.cuda.resize(gpu_frame, (WIDTH, HEIGHT))

    # Download once
    frame_small = gpu_small.download()

    cv2.imshow("GStreamer + CUDA", frame_small)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
