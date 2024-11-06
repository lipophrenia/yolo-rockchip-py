import sys
import importlib.util
opencv_gst_path = "/usr/local/lib/python3.10/dist-packages/cv2/python-3.10/cv2.cpython-310-aarch64-linux-gnu.so"
spec = importlib.util.spec_from_file_location("cv2", opencv_gst_path)
cv2 = importlib.util.module_from_spec(spec)
sys.modules["cv2"] = cv2
spec.loader.exec_module(cv2)
print("OpenCV version:", cv2.__version__)

import time

from rknnpool import rknnPoolExecutor
import functions

cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg, framerate=30/1, width=640, height=480 ! mppjpegdec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture("filesrc location=/home/firefly/video.mp4 ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 640, 640)

# Initialize rknn pool
pool = rknnPoolExecutor(
    rknnModel=functions.RKNN_MODEL,
    threads=functions.THREADS,
    func=functions.run_inference)

# Initialize the frame required for asynchronous
if (cap.isOpened()):
    for i in range(functions.THREADS + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

# exit(-1)

frames, initTime = 0, time.time()
while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break
    pool.put(frame)
    frame, flag = pool.get()
    if flag == False:
        break
    cv2.imshow('Camera', frame)
    # break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print("AVG FPS:\t", frames / (time.time() - initTime))

print("Overall average frame rate\t", frames / (time.time() - initTime))

cap.release()
cv2.destroyAllWindows()
pool.release()
