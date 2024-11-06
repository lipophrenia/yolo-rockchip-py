# YOLOv8 & YOLOv10 MODEL ROCKCHIP PYTHON NPU INFERENCE

Dependencies for inference on Rockchip NPU for the required version of `python` are taken [here](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages).
We also need a runtime lib - [click](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Linux/librknn_api). `librknnrt.so` put in `/usr/lib/`.

```bash
python3 -m venv env
source ./env/bin/activate
pip install rknn_toolkit_lite2-2.2.0-cp310-cp310-linux_aarch64.whl
pip install -r requirements.txt
```

The settings related to the model are in the `functions.py` file. They are common to all scripts.
```python
RKNN_MODEL = './models/yolov8n_3588_i8.rknn' # relative path to rknn model
THREADS = 9 # number of threads for multi-threaded inference
OBJ_THRESH, NMS_THRESH = 0.35, 0.45 # confidence and intersection over union thresholds
IMG_SIZE = (640, 640) # input model resolution (width, height)
CLASSES = ("","","") # list of classes existing in the model
```

### run.py - single-core inference without GUI, the result is saved in ./result/

```python
# change in run.py script
img_path = "/home/firefly/test.jpg" # path to image for inference
video_path = "/home/firefly/video.mp4" # path to video for inference
video_inference = True # video/image switcher
```

For video, the source fps needs to match the fps in `VideoWriter`:
```python
# change in run.py script
output = cv2.VideoWriter('./result/yolooutput.avi', cv2.VideoWriter_fourcc('M','J','P','G'), FPS, functions.IMG_SIZE)
```

### run_multicore.py - multi-core inference with GUI for camera/rtsp.

```python
# change in run_multicore.py script
cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg, framerate=30/1, width=640, height=480 ! mppjpegdec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
```

### rtsp_yolo_mt_server.py - multi-core inference + rtsp-server

```bash
python3 rknn_rtsp_yolo_mt_server.py -src <source> -fps <fps> -w <width> -h <height> -p <port> -uri <uri>
# src,source - video source supported by OpenCV.
# fps - frame rate.
# width - source resolution by width.
# height - source resolution by height.
# port - port for rtsp server.
# uri - stream uri, specify starting with "/".
# -h for help
```

### rkcat.sh

Allows you to monitor NPU load in real time.

### OpenCV
I used OpenCV compiled manually from sources to have GStreamer API support. If you don't need it, you can install `opencv-python` via `pip` and replace all occurrences of the following type:
```python
import sys
import importlib.util
opencv_gst_path = "/usr/local/lib/python3.10/dist-packages/cv2/python-3.10/cv2.cpython-310-aarch64-linux-gnu.so"
spec = importlib.util.spec_from_file_location("cv2", opencv_gst_path)
cv2 = importlib.util.module_from_spec(spec)
sys.modules["cv2"] = cv2
```
with
```python
import cv2
```
