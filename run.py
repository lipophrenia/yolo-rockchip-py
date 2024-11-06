import os

import sys
import importlib.util
opencv_gst_path = "/usr/local/lib/python3.10/dist-packages/cv2/python-3.10/cv2.cpython-310-aarch64-linux-gnu.so"
spec = importlib.util.spec_from_file_location("cv2", opencv_gst_path)
cv2 = importlib.util.module_from_spec(spec)
sys.modules["cv2"] = cv2
spec.loader.exec_module(cv2)
print("OpenCV version:", cv2.__version__)

from rknnlite.api import RKNNLite

import functions

img_path = "/home/firefly/test.jpg"
video_path = "/home/firefly/video.mp4"
# video_path = "v4l2src device=/dev/video0 ! image/jpeg, framerate=30/1, width=640, height=480 ! mppjpegdec ! videoconvert ! appsink"
video_inference = False
frame_counter = 0

if __name__ == '__main__':
    rknn_lite = RKNNLite()
    
    ret = rknn_lite.load_rknn(functions.RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)

    if not os.path.exists('./result'):
        os.makedirs('./result')

    if video_inference == True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("can't open")
            cap.release()
            exit()
        output = cv2.VideoWriter('./result/yolo_result.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, functions.IMG_SIZE)
        while(True):
            while(cap.isOpened()):
                status, img = cap.read()

                if not status:
                    print("Stream disconnected")
                    cap.release()
                    output.release()
                    break

                print('--> Running model for video inference')
                img = functions.run_inference(rknn_lite,img)
                
                output.write(img)
                frame_counter+=1
                print("frame number:")
                print(frame_counter)
            break
    else:
        img = cv2.imread(img_path)
        img = functions.run_inference(rknn_lite,img)
        cv2.imwrite('./result/yolo_result.jpg', img)
