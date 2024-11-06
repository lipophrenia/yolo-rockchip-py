# required
# sudo apt-get install libglib2.0-dev libgstrtspserver-1.0-dev gstreamer1.0-rtsp libgirepository1.0-dev libcairo2-dev
# pip install PyGObject

import argparse

import sys
import importlib.util
opencv_gst_path = "/usr/local/lib/python3.10/dist-packages/cv2/python-3.10/cv2.cpython-310-aarch64-linux-gnu.so"
spec = importlib.util.spec_from_file_location("cv2", opencv_gst_path)
cv2 = importlib.util.module_from_spec(spec)
sys.modules["cv2"] = cv2
spec.loader.exec_module(cv2)
print("OpenCV version:", cv2.__version__)

import gi

import functions
from rknnpool import rknnPoolExecutor

# import required library like Gstreamer and GstreamerRtspServer
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject, GLib

# Sensor Factory class which inherits the GstRtspServer base class and add
# properties to it.
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.cap = cv2.VideoCapture(args.source)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.pool = rknnPoolExecutor(
            rknnModel=functions.RKNN_MODEL,
            threads=functions.THREADS,
            func=functions.run_inference)
        
        if (self.cap.isOpened()):
            for i in range(functions.THREADS):
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    del self.pool
                    exit(-1)
                self.pool.put(frame)

        # self.number_frames = 0
        self.duration = 1 / args.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                            'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                            '! videoconvert ! video/x-raw,format=NV12 ' \
                            '! mpph264enc ! rtph264pay name=pay0 pt=96' \
                            .format(args.frame_width, args.frame_height, args.fps)
    # method to capture the video feed from the camera and push it to the
    # streaming buffer.
    def on_need_data(self, src, length):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # It is better to change the resolution of the camera 
                # instead of changing the image shape as it affects the image quality.   
                # cv2.imwrite("test.jpg", frame)             
                self.pool.put(frame)
                frame, flag = self.pool.get()
                # frame = cv2.resize(frame, (1920,1080))

                data = frame.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                # self.number_frames += 1
                retval = src.emit('push-buffer', buf)
                # print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames, self.duration, self.duration / Gst.SECOND))
                if retval != Gst.FlowReturn.OK:
                    print(retval)
    # attach the launch string to the override method
    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)
    
    # attaching the source element to the rtsp media
    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

# Rtsp server implementation where we attach the factory sensor with the stream uri
class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.set_service(str(args.port))
        self.get_mount_points().add_factory(args.stream_uri, self.factory)
        self.attach(None)
        print(f"Stream is running on rtsp://<yourIP>:{args.port}{args.stream_uri}")

parser = argparse.ArgumentParser(
    description="""RTSP server with YOLO inference.""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-src","--source", default='/dev/video0', help="source, accepted formats are similar to openCV")
parser.add_argument("-fps","--fps", default=30, help="fps of the camera", type = int)
parser.add_argument("-width", "--frame_width", default=640, help="video frame width", type = int)
parser.add_argument("-height", "--frame_height", default=640, help="video frame height", type = int)
parser.add_argument("-port","--port", default=8554, help="port to stream video", type = int)
parser.add_argument("-uri", "--stream_uri", default = "/camera", help="rtsp video stream uri")
args = parser.parse_args()

print("Starting...")

Gst.init(None)
server = GstServer()
loop = GLib.MainLoop()
loop.run()