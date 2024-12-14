import pyrealsense2 as rs
import numpy as np
import time
import cv2

class Camera_RGB(object):
    def __init__(self):
        self.camera_stream = 0
        self.cap = None
        t0 = 0

    def start(self):
        print("Start camera " + str(self.camera_stream))

        self.cap = cv2.VideoCapture(self.camera_stream)
        if not self.cap.isOpened():
            print("Cannot open camera")
            return
        self.t0 = time.time()
        self.valid = False #Check can open camera or not
        try:
            ret, resp = self.cap.read()
            if not ret:
                print("Cannot receive frame (stream end?). Exiting ...")
                self.valid = False
            self.valid = True
        except:
            print("Cannot receive frame! ")
            self.valid = False

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            print("Camera Stopped")

    def get_frame(self):
        if self.valid:
            _, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame is None or frame.size == 0:
                # raise ValueError("Invalid frame") 
                print("End of Camera")
                self.stop()
                print(time.time() - self.t0)
                return
            else:
                frame = cv2.resize(frame, (640, 480))
        return frame

class Webcam(object):
    def __init__(self):
        self.points = rs.points()
        self.pipeline = rs.pipeline()
        self.dirname = ""  # for nothing, just to make 2 inputs the same
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)  # todo: avoid 20 fps, cause the app stop
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

    def start(self):
        print("Starting the webcam")
        profile = self.pipeline.start(self.config)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        print("IR Emitter Status: " + str(depth_sensor.get_option(rs.option.emitter_enabled)))

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        nir_frame = frames.get_infrared_frame()
        color_out = np.asanyarray(color_frame.get_data())
        nir_out = np.asanyarray(nir_frame.get_data(), dtype='uint8')
        '''
        else:
            color_out = np.ones((480, 640, 3), dtype=np.uint8)
            nir_out = np.ones((480, 640, 1), dtype=np.uint8)
            col = (0, 256, 256)
            cv2.putText(color_out, "(Error: Camera not accessible)",
                        (65, 220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        '''
    
        return color_out, nir_out

    def stop(self):
        self.pipeline.stop()
        print("Stopped the webcam")