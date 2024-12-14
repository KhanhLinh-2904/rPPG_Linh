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

