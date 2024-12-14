import numpy as np
import os
import cv2
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqxcb.so'
import dlib
from imutils import face_utils

class FaceDetection(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.tracker1 = cv2.TrackerMIL_create()
         
    def face_detect(self, frame):
        if frame is None:
            print("No frame to do face detection")
            return None
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.copy()
        
        rects = self.detector(gray, 1)
        if len(rects) > 0:
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            bbox = (x, y, w, h)
            face_frame =frame[bbox[1]:bbox[1]+int(bbox[3]), bbox[0]:bbox[0]+bbox[2]].copy()
            self.tracker1.init(frame, bbox)
        else:
            print("failed detect face")
            return None
        face_frame = cv2.resize(face_frame, (36,36), dst=None, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        return face_frame
 

    def face_track(self, frame):
        if frame is None:
            print("No frame to do face tracking")
            return None

        ok, bbox = self.tracker1.update(frame)
        if ok:
            face_frame = frame[bbox[1]:bbox[1]+int(bbox[3]), bbox[0]:bbox[0]+bbox[2]]
            rect = dlib.rectangle(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
            rects = dlib.rectangles()
            rects.append(rect)
        else:
            print("Update Tracker failure")
            return None
        face_frame = cv2.resize(face_frame, (36,36), dst=None, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

        return face_frame

   

  