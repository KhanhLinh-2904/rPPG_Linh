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
            # face_frame =frame[bbox[1]:bbox[1]+int(bbox[3]), bbox[0]:bbox[0]+bbox[2]].copy()
            face_frame = frame[max(y-int(0.5*h), 0):min(y+int(h*1), frame.shape[0]), 
                   x:min(x+int(w*1), frame.shape[1])].copy()
          
            # self.tracker1.init(frame, bbox)
            # face_frame = cv2.resize(face_frame, (36,36), dst=None, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            return face_frame
        else:
            print("failed detect face")
            return None
        
 
    def face_landmark_nose(self, frame):
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
            landmarks = self.landmark_predictor(gray, rects[0])
            # Xác định vị trí mũi (landmark 30)
            nose_x = int((landmarks.part(28).x + landmarks.part(29).x + landmarks.part(30).x + landmarks.part(31).x + 
                          landmarks.part(32).x + landmarks.part(33).x + landmarks.part(34).x + landmarks.part(35).x + landmarks.part(36).x) / 9)
            nose_y = int((landmarks.part(28).y + landmarks.part(29).y + landmarks.part(30).y + landmarks.part(31).y + 
                          landmarks.part(32).y + landmarks.part(33).y + landmarks.part(34).y + landmarks.part(35).y + landmarks.part(36).y) / 9)
            # nose_x, nose_y = self.stabilize_landmark( nose_x, nose_y)
            # Điều chỉnh vị trí ROI sao cho mũi luôn ở trung tâm
            roi_x = nose_x - w // 2
            roi_y = nose_y - h // 2
            # Đảm bảo ROI không vượt ra ngoài khung hình
            roi_x = max(0, min(roi_x, frame.shape[1] - w))
            roi_y = max(0, min(roi_y, frame.shape[0] - h))

            # Cắt ảnh theo ROI ổn định
            stable_face_frame = frame[roi_y:roi_y + w, roi_x:roi_x +  h]

            # # Resize ROI về 36x36
            # stable_face_frame_resized = cv2.resize(stable_face_frame, (36, 36), interpolation=cv2.INTER_CUBIC)
            return stable_face_frame
          
        else:
            print("failed detect face")
            return None
    
     
        
    def face_landmark_center(self, frame):
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
            landmarks = self.landmark_predictor(gray, rects[0])
           # Extract eyebrow and chin landmarks
            eyebrow_indices = list(range(17, 27))  # Both left and right eyebrows
            chin_indices = list(range(0, 17))  # Chin contour

            eyebrow_points = get_landmark_points(landmarks, eyebrow_indices)
            chin_points = get_landmark_points(landmarks, chin_indices)

            # Compute bounding box
            center_x, center_y, width, height, x_min, y_min, x_max, y_max = compute_bounding_box(eyebrow_points, chin_points)
            # Điều chỉnh vị trí ROI sao cho mũi luôn ở trung tâm
            width = int(width*1.4)
            height =int(height*1.4)
            roi_x = center_x - width // 2
            roi_y = center_y - height // 2
            # Đảm bảo ROI không vượt ra ngoài khung hình
            roi_x = max(0, min(roi_x, frame.shape[1] - width))
            roi_y = max(0, min(roi_y, frame.shape[0] - height))

            # Cắt ảnh theo ROI ổn định
            stable_face_frame = frame[roi_y:roi_y + width, roi_x:roi_x +  height]

            # # Resize ROI về 36x36
            # stable_face_frame_resized = cv2.resize(stable_face_frame, (36, 36), interpolation=cv2.INTER_CUBIC)
            return stable_face_frame
          
        else:
            print("failed detect face")
            return None
        
            
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
        # face_frame = cv2.resize(face_frame, (36,36), dst=None, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

        return face_frame

   
def get_landmark_points(landmarks, indices):
    """ Extract points based on landmark indices """
    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in indices])

def compute_bounding_box(eyebrows, chin):
    """ Compute the bounding box based on eyebrow and chin points """
    x_min = min(eyebrows[:, 0])
    x_max = max(eyebrows[:, 0])
    y_min = min(eyebrows[:, 1])
    y_max = max(chin[:, 1])

    # Compute center, width, height
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    width = x_max - x_min
    height = y_max - y_min

    return (center_x, center_y, width, height, x_min, y_min, x_max, y_max) 
