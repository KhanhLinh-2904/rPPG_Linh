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
        self.prev_roi_center = None
        self.smoothing_factor = 0.8  # Adjust between 0.7 - 0.9 for optimal smoothing
        self.prev_height_width = None
    
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
    
     
    def get_landmark_points(self, landmarks, indices):
        return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in indices])

    def compute_bounding_box(self, eyebrows, chin, nose):
        """ Compute the bounding box based on eyebrow, nose, and chin points """
        x_min = min(eyebrows[:, 0])
        x_max = max(eyebrows[:, 0])
        y_min = min(eyebrows[:, 1])
        y_max = max(chin[:, 1])
        
        # Compute center using nose as reference
        nose_center = np.mean(nose, axis=0).astype(int)
        center_x, center_y = nose_center
        # Adjust width and height to include eyebrows and chin
        width = x_max - x_min
        height = y_max - y_min
         # Apply exponential moving average for stability
        if self.prev_roi_center is None:
            self.prev_roi_center = (center_x, center_y)
            self.prev_height_width = (width, height)         
        else:
            alpha = self.smoothing_factor  # Smoothing factor
            center_x = int(alpha * self.prev_roi_center[0] + (1 - alpha) * center_x)
            center_y = int(alpha * self.prev_roi_center[1] + (1 - alpha) * center_y)
            self.prev_roi_center = (center_x, center_y)
            width = int(alpha * self.prev_height_width[0] + (1 - alpha) * width)
            height = int(alpha * self.prev_height_width[1] + (1 - alpha) * height)
            self.prev_height_width = (width, height)
            
        
       
        
        return center_x, center_y, int(width * 1.4), int(height * 1.4)

    def face_landmark_center(self, frame):
        if frame is None:
            print("No frame to do face detection")
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame.copy()
        rects = self.detector(gray, 1)
        
        if len(rects) > 0:
            landmarks = self.landmark_predictor(gray, rects[0])
            
            # Extract eyebrow, chin, and nose landmarks
            eyebrow_indices = list(range(18, 28))  # Both eyebrows
            # print("eyebrow_indices: ", eyebrow_indices)
            chin_indices = list(range(1, 18))  # Chin contour
            nose_indices = list(range(28, 30))  # Nose region
            
            eyebrow_points = self.get_landmark_points(landmarks, eyebrow_indices)
            chin_points = self.get_landmark_points(landmarks, chin_indices)
            nose_points = self.get_landmark_points(landmarks, nose_indices)
            
            # Compute bounding box using eyebrows, chin, and nose
            center_x, center_y, width, height = self.compute_bounding_box(eyebrow_points, chin_points, nose_points)
            
            # Define ROI ensuring it stays within frame
            roi_x = max(0, min(center_x - width // 2, frame.shape[1] - width))
            roi_y = max(0, min(center_y - height // 2, frame.shape[0] - height))
            
            # Extract stable face region
            stable_face_frame = frame[roi_y:roi_y + height, roi_x:roi_x + width]
            
            return stable_face_frame
        else:
            print("Failed to detect face")
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

   
