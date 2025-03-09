import cv2


class VideoFrameExtractor:
    def __init__(self, video_url):
        self.video_url = video_url
        self.cap = None
        self.valid = False
        self.count = 0
    def start(self):
       self.cap = cv2.VideoCapture(self.video_url)
       if not self.cap.isOpened():
            print("Error: Could not open video file.")
            self.valid = False
       else:
           self.valid = True
        
    
    def get_frame(self):
        if self.valid:
            ret, frame = self.cap.read()
            if not ret:
                print("Frame is None!")
                self.valid = False
                self.stop()
                return None
            else:
                frame = cv2.resize(frame, (640, 480))
                self.count += 1
                return frame
    def stop(self):
        if self.cap is not None:        
            self.cap.release()
            print("Camera Stopped")
            
    def print_count(self):
        print("Self count: ", self.count)

# if __name__ == "__main__":
#     # Change the path to your video file
#     video_path = "dataset/01-01.avi"
#     video_frames = VideoFrameExtractor(video_path)
#     video_frames.start()
#     while True:
#         frame = video_frames.get_frame()
#         if frame is None:
#             break
#         cv2.imshow("Frame", frame)
#     video_frames.print_count()
        