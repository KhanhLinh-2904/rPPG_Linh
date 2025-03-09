import cv2
import time
def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    start_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            end_time = time.time()
            break  # Exit loop if no more frames
        if frame is None:
            print("Frame is None!")
        frame_count += 1
        # cv2.imshow("Frame", frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  # Press 'q' to exit early
    
    print(f"Total frames read: {frame_count}")
    print("time: ", end_time-start_time)
    cap.release()
    cv2.destroyAllWindows()

# Change the path to your video file
video_path = "dataset/01-01.avi"
read_video_frames(video_path)

