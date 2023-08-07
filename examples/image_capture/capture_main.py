
import cv2 as cv
import os
import imagehash
from PIL import Image
from examples.image_capture.utils.settings import get_frame_save_dir
from datetime import datetime
import threading
import time



class FrameSaver:



    def __init__(self, source, image_path_to_save=None):

        self.source = source
        self.image_path_to_save = image_path_to_save
        if self.image_path_to_save is None or not os.path.exists(self.image_path_to_save):
            self.image_path_to_save = get_frame_save_dir()
        self.thread_running = False
        self.frame_save_thread = None
        self.cap = cv.VideoCapture(self.source)




    def start_stream(self):
        self.frame_save_thread = threading.Thread(target=self.video_webcam_frame_extraction)
        self.frame_save_thread.start()
        self.thread_running = True



    def stop_stream(self):
        if self.thread_running:
            self.frame_save_thread.join()
            self.thread_running = False
        self.cap.release()


    def hashing_dif(self, prev_frame, current_frame):
        if prev_frame is None:
            return None
        else:
            hash1 = imagehash.average_hash(Image.fromarray(prev_frame))
            hash2 = imagehash.average_hash(Image.fromarray(current_frame))
            return hash2 - hash1

    def video_webcam_frame_extraction(self):
        previous_frame = None
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                hash_diff = self.hashing_dif(previous_frame, frame)
                threshold = 5
                if hash_diff is None or hash_diff > threshold:
                    if not isinstance(self.source, str):
                        timestamp_ms = self.cap.get(cv.CAP_PROP_POS_MSEC)
                        timestamp_sec = (timestamp_ms / 1000.0)
                        timestamp_min = int(timestamp_sec // 60)
                        timestamp_sec %= 60.
                        time_str = f"{timestamp_sec:.2f}"
                        time = f'{timestamp_min}.{(time_str)}'
                        name = os.path.join(self.image_path_to_save, f"{time}.jpg")
                        cv.imwrite(name, frame)

                    else:
                        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        name = os.path.join(self.image_path_to_save, f"{time}.jpg")
                        cv.imwrite(name, frame)
                        # yield frame, time
                    previous_frame = frame
                    print(f"Frame {time} saving...")

        print("Frame saving thread stopped.")
        exit()



if __name__ == '__main__':
    source = 'rtsp://192.168.1.250:8080/h264_opus.sdp'
    start_stop = FrameSaver(source)
    start_stop.start_stream()
    time.sleep(5)
    start_stop.stop_stream()
