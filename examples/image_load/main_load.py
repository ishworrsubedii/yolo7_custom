import time

import cv2 as cv
import os
import threading

from examples.detection_example_cam import CamDetection

class ImageLoadThreading:
    def __init__(self):
        self.flag=False
        self.thread_running = False
        self.image_load = None
        self.image_path_img = '/home/ishwor/.streamsave/'
        self.latest_image_path = None
        self.cam_det=CamDetection()

    def start_load_image(self):
        self.image_load = threading.Thread(target=self.image_list)
        self.image_load.start()
        self.thread_running = True

    def stop_load_image(self):
        if self.thread_running:
            print("Stopping the image loading thread...")
            self.thread_running = False
            self.image_load.join()
            print("Image loading thread stopped.")

    def image_list(self):
        time.sleep(10)
        while self.thread_running:
            files = sorted(os.listdir(self.image_path_img))
            for filename in files:
                # if filename not in self.checked_files:
                image_path = os.path.join(self.image_path_img, filename)

                self.latest_image_path = image_path
                print("Processing:", self.latest_image_path)

                rd = cv.imread(self.latest_image_path)
                time.sleep(0.5)
                if rd is not None:
                    self.cam_det.cam_image_detection(self.latest_image_path,draw=True)
                    cv.imshow('Image', rd)
                    cv.waitKey(3)
                    print('done')
                    os.remove(self.latest_image_path)

                else:
                    print('Image Loading Failed .......')

        cv.destroyAllWindows()

# if __name__ == '__main__':
#     image_load_start_stop = ImageLoadThreading()
#     image_load_start_stop.start_load_image()
#
#     time.sleep(5)
#     image_load_start_stop.stop_load_image()



