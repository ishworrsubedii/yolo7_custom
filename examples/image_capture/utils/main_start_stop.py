import threading
import time

from examples.image_capture.capture_main import FrameSaver


class Startstop:

    def __init__(self, source='rtsp://192.168.1.250:8080/h264_opus.sdp'):
        self.source = source
        self.frame_saver_instance = FrameSaver(source=source, image_path_to_save=None)

    def start(self):
        self.frame_saver_instance.start_stream()

    def stop(self):
        self.frame_saver_instance.stop_stream()

# if __name__ == '__main__':
#     start_stop = Startstop()
#     start_stop.start()
# time.sleep(20)
# start_stop.stop()


