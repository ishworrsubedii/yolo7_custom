import time

from examples.image_capture.utils.main_start_stop import Startstop


def stream_startt():
    start_stop = Startstop()
    start_stop.start()
    # time.sleep(1)
    # start_stop.stop()


if __name__ == '__main__':
    stream_startt()


