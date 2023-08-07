import time


from examples.image_capture.utils.main_start_stop import Startstop

def stream_stopp():

    start_stop=Startstop()
    start_stop.stop()

    # time.sleep(24)

if __name__ == '__main__':
    stream_stopp()