'''from examples.image_load.main_load import ImageLoadThreading


class StartStopImageLoad:

    def __init__(self):
        self.image_load_instance = ImageLoadThreading()

    def start_(self):
        self.image_load_instance.start_load_image()

    def stop(self):
        self.image_load_instance.stop_load_image()


if __name__ == '__main__':
    image_load_start_stop = StartStopImageLoad()
    try:
        image_load_start_stop.start_()
    except KeyboardInterrupt:
        image_load_start_stop.stop()

'''

# examples/image_load/main_start_stop.py

from examples.image_load.main_load import ImageLoadThreading

class StartStopImageLoad:
    def __init__(self):
        self.image_load_instance = ImageLoadThreading()

    def start_(self):
        self.image_load_instance.start_load_image()

    def stop(self):
        self.image_load_instance.stop_load_image()

# if __name__ == '__main__':
#     image_load_start_stop = StartStopImageLoad()
#     try:
#         image_load_start_stop.start_()
#     except KeyboardInterrupt:
#         image_load_start_stop.stop()
