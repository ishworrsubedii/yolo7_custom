import os
import time

import cv2 as cv
from PIL import Image
class DataAugmentation:
    def __init__(self):
        self.input_image_path = '/home/ishwor/Downloads/attendance video/images/images/'
        self.input_image_label = ''
        self.output_image_path = ''

    def image_input(self,):
        image_path_input = os.path.join(self.input_image_path)

        for i in os.listdir(image_path_input):
            image_full_path=self.input_image_path+i
            image_read=cv.imread(image_full_path)
            # image_read=Image.open(image_full_path)
            # image_read.rotate(45).show()
            rotated=cv.rotate(image_read,cv.ROTATE_90_CLOCKWISE)
            cv.imshow(f'{image_full_path}',rotated)
            cv.waitKey(0)





    def output_image(self):
        # Implement data augmentation logic here
        pass

    def saving_images(self):
        # Implement saving the augmented images here
        pass

    def processing(self):
        # Implement image processing logic here
        pass

da = DataAugmentation()
da.image_input()
