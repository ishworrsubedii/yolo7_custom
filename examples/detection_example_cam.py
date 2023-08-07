import cv2 as cv
import os
from detector.detector import YoloV7Detector


class CamDetection:

    def __init__(self):
        self.model_path = '/home/ishwor/Desktop/Yolov7_custom/yolov7/best.pt'
        self.detected_images_dir = '/home/ishwor/Desktop/Yolov7_custom/yolov7/detected_details'
        self.file_path = '/home/ishwor/Desktop/Yolov7_custom/yolov7/detected_details/detection_activities.txt'

    def draw_bbox(self, bbox, image):
        dh, dw, _ = image.shape
        for box in bbox:
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv.rectangle(image, (x, y), (w, h), (0, 0, 255), 1)
        return image

    def cam_image_detection(self, image_path, draw=True):
        image_display = cv.imread(image_path)
        yolov7_detector = YoloV7Detector(self.model_path)
        model_prediction, conf, classes = yolov7_detector.detect(image_display)

        filename = os.path.basename(image_path)
        detection_time = os.path.splitext(filename)[0]
        count = sum(1 for i in classes if i == 0)
        if count > 0:
            if draw:
                image__ = self.draw_bbox(model_prediction, image_display)
                resized = cv.resize(image__, dsize=(1024, 720))
                filename = os.path.basename(image_path)
                filename = os.path.join(self.detected_images_dir, filename)
                cv.imwrite(filename, resized)

            with open(self.file_path, 'a') as file:
                file.write(f'{detection_time}, detected: {count}\n')
            print(f'Frame_time: {detection_time}, Detected: {count}')

        else:
            print(f'Frame_time: {detection_time}, Detected: {count}')


if __name__ == '__main__':
    image_path = '/home/ishwor/Desktop/Yolov7_custom/yolov7/detected_details/monkey1.jpg'
    cam_det = CamDetection()
    cam_det.cam_image_detection(image_path, draw=True)
