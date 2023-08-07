# from detector.yolov7_detector import YOLOV7_Detector
import torch
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, utils, models
import torchvision.transforms as T
from PIL import Image



class SalencyMap:
    def __init__(self, model_path):
        self.model_path = model_path
    def preprocess(image_path, size=640):


        image = Image.open(image_path)
        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x[None]),
        ])
        return transform(image)


    def postprocess_img(image_path):
        transform = T.Compose([
            T.Lambda(lambda x: x[0]),
            T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            T.ToPILImage(),
        ])
        return transform(image_path)



    def main_implementation_saliency(self, image_path):


        device = torch.device("cuda")




        model = torch.load(self.model_path,map_location='cuda')['model']
        model=model.to(torch.float32)
        model.eval()


        img = SalencyMap.preprocess(image_path).to(device)  # padding and resizing input image into 384x288


        pred_saliency = model(img)
        toPIL = transforms.ToPILImage()
        pic = toPIL(pred_saliency[0].squeeze().cpu())


        actual_image = cv2.imread(image_path)


        pred_saliency = SalencyMap.postprocess_img(pic) # restore the image to its original size as the result


        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(actual_image, cv2.COLOR_BGR2RGB))
        plt.title('Actual Image')
        plt.axis('off')


        plt.subplot(1, 2, 2)
        plt.imshow(pred_saliency, alpha=0.5)  # alpha=0.5
        plt.title('Saliency Map')
        plt.axis('off')


        plt.show()




image_path = '/home/ishwor/Desktop/detection/archive/3k/train/images/Monkey_images_from_monkey 273.jpg'
model_path = '/home/ishwor/Downloads/3k_monkey_pretrained_.pt'


salency_map = SalencyMap(model_path)
salency_map.main_implementation_saliency(image_path)