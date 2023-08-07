from detector.detector import YoloV7Detector
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
# import torchvision.transforms as T

import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg'

class SalencyMap:
    def __init__(self,model_path):
        self.model_path=model_path
    def preprocess_img(image, channels=3):
        image = Image.open(image_path)

        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x[None]),
        ])
        return transform(image)


    # def postprocess_img(pred, org_dir):
    #     # def deprocess(image):
    #     #     transform = transforms.Compose([
    #     #         transforms.Lambda(lambda x: x[0]),
    #     #         transforms.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
    #     #         transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    #     #         transforms.ToPILImage(),
    #     #     ])
    #     #     return transform(image)
    #     pred = np.array(pred)
    #     org = cv2.imread(org_dir, 0)
    #     shape_r = org.shape[0]
    #     shape_c = org.shape[1]
    #     predictions_shape = pred.shape
    #
    #     rows_rate = shape_r / predictions_shape[0]
    #     cols_rate = shape_c / predictions_shape[1]
    #
    #     if rows_rate > cols_rate:
    #         new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
    #         pred = cv2.resize(pred, (new_cols, shape_r))
    #         img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    #     else:
    #         new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
    #         pred = cv2.resize(pred, (shape_c, new_rows))
    #         img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]
    #
    #     return img

    def main_implementation_saliency(self,test_img):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(self.model_path, map_location='cuda')['model']
        model = model.to(torch.float32)

        # img = SalencyMap.preprocess_img(image_path) .to(device) # padding and resizing input image into 384x288
        # # img = np.array(img) / 255.
        # # img = np.transpose(img, (2, 0, 1))
        # # img = torch.from_numpy(img).unsqueeze(0)
        # # img = img.to(device)
        # # img = img.float()
        #
        # pred_saliency = model(img)
        #
        # toPIL = transforms.ToPILImage()
        # pic = toPIL(pred_saliency[0].squeeze())
        #
        #
        #
        #
        #
        #
        # actual_image = cv2.imread(image_path)
        #
        # pred_saliency = SalencyMap.postprocess_img(pic,
        #                                            test_img)  # restore the image to its original size as the result
        X = SalencyMap.preprocess_img(test_img).to(device)

        model.eval()

        X.requires_grad_()

        scores = model(X)

        score_max_index = scores.argmax()

        score_max = scores[0, score_max_index]

        score_max.backward()
        saliency, _ = torch.max(X.grad.data.abs(), dim=1)


        # plt.subplot(1, 2, 1)
        # plt.imshow(cv2.cvtColor(actual_image, cv2.COLOR_BGR2RGB))
        # plt.title('Actual Image')
        # plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(saliency[0], alpha=0.5)#alpha=0.5
        plt.title('Saliency Map')
        plt.axis('off')

        plt.show()
image_path='/home/ishwor/Desktop/TreeLeaf/yolov7/visualization experiment/images/monkey1.jpg'
model_path= '/home/ishwor/Downloads/3k_monkey_pretrained_.pt'

salency_map=SalencyMap(model_path)
salency_map.main_implementation_saliency(image_path)
