from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import numpy as np
# import pytesseract
import cv2
import os

input_path = "API/config/api/backend/originalImages/"
dirs = os.listdir(input_path)
print(dirs)

def imageProcessing():
    """ Do image processing/convert image and save in new file """
    for item in dirs:
        if os.path.isfile(input_path+item):
            """ Get path of original photos """
            output_path = "API/config/api/backend/convertedImages/"
            image = Image.open(input_path+item)
            f, e = os.path.splitext(output_path+item)

            """ Do image processing tasks """
            convert_image = image.convert('RGB')
            image_enhance = ImageEnhance.Contrast(convert_image)
            contrast = image_enhance.enhance(8)

            # If image needs to be binarized
            """ Save newly converted image to new folder for ML tasks """
            contrast.save(f + '.jpg', 'JPEG', quality=100)

imageProcessing()

def imageContours():
    """ Do image processing/convert image and save in new file """
    img = cv2.imread("API/config/api/backend/originalImages/test.png", 0)
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite("API/config/api/backend/contourImages/test.jpg", edges)

imageContours()

#def invertImage(image_name):
#    ImageOps.invert(image_name)

#cv2.imshow(invertImage("API/config/api/backend/originalImages/test.png"))