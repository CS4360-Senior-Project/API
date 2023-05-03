from PIL import Image
import numpy as np
import os

""" CODE REFERENCE: Dhruv Aditya Mittal https://www.kaggle.com/code/dhruvaditya/optical-character-recognition/notebook """


txt_file = open(r"../SeniorExperience/API/config/api/backend/prompt_dataset.txt", "r")
img_dir = "../SeniorExperience/OriginalImages/"
output_dir = "../SeniorExperience/ReshapedImages/"

details=[]
outputs=[]
names=[]
X=[]

for line in txt_file:
    a = line.split('#')
    outputs.append(a[1].strip('\n'))
    details.append(a[0])

for detail in details:
    a = detail.split(' ')
    names.append(a[0])

for name in names:
    print(name)
    img = Image.open(img_dir + name + '.jpg', 'r')
    # Resize image
    # img = img.resize((2521, 2521), Image.LANCZOS) # TOO BIG
    img = img.resize((500, 500), Image.LANCZOS)
    # Crop image
    left = int(0)
    upper = int(0)
    right = int(500)
    lower = int(500)
    region = (left, upper, right, lower)

    cropped_img = img.crop(region)

    cropped_img.save(output_dir + name + '.jpg')
    