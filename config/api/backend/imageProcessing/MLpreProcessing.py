import cv2
import os
import numpy as np

input_dir = '../AllHandwritingImages'
output_dir = '../SeniorExperience/ReshapedImages'

for filename in os.listdir(input_dir):
    # Load image
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    
    # Preprocessing

    # Resize image to (384,384)
    resized_img = cv2.resize(img, (256,256))

    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Add new dimension to create (384,384,1) array
    reshaped_img = np.expand_dims(gray_img, axis=-1)


    # Save
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, gray_img)