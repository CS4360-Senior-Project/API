import tensorflow as tf
from PIL import Image
import numpy as np
import string
import os

image = "../SeniorExperience/ReshapedImages/w0001_s01_pLND_r01.jpg"
model_filename = '../SeniorExperience/API/config/api/backend/modelAttempts/ocr/ocr_model.h5'

# Load the model
OCR = tf.keras.models.load_model(model_filename)

# Define the symbols
symbols = " " + string.ascii_lowercase + string.ascii_uppercase + "0123456789.,'*&!@~():`^]¢‘;|-«"

img = Image.open(image, 'r')
img = img.resize((500, 500))
img = np.asarray(img)
img = img[:, :, 0]
img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))

# Predict the text in the image
xx = OCR.predict(img)
c = ""
for i in range(len(xx[0])):
    c += (symbols[np.argmax(xx[0][i])])
print("Predicted text:", c)