import cv2
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from skimage.feature import hog
from keras.models import Sequential, load_model

siamese_model = '../SeniorExperience/API/config/api/backend/siamese_model.h5'
image1 = '../SeniorExperience/API/config/api/backend/Example1.png'
image2 = '../SeniorExperience/API/config/api/backend/Example2.png'

# Load the trained Siamese network model (saved in previous code)
model = load_model(siamese_model)

# Define the function to calculate the siamese features of an image
def get_siamese_features(img):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(image, (256, 256))
    return resized

# Compute the Siamese features of the images
siamese1 = get_siamese_features(image1)
siamese2 = get_siamese_features(image2)

# Create a numpy array with the input data
X1 = np.array([siamese1])
X2 = np.array([siamese2])

# Use the trained Siamese network to predict the similarity score between the two images
similarity_score = model.predict([X1, X2])[0][0]

# Output the percent similarity score
print("The percent similarity between the two images is " + str(similarity_score*100))