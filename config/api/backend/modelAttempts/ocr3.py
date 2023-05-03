from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Reshape
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import string
import cv2
import os
from PIL import Image
""" CODE REFERENCE: Dhruv Aditya Mittal https://www.kaggle.com/code/dhruvaditya/optical-character-recognition/notebook """
""" THIS FILE CREATES THE OPTICAL CHARACTER RECOGNITION FILE FOR """

txt_file = open(r"../SeniorExperience/API/config/api/backend/prompt_dataset.txt", "r") # CHANGE
img_dir = "../SeniorExperience/ReshapedImages/"
img_size = (50, 50)
batch_size = 100
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

for filename in os.listdir(img_dir):
    img_path = os.path.join(img_dir, filename)
    img = Image.open(img_path).resize(img_size).convert('L')
    img = np.array(img) / 255.0
    X.append(img)
    print("FILE: ", filename)

# Convert the list to a numpy array
X = np.array(X)
# Reshape the input array to have a channel dimension of 1
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

symbols = string.printable
print("SYMBOLS: ", symbols)

# X = names # input images
Y = outputs # output labels

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, random_state=42)

# Convert to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

Y_train = np.array([np.argmax(y) for y in Y_train])
Y_test = np.array([np.argmax(y) for y in Y_test])
# X train is already preprocessed
X_test = np.array(X_test) / 255.0
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

print("X train shape: ", X_train.shape)
print("Number of samples in training set:", len(X_train))
print("Number of samples in testing set:", len(X_test))

# Define the model architecture
def OCRModel():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D((2, 2))),
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(symbols), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


model = OCRModel()
history = model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test))

loss, accuracy = model.evaluate(X_test, Y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# PREDICT AN EXAMPLE WITH THE MODEL
test_img_path = "../SeniorExperience/ReshapedImages/w0001_s01_pLND_r01.jpg"
test_img = Image.open(test_img_path).resize(img_size).convert('L')
test_img = np.array(test_img) / 255.0
test_img = np.reshape(test_img, (1, test_img.shape[0], test_img.shape[1], 1))

# Predict the output of the image
prediction = model.predict(img)
print("PREDICTION: ", prediction)
# Get the predicted label
predicted_label = symbols[np.argmax(prediction)]
print("Predicted label: ", predicted_label)

# Save the file
model_filename = '../SeniorExperience/API/config/api/backend/ocr_model.h5'
model.save(model_filename)