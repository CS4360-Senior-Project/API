import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


directory = '../SeniorExperience/API/config/api/backend/npy-files/'
# Load the preprocessed images and labels
X = np.load(directory + 'preprocessed_images.npy')
Y = np.load(directory + 'labels.npy')
label_encoder = LabelEncoder()
label_encoder.fit(Y)
Y = label_encoder.transform(Y)
Y = to_categorical(Y)

print("X SHAPE: ", X.shape)
print("Y SHAPE: ", Y.shape)
print("Y: ", Y)

# Resize the images
X_resized = np.zeros((X.shape[0], 28, 28, 1))
for i in range(X.shape[0]):
    img = cv2.resize(X[i], (28, 28))
    X_resized[i] = img.reshape(28, 28, 1)

print("X NEW SHAPE: ", X_resized.shape)

# Split the data into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X_resized, Y, test_size=0.6, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
print(model.summary())

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))


# loss, accuracy = model.evaluate(X_test, Y_test)
# print(f'Test Loss: {loss:.4f}')
# print(f'Test Accuracy: {accuracy:.4f}')

# Save the trained model
# model.save('ocr_model.h5')