import cv2
import keras
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import Model
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Lambda

######################################################################################
#                                                                                    #
#                         MACHINE LEARNING CODE BY JADA SACHETTI                     #
#                                                                                    #
######################################################################################

# load the csv file containing the image info
image_info = pd.read_csv('../SeniorExperience/API/config/api/backend/handwriting_dataset.csv')

# Define the stratification column to be the author's ID
stratify_col = 'wid'

# Split the dataset into training and testing sets using stratified sampling
train_data = pd.DataFrame()
test_data = pd.DataFrame()
for wid in image_info[stratify_col].unique():
    author_data = image_info[image_info[stratify_col] == wid]
    author_train_data, author_test_data = train_test_split(author_data, test_size=0.2, stratify=author_data[stratify_col])
    train_data = pd.concat([train_data, author_train_data])
    test_data = pd.concat([test_data, author_test_data])

# Define a function to extract features from an image
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized = np.expand_dims(image, axis=-1)
    # features = resized.flatten()
    return resized

# Create arrays of features for the images in the training and testing sets
train_features = np.array([extract_features(row['image_path']) for _, row in train_data.iterrows()])
test_features = np.array([extract_features(row['image_path']) for _, row in test_data.iterrows()])

# create a dictionary to map author-prompt combination to a unique label
label_dict = {}
label = 0
num_classes = len(label_dict)

for author in train_data['wid'].unique():
    for prompt in train_data['prompt'].unique():
        label_dict[(author, prompt)] = label
        label += 1

# create an array of labels for the images
train_labels = np.array([label_dict[(row['wid'], row['prompt'])] for _, row in train_data.iterrows()])
test_labels = np.array([label_dict[(row['wid'], row['prompt'])] for _, row in test_data.iterrows()])

# print the shapes of the arrays
print("Train features shape:", train_features.shape)
print("Train labels shape:", train_labels.shape)
print("Test features shape:", test_features.shape)
print("Test labels shape:", test_labels.shape)

"""  Define the siamese network architecture  """
input_shape = train_features.shape[1]
left_input = Input(input_shape)
right_input = Input(input_shape)
model = Sequential()
model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Lambda(lambda x: keras.backend.l2_normalize(x, axis=1)))  # L2 normalize embeddings
encoded_l = model(left_input)
encoded_r = model(right_input)

# Define the distance metric layer
L1_layer = Lambda(lambda tensors: keras.backend.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_r])

# Define the final output layer
prediction = Dense(1, activation='sigmoid')(L1_distance)
siamese_model = Model(inputs=[left_input, right_input], outputs=prediction)

# Compile the model
siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = siamese_model.fit(
    [train_features[:, 0], train_features[:, 1]],
    train_labels,
    batch_size=16,
    epochs=10,
    validation_data=([test_features[:, 0], test_features[:, 1]], test_labels))  ## ERRORS HERE


"""  Test Loss and Accuracy  """
# Predict similarity scores for test data
test_predictions = siamese_model.predict([test_features[:, 0], test_features[:, 1]])

# Calculate accuracy and loss on test data
test_loss, test_accuracy = siamese_model.evaluate([test_features[:, 0], test_features[:, 1]], test_labels)

print('Test Loss: ', test_loss)
print('Test Accuracy: ', test_accuracy)


"""  Model Evaluation  """
# Evaluate the Siamese Network
train_loss, train_accuracy = siamese_model.evaluate([train_features[:, 0], train_features[:, 1]], train_labels)
test_loss, test_accuracy = siamese_model.evaluate([test_features[:, 0], test_features[:, 1]], test_labels)
print('Train Loss:', train_loss)
print('Train Accuracy:', train_accuracy)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Calculate the similarity scores for the test set
test_similarity_scores = siamese_model.predict([test_features[:, 0], test_features[:, 1]])

# Calculate the confusion matrix for the test set
test_predictions = np.round(test_similarity_scores).flatten()
test_confusion_matrix = confusion_matrix(test_labels, test_predictions)
print('Test Confusion Matrix:', test_confusion_matrix)


# """ Save the trained model for loading with new input images """
# model_filename = '../SeniorExperience/API/config/api/backend/siamese_model.h5'
# siamese_model.save(model_filename)