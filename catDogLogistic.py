import os
import numpy as np

from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical

# Funciton to load and process images from a directory
def load_images_from_directory(directory):
    images = []
    labels = []
    class_labels = sorted(os.listdir(directory))
    
    for class_label in class_labels:
        if class_label == "cats" or class_label == "dogs":
            class_path = os.path.join(directory, class_label)
            for file_name in os.listdir(class_path):
                img_path = os.path.join(class_path, file_name)
                img = load_img(img_path, target_size=(224, 224))
                img_gray = img.convert('L')
                img_array= img_to_array(img_gray)
                img_array = img_array.reshape(img_array.shape[0], -1)
                images.append(img_array)
                labels.append(class_labels.index(class_label))

    images = np.array(images)
    images = images.reshape(images.shape[0], -1)
    labels = to_categorical(labels, num_classes=len(class_labels))

    return images, labels


# Setting path of the dataset
dataset_path = "C:\\Users\\Lenovo\\Documents\\TrainDataPython\\images"

# Loading and processing dataset
images, lables = load_images_from_directory(dataset_path)

# Split the data into training and testing sets
X = images
Y = np.concatenate([np.array(["Cat"] * 30), np.array(["Dog"] * 30)])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model on flattened training data
model.fit(X_train, Y_train)

# Loading image
image_path = os.path.join(dataset_path, "C:\\Users\\Lenovo\\Documents\\TrainDataPython\\dog.jpg")
image = load_img(image_path, target_size=(224, 224))

# Testing data
gray_image = image.convert('L')
array_image = img_to_array(gray_image)
array_image = array_image.reshape(1, -1)

# Prediction
output = model.predict(array_image)
print(f"The image corresponds to a {output[0]}")

# Accuracy of the module
excepcted_data = ['Dog']
accuracy_score = accuracy_score(Y_test, model.predict(X_test))
print(f'Accuracy {accuracy_score}')