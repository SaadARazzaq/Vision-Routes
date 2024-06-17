'''

-> This is the training file to train the model.
-> This file starts training the model from scratch.

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random

def main():
    # Define directory and load dataset
    directory = 'Dataset_Self_Driving_Car_'
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv(os.path.join(directory, 'driving_log.csv'), names=columns)
    pd.set_option('display.max_colwidth', 1)

    # Preprocess data
    data['center'] = data['center'].apply(path_leaf)
    data['left'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)
    data = preprocess_data(data)

    # Generate image paths and steering angles
    image_paths, steerings = load_img_steering(os.path.join(directory, 'IMG'), data)

    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)

    # Create model
    model = nvidia_model()

    # Train model
    train_model(model, X_train, y_train, X_valid, y_valid)

    # Save model
    model.save('models/nvidia_model.h5')

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail

def preprocess_data(data):
    # Preprocess data
    num_bins = 25
    samples_per_bin = 600
    hist, bins = np.histogram(data['steering'], num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5

    remove_list = []
    for j in range(num_bins):
        list_ = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j + 1]:
                list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)

    data.drop(data.index[remove_list], inplace=True)

    return data

def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(df)):
        indexed_data = df.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
        # left image append
        image_path.append(os.path.join(datadir, left.strip()))
        steering.append(float(indexed_data[3]) + 0.15)
        # right image append
        image_path.append(os.path.join(datadir, right.strip()))
        steering.append(float(indexed_data[3]) - 0.15)
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

def nvidia_model():
    kernel1 = 5
    filters1 = 24
    kernel2 = 5
    filters2 = 36
    kernel3 = 5
    filters3 = 48
    kernel4 = 3
    filters4 = 64
    kernel5 = 3
    filters5 = 64
    model = Sequential()

    # Define the model architecture
    model.add(Conv2D(filters1, (kernel1, kernel1), strides=(2, 2), input_shape=(66, 200, 3), activation='sigmoid'))  # Input layer (convolution layer)
    model.add(Conv2D(filters2, (kernel2, kernel2), strides=(2, 2), activation='elu'))  # Convolution layer
    model.add(Conv2D(filters3, (kernel3, kernel3), strides=(2, 2), activation='elu'))  # Convolution layer
    model.add(Conv2D(filters4, (kernel4, kernel4), activation='elu'))  # Convolution layer
    model.add(Conv2D(filters5, (kernel5, kernel5), activation='elu'))  # Convolution layer
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
    
    model.add(Flatten())  # Flatten layer to convert 2D feature maps to 1D feature vectors
    
    model.add(Dense(100, activation='elu'))  # Hidden layer (fully connected)
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
    
    model.add(Dense(50, activation='elu'))  # Hidden layer (fully connected)
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting

    model.add(Dense(10, activation='elu'))  # Hidden layer (fully connected)
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting

    model.add(Dense(1))  # Output layer (prediction of steering angle)

    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)  # Loss function: mean squared error

    return model



def train_model(model, X_train, y_train, X_valid, y_valid):
    history = model.fit(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch=300,
                                  epochs=200,
                                  validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()
    

def batch_generator(image_paths, steering_ang, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)

            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])

            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]

            im = img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))

def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = img_random_flip(image, steering_angle)

    return image, steering_angle

def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

def img_random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image

def pan(image):
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1),  "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

if __name__ == "__main__":
    main()
