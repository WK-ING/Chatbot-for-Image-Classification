# Imports
import os 
import shutil

import numpy as np
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop,SGD,Adam

from tensorflow.keras import backend as K

import h5py

def copyFiles(sourceDir, targetDir, indexList):
    """ 
    Copy files from source directory to target directory according to an index list. 

    :param sourceDir: str, the path of source directory 
    :param targetDir: str, the path of target directory 
    :param indexList: list of integers which specifies the index of files that will be copied
    :returns 

    """
    indexList = [str(i).zfill(3) for i in indexList]

    if not os.path.isdir(targetDir):
        os.makedirs(targetDir)
    
    for f in os.listdir(sourceDir):
        if f[:3] in indexList:
            sourceFile = os.path.join(sourceDir, f)
            targetFile = os.path.join(targetDir, f)
            shutil.copy(sourceFile, targetFile)
    print("{} has been copied to {}".format(sourceDir, targetDir))

def f1(y_true, y_pred):
    """F-measure computation.
    Reference: https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    """
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def seed_tensorflow(seed=2022):
    # for reproducibility
    # Reference: https://blog.csdn.net/weixin_43987408/article/details/122492885
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed) 


if __name__ == '__main__':
    seedValue = 0
    seed_tensorflow(seedValue)

    # Data Collection and Preprocessing

    # Preprocess data of Asian Brown Flycatcher images
    id_train_asian = [3, 4, 6, 7, 10, 11, 13, 14, 15, 16, 19, 20, 21, 24, 25, 26, 27, 28, 31, 32, 33, 36, 37, 39, 42, 43, 45, 53, 54, 55, 57, 58, 60, 62, 65, 68, 70, 71, 72, 73]
    id_val_asian = [76, 77, 80, 81, 82, 83, 84, 85, 86, 87]
    id_test_asian = [88, 89, 90, 91, 92, 94, 95, 97, 98, 99]
    copyFiles("./BirdDataset/Asian Brown Flycatcher", "./train/Asian Brown Flycatcher", id_train_asian)
    copyFiles("./BirdDataset/Asian Brown Flycatcher", "./val/Asian Brown Flycatcher", id_val_asian)
    copyFiles("./BirdDataset/Asian Brown Flycatcher", "./test/Asian Brown Flycatcher", id_test_asian)

    # Preprocess data of Blue Rock Thrush images
    id_train_blue = [1, 2, 3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 23, 25, 28, 30, 31, 34, 35, 40, 41, 44, 45, 47, 49, 51, 59, 66, 67, 92, 90, 87, 86, 82, 79, 77]
    id_val_blue = [75, 73, 72, 68, 65, 64, 61, 57, 56, 55]
    id_test_blue = [69, 70, 74, 78, 80, 81, 85, 88, 91, 93]
    copyFiles("./BirdDataset/Blue Rock Thrush", "./train/Blue Rock Thrush", id_train_blue)
    copyFiles("./BirdDataset/Blue Rock Thrush", "./val/Blue Rock Thrush", id_val_blue)
    copyFiles("./BirdDataset/Blue Rock Thrush", "./test/Blue Rock Thrush", id_test_blue)

    # Preprocess data of Brown Shrike images
    id_train_brown = [27, 29, 31, 34, 35, 36, 39, 44, 45, 46, 50, 52, 53, 56, 60, 61, 62, 63, 64, 65, 66, 67, 69, 71, 72, 73, 76, 77, 81, 84, 91, 93, 94, 95, 96, 92, 83, 82, 80, 74]
    id_val_brown = [13, 14, 16, 18, 21, 22, 23, 24, 25, 26]
    id_test_brown = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12]
    copyFiles("./BirdDataset/Brown Shrike", "./train/Brown Shrike", id_train_brown)
    copyFiles("./BirdDataset/Brown Shrike", "./val/Brown Shrike", id_val_brown)
    copyFiles("./BirdDataset/Brown Shrike", "./test/Brown Shrike", id_test_brown)

    # Preprocess data of Grey-faced Buzzard images
    id_train_grey = [34, 35, 36, 37, 39, 42, 43, 44, 50, 51, 52, 54, 55, 56, 58, 59, 60, 61, 65, 67, 68, 69, 72, 73, 74, 75, 76, 77, 79, 80, 81, 84, 85, 88, 89, 93, 91, 87, 86, 64]
    id_val_grey = [16, 17, 19, 20, 23, 24, 29, 30, 32, 33]
    id_test_grey = [3, 5, 6, 7, 8, 9, 12, 13, 14, 15]
    copyFiles("./BirdDataset/Grey-faced Buzzard", "./train/Grey-faced Buzzard", id_train_grey)
    copyFiles("./BirdDataset/Grey-faced Buzzard", "./val/Grey-faced Buzzard", id_val_grey)
    copyFiles("./BirdDataset/Grey-faced Buzzard", "./test/Grey-faced Buzzard", id_test_grey)

    # Training a Convolutional Neural Network

    # Load Image Datasets
    bs=20 # Setting batch size
    train_dir = "./train/" # Setting training directory
    val_dir = "./val/" # Setting validation directory
    test_dir = "./test/" # Setting testing directory

    # Rescale the Values for the Three Color Channels

    # All images will be rescaled by 1./255.
    train_datagen = ImageDataGenerator(rescale = 1.0/255.)
    val_datagen = ImageDataGenerator(rescale = 1.0/255.)
    test_datagen = ImageDataGenerator(rescale = 1.0/255.)

    # Resize the Images to the Same Size
    img_height = 180
    img_width = 180
    # Flow training images in batches of 20 using train_datagen generator
    # Flow_from_directory function lets the classifier directly identify the labels from the name of the directories the image lies in
    train_generator=train_datagen.flow_from_directory(train_dir,batch_size=bs, class_mode='categorical',target_size=(img_height,img_width))
    # Flow validation images in batches of 20 using val_datagen generator
    val_generator=val_datagen.flow_from_directory(val_dir,batch_size=bs, class_mode='categorical',target_size=(img_height,img_width))

    # Construct the CNN Model
    # tf.compat.v1.set_random_seed(2022) # for reproducibility
    

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),activation = "relu", input_shape =(img_height,img_width,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation = "relu") ,
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation = "relu") ,
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(550,activation="relu"), #Adding the Hidden layer
        tf.keras.layers.Dropout(0.1,seed = seedValue),
        tf.keras.layers.Dense(400,activation ="relu"),
        tf.keras.layers.Dropout(0.3,seed = seedValue),
        tf.keras.layers.Dense(300,activation="relu"),
        tf.keras.layers.Dropout(0.4,seed = seedValue),
        tf.keras.layers.Dense(200,activation ="relu"),
        tf.keras.layers.Dropout(0.2,seed = seedValue),
        tf.keras.layers.Dense(4,activation="softmax") #Adding the Output Layer
    ])

    adam=Adam(lr=0.001)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc', f1])

    # Train the Model with the Training and Validation Sets
    history = model.fit(train_generator,validation_data=val_generator, steps_per_epoch=150//bs,epochs=50,validation_steps=50//bs,verbose=2, shuffle=False, workers=1)

    # Test the Model with the Testing Set

    # Flow test images in batches of 20 using test_datagen generator
    test_generator = test_datagen.flow_from_directory(test_dir,batch_size=20,class_mode='categorical',target_size=(img_height,img_width))
    # Generate generalization metrics
    score = model.evaluate(test_generator, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]} / Test F-measure:{score[2]}')

    # Save the Model for Future Use
    model.save('./bird_model')