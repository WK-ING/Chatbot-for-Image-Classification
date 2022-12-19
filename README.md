# Chatbot-for-Image-Classification
 This is a chatbot based on Telegram to classify different kinds of bird.

## Table of Contents

- [Background](#background)
    - [Data Collection](#data-collection)
    - [Data Preprocessing](#data-preprocessing)
    - [Training CNN](#training-a-convolutional-neural-network)
    - [Deploying the Model as the Server Program](#deploying-the-model-as-the-server-program)
    - [Communicating with your Server Program in the Telegram Bot](#communicating-with-your-server-program-in-the-telegram-bot)
- [Framework](#framework)
- [Usage](#usage)
- [Demo](#demo)

## Background
In this project, we are going to 
1. collect the images for four common types of the migratory birds, 
2. build a convolutional neural network for classifying the four types of birds, 
3. write a server program to run our trained model and 
4. write a telegram bot to communicate with our server program.

The four types of migratory birds are Asian Brown Flycatcher, Blue Rock Thrush, Brown Shrike and Grey-faced Buzzard. The copyright of the image belongs to Google Image and their respective owners.

The details are as follows:
### Data Collection
In this step, we need to collect enough pictures for training. Here we used the ``google_images_download`` python package to automately download the images. See help in [manually installing using CLI](
https://google-images-download.readthedocs.io/en/latest/installation.html) and [Usage](https://www.geeksforgeeks.org/how-to-download-google-images-using-python/). 

Rename the files after downloading:
```python
from os import listdir
from os.path import isfile, join
from os import rename
mypath = "Grey-faced Buzzard" # modify it to the kind of your images
filename_tag = "Grey-faced Buzzard" # modify it to the kind of your images
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
counter = 1
for name in onlyfiles:
    oldname = mypath+"\\"+name
    newname = mypath+"\\"+"{0:03}".format(counter)+"_"+filename_tag+".jpg"
    rename(oldname, newname)
    counter = counter + 1
```
### Data Preprocessing
- For each type of bird, select 40 images as training set, select 10 images as validation set and select 10 images as testing set manually.
- Report the IDs of the select images, both training set and testing set, using lists.
    - For example, images of Asian Brown Flycatcher used in the training set are [1, 2, 3, …]
- Put three datasets into three separate folders, which are “train”, “val” and “test”. Inside each folder, there are five four folders, using the name of the migratory birds as the folder name. The images are stored in each subfolder.

### Training a Convolutional Neural Network
- Save program as bird.py
- Train CNN model based on the training set and the validation set.
- Compute (i) accuracy and (ii) F-measure based on the testing set.
- Save model as ``bird_model`` using ``h5py``.

### Deploying the Model as the Server Program
- Save server program as ``server.py``.
- Create a thread to listen to the telegram bot’s request (image) using TCP
connection, and pass the request to a queue.
- Create another thread to process the request in the queue, and respond to the telegram bot.

### Communicating with your Server Program in the Telegram Bot
- Create a bot based on Telegram. [See](https://core.telegram.org/bots/features#botfather).
- Create a thread to handle the user’s request (image or url to the image).
    - If it is an image, pass the image to the queue.
    - If it is the url of the image, download the image first, and pass the image to the queue.
- Create another thread to communicate with the server using TCP connection (Note: image classification is done in the ``server.py``.), and pass the response from the server to another queue.
- Create one more thread to generate feedback to the user.
    - In the response, we only need to answer the probabilities of the four types of the migratory birds based on the image.

## Framework
- ``bird.py`` file stores programs for data preparations and training the model.
- ``bot.py`` program stores telegram services.
- ``server.py`` program stores backend server.
- ``bird_model`` folder stores trained CNN model.

![image](/image/Framework.png)

## Usage
1. Replace the ``your_token`` field in ``line 217`` and ``line 21`` of ``bot.py`` with your own bot's token.
2. Run ``server.py`` and ``bot.py``.
3. Open Telegram app, find your own bot, send messages to it. 

## Demo
![image](/image/start-demo.png)
![image](/image/processURL-demo.png)
![image](/image/processImage-demo.png)
