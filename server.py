import socket
import threading
from threading import Thread
from queue import Queue
from io import BytesIO
import tensorflow as tf
from bird import f1
from PIL import Image
import numpy as np
import struct
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("TIME=%(asctime)s,%(threadName)s,[%(levelname)s]:%(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class ListenThread(Thread):
    """
        Listen to the telegram bot’s request (image) using TCP connection, and pass the request to a queue.
    """
    def __init__(self, queue, host = "127.0.0.1", listenPort = 50004):
        Thread.__init__(self) # important!
        self.queue = queue
        self.host = host
        self.listenPort = listenPort
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def deal_image(self, client_socket, address):
        #Ref: https://segmentfault.com/a/1190000016324980
        logger.info("Receive image from {0}.".format(address)) 
        while True:
            fileinfo_size = struct.calcsize('q')
            buf = client_socket.recv(fileinfo_size)   # size of the image
            
            if buf:
                filesize = struct.unpack('q', buf)[0]
                logger.info("image size:{}".format(filesize))
                recvd_size = 0
                data = b''
                while not recvd_size == filesize:
                    if filesize - recvd_size > 1024:
                        data += client_socket.recv(1024)
                        recvd_size += 1024
                    else:
                        data += client_socket.recv(1024)
                        recvd_size = filesize
                    logger.info("{} bytes data received...".format(recvd_size))
                logger.info("Received image completely.")
            client_socket.close()   
            logger.info("connection closed.")
            break
        image = Image.open(BytesIO(data))
        return image

    def serve_client(self, client_socket, address):
        logger.info("Serving client on {}".format(address))

        image = self.deal_image(client_socket, address)
        self.queue.put(image) 
        logger.info("Put image in the queue.")


    def run(self): 
        self.server_socket.bind((self.host, self.listenPort))
        self.server_socket.listen(10)
        while True:
            (client_socket, address) = self.server_socket.accept()
            self.serve_client(client_socket, address)

class ProcessThread(Thread):
    """
        Process the request in the queue, and respond to the telegram bot.
    """
    def __init__(self, queue, host = "127.0.0.1", connectPort = 50002):
        Thread.__init__(self) 
        self.queue = queue
        self.host = host
        self.connectPort = connectPort
        

    def predict_img(self, image):
        # image: PIL instance
        logger.info("predict the type of bird in the image using loaded model...")
        image = image.resize((180, 180)) #(img_height,img_width)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        predictions = model.predict(input_arr)
        score = tf.nn.softmax(predictions[0])
        class_names = ['Asian Brown Flycatcher', 'Blue Rock Thrush', 'Brown Shrike', 'Grey-faced Buzzard']
        data = "{}:{:.2f}%, {}:{:.2f}%, {}:{:.2f}%, {}:{:.2f}%".format(class_names[0], 100 * score[0], class_names[1], 100 * score[1], class_names[2], 100 * score[2], class_names[3], 100 * score[3])
        logger.info("prediction result:{}".format(data))
        return data
            
    def run(self): 
        while True:
            while not self.queue.empty():
                image = self.queue.get() 
                logger.info("get image from queue")
                data = self.predict_img(image)
                try:
                    self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.soc.connect((self.host, self.connectPort))
                    logger.info("connect to the bot at {}:{}".format(self.host, self.connectPort))
                except socket.error as msg:
                    logger.error(msg)
                self.soc.send(data.encode("utf-8"))
                logger.info("sent prediction result.")
                self.queue.task_done()
                self.soc.close()
                logger.info("connection closed")


if __name__ == "__main__":

    # To load the saved model
    model = tf.keras.models.load_model('./bird_model', custom_objects={'f1':f1})

    queue = Queue()
    t1 = ListenThread(queue, listenPort = 50004)
    t1.start()
    t2 = ProcessThread(queue, connectPort = 50002)
    t2.start()
    for t in threading.enumerate():
        if t != threading.main_thread():
            t.join()