from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
import telegram

from threading import Thread
import threading
import socket
import struct
from queue import Queue
from urllib import request
import logging

import joblib
import pandas as pd

global bot
bot = telegram.Bot(token="your_token") # !!! Replace the ``your_token`` field with your bot's token.


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("TIME=%(asctime)s,%(threadName)s,[%(levelname)s]:%(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "general"
        self.image = None
        self.url = None

class Thread_1(Thread):
    """
        Handle the user's request (image or url to the image).
            If it is an image, pass the image to the queue.
            If it is the url of the image, download the image first, and pass the image to the queue.
    """

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self): 
        while True:
            if s.state == "image" and s.image is not None:
                # Request is an image, pass the image to the queue.
                logger.info("receive image from message.")
                imagePath = s.image.download()
                with open(imagePath, 'rb') as fh:
                    imageData = fh.read()
                self.queue.put(imageData)
                logger.info("put image in the queue")
                s.image = None
                s.state = "pending"
            if s.state == "url" and s.url is not None:
                # Request is an url, download the image first, and pass the image to the queue.
                logger.info("receive url:{} from message, downloading...".format(s.url))
                imagePath = request.urlretrieve(s.url)[0]
                with open(imagePath, 'rb') as fh:
                    imageData = fh.read()
                self.queue.put(imageData)  
                logger.info("put image in the queue")
                s.url = None
                s.state = "pending"


class Thread_2(Thread):
    """
        Communicate with the server using TCP connection and pass the response from the server to another queue.
    """
    def __init__(self, queue_1, queue_2, host = "127.0.0.1", listenPort = 50002, connectPort = 50004):
        Thread.__init__(self)
        self.queue_1 = queue_1
        self.queue_2 = queue_2
        self.host = host
        self.listenPort = listenPort
        self.connectPort = connectPort
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def serve_client(self):
        # Pass the response from the server to another queue(queuee 2).
        (client_socket, address) = self.server_socket.accept()
        logger.info("Serving client on {}".format(address))
        data = client_socket.recv(1024)
        logger.info("receive data from client")
        self.queue_2.put(data) 
        logger.info("put data in queue 2")
        client_socket.close()      
        logger.info("connection closed")  
    
    def connect_server(self):
        # Communicate with the server using TCP connection.
        while True:
            imageData = None
            while not self.queue_1.empty():
                logger.info("found something in queue 1...")
                imageData = self.queue_1.get()
                logger.info("get data from queue 1.")
                self.queue_1.task_done()
            
            if imageData:
                try:
                    self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.soc.connect((self.host, self.connectPort))
                    logger.info("send data to server at {}:{}".format(self.host, self.connectPort))
                except socket.error as msg:
                    print(msg)

                imghead = struct.pack(b'q', len(imageData))
                self.soc.send(imghead)
                logger.info("data size:{}.".format(len(imageData)))
                i = 1024
                while True:
                    self.soc.send(imageData[i-1024:i])
                    logger.info("{} bytes data transfered...".format(i))
                    i += 1024
                    if i > len(imageData):
                        self.soc.send(imageData[i-1024:])
                        logger.info("{} bytes data transfered...".format(len(imageData)))
                        break
                logger.info("transfer complete.")
                self.soc.close()
                logger.info("connection closed.")
                return
            else:
                continue
        
                   
    def run(self): 
        self.server_socket.bind((self.host, self.listenPort))
        self.server_socket.listen(10)
        
        while True:
            self.connect_server()
            
            self.serve_client()

class Thread_3(Thread):
    """
        Generate feedback to the user which contains the probabilities of the four types of the migratory birds based on the image.
    """
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self): 
        while True:
            while not self.queue.empty():
                logger.info("found data in queue 2")
                data = self.queue.get()
                logger.info("get data from queue 2")
                bot.sendMessage(chat_id=s.id,text=data.decode("utf-8"))
                logger.info("send data to bot.")
                s.state = "over"
                self.queue.task_done()



def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hello sir, Welcome to IEMS5780 Assignment 3 demo bot. Please write\
        /help to see the commands available.")

def help(update: Update, context: CallbackContext):
    update.message.reply_text("""Available Commands :-
        /processImage - To pridict the type of bird based on uploaded image.
        /processURL - To pridict the type of bird based on the url of the image. 
        Otherwise - cannot interpret.
        """)

def processImage(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Please upload the image.")
    s.state = "image"

def processURL(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Please input the url of the image")
    s.state = "url"


def general(update: Update, context: CallbackContext):
    recvMess = update.message
    s.id = recvMess.chat_id
    if s.state == "image":
        try:
            s.image = context.bot.get_file(recvMess.document.file_id)
        except:
            s.image = context.bot.get_file(recvMess.photo[-1].file_id)
        # update.message.reply_text("Start processing the image...")
        context.bot.sendMessage(chat_id=update.message.chat_id, text="Start processing the image, please wait a moment...")
    elif s.state == "url":
        s.url = update.message.text
        context.bot.sendMessage(chat_id=update.message.chat_id, text="Start processing the url, please wait a moment...")
    elif s.state == "pending":
        update.message.reply_text("Peocessing...please wait")
    elif s.state == "over":
        update.message.reply_text("Done.")
    else:
        update.message.reply_text(
            "Sorry I can't recognize you , you said '%s'" % update.message.text)
    return 



if __name__ == "__main__":
    
    # Provide your bot's token !!!
    updater = Updater("your_token",
				use_context=True) 


    s = State()
    queue_1 = Queue()
    queue_2 = Queue()
    t1 = Thread_1(queue_1)
    t1.start()
    t2 = Thread_2(queue_1, queue_2, host = "127.0.0.1", listenPort = 50002, connectPort = 50004)
    t2.start()
    t3 = Thread_3(queue_2)
    t3.start()

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('help', help))
    updater.dispatcher.add_handler(CommandHandler('processImage', processImage))
    updater.dispatcher.add_handler(CommandHandler('processURL', processURL))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, general))
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, general))
    updater.dispatcher.add_handler(MessageHandler(Filters.document, general))
    updater.start_polling()   

    for t in threading.enumerate():
        if t != threading.main_thread():
            t.join()

