import Settings
import cv2
from mss import mss
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

class Model_worker():

    def __init__(self, name):
        self.model = self.load_model(name)

    def load_model(self, name):
        model = load_model(name)

        return model

    def __get_img(self):
        with mss() as sct:
            img = sct.grab(Settings.MONITOR)
            img = np.array(img)
            if Settings.CHANNELS == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def work(self):

        while True:
            screen_data = []
            for step in range(Settings.TIME_STEP):
                img = self.__get_img()
                #img, radius, center = lane_detection.detect(img) #line detection- for NN not required
                screen = cv2.resize(np.array(img), (Settings.WIDTH_SCREEN_DATA, Settings.HEIGHT_SCREEN_DATA))
                screen_data.append(screen)

            screen_data = np.array(screen_data)
            pred = self.model.predict(screen_data)
            print(pred)

