import json
import numpy as np
import os
import random
#import tensorflow as tf
import time

from  matplotlib.image import imread

class Data:
    def __init__(self):
        self.data = []
        self.classification = []
        self.training = []
        self.test = []
        self.class_dict = dict()

    def load_data(self):
        class_list = [
                "bedroom",
                "suburb",
                "industrial",
                "kitchen",
                "living_room",
                "coast",
                "forest",
                "highway",
                "inside_city",
                "mountain",
                "open_country",
                "street",
                "tall_building",
                "office",
                "store"
                ]

        directory = os.walk("data/")
        for x, y, z in directory:
            a = x.replace("data/","")
            if a == "": continue
            self.class_dict[class_list[int(a)]] = [len(self.data), 0]
            for f in z:
                if ".jpg" not in f: continue
                self.class_dict[class_list[int(a)]][1] += 1
                self.classification.append(class_list[int(a)])
                self.data.append(imread(x+"/"+f))

    def split_data(self, percent_train):
        """

        @param percent_train: percentage of all data to be used for training
        @param percent_val:   percentage of training data to be used for validation
        """
        for key in self.class_dict.keys():
            train = int(self.class_dict[key][1] * percent_train)
            for _ in range(train):
                stay = True
                while stay:
                    i = random.randrange(self.class_dict[key][0],self.class_dict[key][0] + self.class_dict[key][1])
                    if i not in self.training:
                        self.training.append(i)
                        stay = False
        self.test = list(range(len(self.data)))
        self.test = list(set(self.test) - set(self.training))

        return self.training, self.test

    def get_data(self):
        train_x = []
        train_y = []
        test_x  = []
        test_y  = []

        for i in self.training:
            train_x.append(self.data[i])
            train_y.append(self.classification[i])

        for i in self.test:
            test_x.append(self.data[i])
            test_y.append(self.classification[i])

        return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    d = Data()
    d.load_data()
    d.split_data(.8)
    train_x, train_y, test_x, test_y = d.get_data()
    print(train_x)
    x = train_x[0].shape[0] - 220
    y = train_x[0].shape[1] - 220
    if x: tx = random.randrange(0, x)
    else: tx = 0
    if y: ty = random.randrange(0, y)
    else: ty = 0
    print(train_x[0][tx:tx+220,ty:ty+220].shape)
    #dd = dict()
    #for datum in d.data:
    #    if str(datum.shape[1]) not in dd.keys(): dd[str(datum.shape[1])] = 0
    #    dd[str(datum.shape[1])] += 1

    #s = 0
    #k = []
    #for key in dd.keys():
    #    if dd[key] > 100:
    #        print("100: ",end="")
    #        print(dd[key])
    #        s += dd[key]
    #        k.append(key)
    #    elif dd[key] > 5:
    #        print("10:  ",end="")
    #        print(dd[key])
    #        s += dd[key]
    #        k.append(key)
    #print(s)
    #print(len(d.data))
    #print(len(k))
    #print(k)
    #d.split_data(.8)
    #train_x, train_y, test_x, test_y = d.get_data()
    #print(train_x[0].shape)
