#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Sigmoid激活函数类
from fc import *
import numpy as np
from conv import *
from activator import *
from pool import *
import sys

#构造 MNIST手写数字识别的简单CONV网络
class Network(object):
    def __init__(self):
        self.layers = []
        self.convLayer = ConvLayer(28,28,1,
                                   6,6,1,1,2,ReluActivator(),0.001)
        self.layers.append(self.convLayer)

        self.poolLayer = MaxPoolingLayer(13,13,1,4,4,1)
        self.layers.append(self.poolLayer)

        self.fcLayer1 = FullConnectedLayer(100,300,0.001,SigmoidActivator())
        self.layers.append(self.fcLayer1)

        self.fcLayer2 = FullConnectedLayer(300,10,0.001,
                                           SigmoidActivator())
        self.layers.append(self.fcLayer2)

    def predict(self,sample):
        output = sample

        self.convLayer.forward(output)
        output = self.convLayer.output_array

        self.poolLayer.forward(output)
        output = self.poolLayer.output_array
        output = output.reshape((output.size,1))

        self.fcLayer1.forward(output)
        output = self.fcLayer1.output

        self.fcLayer2.forward(output)
        output = self.fcLayer2.output

        return output

    def update_weight(self):
        for layer in self.layers:
            layer.update()

    def cal_gradient(self,label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output) * (self.layers[-1].output)

        self.fcLayer2.backward(delta)
        delta = self.fcLayer2.delta


        self.fcLayer1.backward(delta)
        delta = self.fcLayer1.delta
        delta =delta.reshape((1,10,10))

        self.poolLayer.backward(delta)
        delta = self.poolLayer.delta_array

        self.convLayer.backward(delta)

    def train_one_sample(self,label,sample):
        self.predict(sample)
        self.cal_gradient(label)
        self.update_weight()

    def train(self,labels,data_set,epoch):
        count=0
        for i in range(epoch):
            count+=1
            for d in range(len(data_set)):
                temp = np.array(labels[d])
                l =temp.reshape(len(temp),-1)
                temp = np.array(data_set[d])
                d = temp.reshape((1, 28, 28))
                self.train_one_sample(l,d)

