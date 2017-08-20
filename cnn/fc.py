# -*- coding:utf-8 -*-
#!/usr/bin/local/bin/python

import numpy as np

class FullConnectedLayer(object):
    def __init__(self,input_size,
                 output_size,
                 learing_rate,
                 activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.learning_rate = learing_rate
        self.W = np.random.uniform(-0.1,0.1,(output_size,input_size))
        self.b = np.zeros((output_size,1))
        self.output = np.zeros((output_size,1))

    def forward(self,input_array):
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W,input_array)+self.b
        )

    def backward(self,delta_array):
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T,delta_array
        )
        self.W_grad = np.dot(delta_array,self.input.T)
        self.b_grad = delta_array

    def update(self):
        self.W += self.learning_rate * self.W_grad
        self.b += self.learning_rate * self.b_grad