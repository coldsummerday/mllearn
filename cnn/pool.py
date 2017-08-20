# -*- coding:utf-8 -*-
#!/usr/bin/local/bin/python
import numpy as np
from tools import *
class MaxPoolingLayer(object):
    def __init__(self,input_width,input_height,
                 channel_number,filter_width,
                 filter_height,stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (input_width-filter_width)/self.stride + 1
        self.output_height = (input_height - filter_height)/self.stride +1
        self.output_array = np.zeros((self.channel_number,
                                      self.output_height,self.output_width))

    def forward(self,input_array):
        self.input_array = input_array
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d,i,j] = (get_patch(
                        input_array[d],i,j,
                        self.filter_width,
                        self.filter_height,
                        self.stride
                    ).max())

    def backward(self,sensitivity_array):
        self.delta_array = np.zeros(self.input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        self.input_array[d],i,j,
                        self.filter_width,
                        self.filter_height,
                        self.stride
                    )
                    k,l =get_max_index(patch_array)
                    self.delta_array[d,
                                     i*self.stride+k,
                                    j*self.stride+l] = sensitivity_array[d,i,j]
    def update(self):
        pass
