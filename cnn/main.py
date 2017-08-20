# -*- coding: UTF-8 -*-
from loader import *
from Network import *
from datetime import datetime

import numpy

def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def evaluete(network,test_data_set,test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        temp = numpy.array(test_data_set[i])
        temp = temp.reshape((1,28,28))
        predict = get_result(network.predict(temp))
        if label != predict:
            error +=1
    return  float(error) / float(total)

def train_and_evalute():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set,train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = Network()
    print("%s start train" % datetime.now())
    while True:
        epoch += 1
        network.train(train_labels,train_data_set,1)
        print("%s epoch %d finished \n" %(datetime.now(),epoch))
        if epoch %2 ==0:
            error_ratio = evaluete(network, test_data_set, test_labels)
            print("%s after epoch %d,error ratio is %f" %(datetime.now(),epoch,error_ratio))
            if error_ratio > last_error_ratio:
                pass
            else:
                last_error_ratio = error_ratio

        if epoch ==100:
            break

if __name__ == '__main__':
    train_and_evalute()
