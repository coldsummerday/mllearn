# -*- coding:utf-8 -*-
import numpy as np






def get_patch(input_array, i, j, kernel_width,
                    kernel_height, stride):
    '''
    获得移动后input的array
    '''
    i*=stride
    j*=stride
    max_height = i + kernel_height
    max_width = j + kernel_width
    if input_array.ndim == 3:
        max_z = input_array.shape[0] + 1
        return input_array[0:max_z, i:max_height, j:max_width]
    else:
        return input_array[i:max_height, j:max_width]
def get_max_index(arr):
    '''
    获取数组中的最大值，返回坐标
    :param arr:
    :return:
    '''
    idx = np.argmax(arr)
    return (int(idx / arr.shape[1]), idx % arr.shape[1])
