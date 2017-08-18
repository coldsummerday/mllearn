# -*- coding:utf-8 -*-
#!/usr/bin/local/bin/python
import numpy as np
class ConvLayer(object):
    def __init__(self,input_width,input_height
                 ,channel_number,filter_width
                 ,filter_height,filter_number
                 ,zero_padding,stride,activator,
                 learning_rate):
        '''
        :param input_width: 输入矩阵的宽
        :param input_height:输入矩阵的高
        :param channel_number:
        :param filter_width:共享权重的filter矩阵宽
        :param zero_padding:补几圈0
        :param stride:窗口每次移动的步长
        :param activator:激励函数
        :param learning_rate:学习率
        :param filter_height共享权重的filter矩阵宽
        :param filter_number filter的深度
        '''
        self.input_width = input_height
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.activator = activator
        self.learning_rate = learning_rate
        self.output_height = ConvLayer.calculate_output_size(
            self.input_height,filter_height,zero_padding,stride
        )
        self.output_width = ConvLayer.calculate_output_size(
            self.input_width,self.filter_width,zero_padding,stride
        )
        self.output_array = np.zeros((self.filter_number,
                                      self.output_height,
                                      self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,
                                       filter_height,
                                       self.channel_number))

    @staticmethod
    def calculate_output_size(input_size,filter_size
                              ,zero_padding,stride):
        return int( (input_size - filter_size + 2 * zero_padding) / stride + 1)

    def forward(self,input_array):
        '''
        计算卷积层的输出
        :param input_array: 前一层的输出
        :return: 没有返回，输出结果保存到self.output_array
        '''
        self.input_array = input_array
        self.padded_input_array = padding(input_array,self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array,filter.get_weights(),
                 self.output_array[f],self.stride,filter.get_bias())
        element_wise_op(self.output_array,self.activator.forward)
    def bp_sensitivity_map(self,sensitivity_array,activator):
        '''
        卷积层反向传播算法的实现
        1，将误差项传递到上一层
        2：计算每个参数的梯度
        3：更新参数
        :param sensitivity_array: 本层的sensitivity map
        :param activator: 上一层的激活函数
        :return:
        '''
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        #full 卷积
        expanded_width = expanded_array.shape[2]
        #获得补0 数
        zp = (self.input_width+self.filter_width-1-expanded_width)/2
        padded_array = padding(expanded_array,zp)
        #创建初始误差矩阵
        self.delta_array = self.create_delta_array()

        #对于具有多个filter的卷积层来说，最终传递到上一层的sensitivity map
        #相当于把所有的filter的sensitivity map之和
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重翻转180度

            flipped_weights = np.array(map(
                lambda i: np.rot90(i, 2),
                filter.get_weights()))
            #python3运行的时候这个map有问题
            #计算每一个filter的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f],flipped_weights[d],delta_array[d],1,0)
            self.delta_array+=delta_array
        #创建激活函数矩阵（卷积反向传播误差项的第二项）
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array,self.activator.backward)
        self.delta_array *= derivative_array

    def bp_gradient(self,sensitivity_array):
        '''
        计算梯度，包括权重跟偏置项
        :param sensitivity_array:
        :return:
        '''
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            filter = self.filters[f]
            for d in range(filter.get_weights().shape[0]):
                conv(self.padded_input_array[d],expanded_array[f],
                    filter.weights_grad[d],1,0)
            filter.bias_grad = expanded_array[f].sum()
    def expand_sensitivity_map(self,sensitivity_array):
        '''
        将步长为S的map 还原成步长1的map
        :param sensitivity_array:
        :return:
        '''
        expanded_depth = sensitivity_array.shape[0]
        expanded_height = (self.input_height-self.filter_height+2*self.zero_padding+1)
        expanded_width = (self.input_width-self.filter_width+2*self.zero_padding+1)
        expanded_array = np.zeros((expanded_depth,expanded_height,expanded_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expanded_array[:,i_pos,j_pos]=sensitivity_array[:,i,j]
        return expanded_array
    def create_delta_array(self):
        return np.zeros((self.channel_number,self.input_height,self.input_width))
    def update(self):
        '''
        更新这一层的权重跟偏置项，很简单依次更新每一个filter就行了
        :return:
        '''
        for filter in self.filters:
            filter.update(self.learning_rate)
    def backward(self, sensitivity_array, activator=None):
        if not activator:
            activator = self.activator
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)
def padding(input_array, zp):
    '''
    将输入矩阵补0
    :param input_array:
    :param zp: 补0的圈数
    :return:
    python3 玄学除法，int 变float
    '''
    zp = int(zp)
    if zp ==0:
        return input_array
    else:
        if input_array.ndim==3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padder_array = np.zeros((input_depth,input_height+2*zp,input_width+2*zp))
            padder_array[:,zp:zp+input_height,zp:zp+input_width]=input_array
            return padder_array
        elif input_array.ndim==2:
            input_height = input_array.shape[0]
            input_width = input_array.shape[1]
            padder_array = np.zeros((input_height+2*zp,input_width+2*zp))
            padder_array[zp:zp+input_height,zp:zp+input_width]=input_array
            return padder_array
def element_wise_op(array, op):
    '''
    对numpy数组元素依次进行op操作（这里是函数）
    :param array:
    :param op:
    :return:
    '''
    for i in np.nditer(array,
                       op_flags=['readwrite']):
        i[...] = op(i)
def conv(input_array,kernel_array,output_array,stride,bias):
    '''
    计算卷积
    :param input_array:
    :param kernel_array:
    :param output_array:
    :param stride:
    :param bias:
    :return:
    '''
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            #依次计算每一格的卷积
            output_array[i][j] = (get_patch(input_array,i,j,
                                            kernel_width,kernel_height,stride) * kernel_array ).sum()+bias
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
class Filter(object):
    #Filter 类 保存了卷积层的参数以及梯度，并用梯度下降的办法更新参数
    #权重随机初始化为一个很小的值，而偏置项初始化为0。
    def __init__(self,width,height,depth):
        self.weights = np.random.uniform(-1e-4,1e-4,(depth,height,width))
        self.bias =0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0
    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias))
    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self,learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad
class ReluActivator(object):
    def forward(self,weighted_input):
        return max(0,weighted_input)
    def backward(self,output):
        return 1 if output > 0 else 0

