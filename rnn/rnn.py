
import  numpy as np
from cnn.activator import ReluActivator,IdentityActivator
from cnn.conv import element_wise_op


class RecurrentLayer(object):
    def __init__(self,input_width,state_width,
                 activator,learning_rate):
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        self.times = 0
        self.state_list = []
        self.state_list.append(np.zeros((state_width,1)))
        self.U = np.random.uniform(-1e-4,1e-4,(state_width,input_width))
        self.W = np.random.uniform(-1e-4,1e-4,(state_width,state_width))


    def forward(self,input_array):
        self.times +=1
        state = (np.dot(self.U,input_array)+np.dot(self.W,self.state_list[-1]))
        element_wise_op(state,self.activator.forward)
        self.state_list.append(state)


    def backward(self,sensitivity_array,activator):
        self.calc_delta(sensitivity_array,activator)
        self.calc_gradient()

    def calc_delta(self,sensitivity_array,activator):
        self.delta_list = []
        for i in range(self.times):
            self.delta_list.append(np.zeros((self.state_width,1)))
        self.delta_list.append(sensitivity_array)
        for k in range(self.times-1,0,-1):
            '''
            计算每个时刻的误差值
            '''
            self.calc_delta_k(k,activator)

    def calc_delta_k(self,k,activator):
        '''
        根据K+1时刻的delta计算k时刻的delta
        :param k:
        :param activator:
        :return:
        '''
        state = self.delta_list[k+1].copy()
        element_wise_op(self.state_list[k+1],activator.backward)
        self.delta_list[k] = np.dot(
            np.dot(self.delta_list[k+1].T,self.W),np.diag(state[:,0])
        ).T

    def calc_gradient(self):
        self.gradient_list = []#保存各个时刻的权重梯度

        for t in range(self.times+1):
            self.gradient_list.append(np.zeros((self.state_width,self.state_width)))
        for t in range(self.times,0,-1):
            self.calc_gradient_t(t)

        self.gradent = reduce(
            lambda a,b:a+b,self.gradient_list,self.gradient_list[0]
        )

    def calc_gradient_t(self,t):
        '''
        计算每个时刻T的权重梯度
        :param t:
        :return:
        '''
        gradient = np.dot(self.delta_list[t],
                          self.state_list[t-1].T)
        self.gradent_list[t] = gradient

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        self.W -= self.learning_rate * self.gradient

