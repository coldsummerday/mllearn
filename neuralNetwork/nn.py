import numpy as np

#定义tan函数以及tan函数的导数
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return  1.0-np.tanh(x)*np.tanh(x)

#定义logistich函数以及其导数
def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_derivatrive(x):
    return logistic(x)*(1-logistic(x))

class NeturalNetwork(object):
    def __init__(self,layers,activations='tanh'):
        '''
        :param layers: 一个list  包括每一层的神经元数
        :param activations:激活函数
        '''
        if activations=='tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        if activations=='logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivatrive
        self.weights = []
        for i in range(1,len(layers)-1):
            #i跟前一层的权重
            self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
            #i层跟i+1层进行赋值 权重
            self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)

    def fit(self,X,y,learning_rate = 0.2,epochs=10000):
        '''
        :param X:
        :param y:
        :param learning_rate: 学习率
        :param epochs: 学习步骤
        :return:
        '''
        #二维矩阵
        X = np.atleast_2d(X)
        #ones 矩阵全是1   shape函数返回的是行列数（返回一个List）跟X一样维度的矩阵
        temp = np.ones([X.shape[0],X.shape[1]+1])
        #temp等于第一列到最后一列跟x一样的矩阵
        temp  [:,0:-1]=X
        X= temp
        Y=np.array(y)

        #第几次循环
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            #随机取一个数，代表第i行，对i行数据进行更新
            a = [X[i]]
            #形成这一行数据为输入的神经网络,dot代表内积
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))
            #误差
            error = y[i]-a[-1]
            deltas = [error * self.activation_deriv(a[-1])]
            #开始往回算每一层的误差
            #deltas是所有权重的误差列表
            for l in range(len(a)-2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layers = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate*layers.T.dot(delta)
    def predict(self,x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a =temp
        for l in range(0,len(self.weights)):
            a = self.activation(np.dot(a,self.weights[l]))
        return a