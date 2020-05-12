import numpy as np
import matplotlib.pyplot as plt

epoch = 1000
num = 100
learningrate = 0.001
np.random.seed(1)
#输入参数为五个数  共一百组数据
x1 = np.random.rand(5, num)
nose = np.random.randn(1, num) * 0.1            #噪声
y = x1.sum(0) + nose
L = 2                                           #两层网络

def initiarion_parameters_deep(layer_dims):
    '''
    输入包含输出层与输出层的各层节点数
    初始化参数
    :param layer_dims:
    :return:
    '''
    parameters = {}
    for i in range(1, L+1):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])
        parameters['b' + str(i)] = np.random.randn(layer_dims[i], 1)
    parameters['A0'] = x1
    return parameters

parameters = initiarion_parameters_deep([5, 8, 1])

def linear_forward(A, W, b):                 #前向传播
    return np.dot(W, A) + b

def relu(x):
    x[x < 0] = 0
    return x

def L_model_forward(X, parameters):          #L层网络向前传播
    A = X

    for i in range(1, L+1):

        A_pre = A
        A = linear_forward(A_pre, parameters['W' + str(i)], parameters['b' + str(i)])
        A = relu(A)
        parameters['A' + str(i)] = A

    return A

def backward(dZ, i):                        #后向传播
    dw = 1/num * np.dot(dZ, parameters['A' + str(i-1)].T)
    db = 1/num * np.sum(dZ)
    da = np.dot(parameters['W' + str(i)].T, dZ)
    parameters['W' + str(i)] -= learningrate * dw
    parameters['b' + str(i)] -= learningrate * db

    return da


for i in range(epoch):                      #训练部分
    result = []
    Z = L_model_forward(x1, parameters)
    L = len(parameters) // 3
    loss = np.sum(pow((y-Z), 2))
    dz = -2 * (y-Z)
    for i in range(L, 0, -1):
        dz = backward(dz, i)
    print('LOSS = ', np.sum(loss))

test = np.ones((5, 1))
print(test)
print("----------")
print(L_model_forward(test, parameters))




