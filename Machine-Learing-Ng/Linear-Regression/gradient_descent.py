import numpy as np 
import matplotlib.pyplot as plt
from compute_cost import compute_cost
# Gradient descent
def gradient_descent(theta,alpha, iterations, X, y, m):
    j_history = []
    for i in range(iterations):
        costs = X.dot(theta) - y
        theta = theta - costs.T.dot(X).T * (alpha / m)
        j_history.append(compute_cost(theta,X,y,m))
    return theta, j_history
if __name__ == "__main__":
    # 加载数据
    data = np.loadtxt("data/ex1data1.txt",dtype="float",delimiter=",")

    # 分别将population、profit存入 X,y
    # 注意保留原始维度，便于矩阵运算
    X = data[:,:1]
    y = data[:,1:2]
    # 为X添加一行，方便用矩阵运算表示h(x)
    X = np.insert(X,0,values=np.ones(1),axis=1)

    m = X.shape[0] # 表示样本数量
    theta = np.zeros((2,1)) #初始theta
    alpha = 0.01 #学习速率
    iterations = 2000

    theta, J_history = gradient_descent(theta,alpha, iterations, X, y, m)

    # 画出梯度下降过程中的收敛情况
    plt.figure()
    plt.plot([i for i in range(len(J_history))], J_history)
    plt.title("learning rate: %f" % alpha)
    plt.show()

    