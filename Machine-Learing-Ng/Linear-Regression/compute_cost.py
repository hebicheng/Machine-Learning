import numpy as np 
import matplotlib.pyplot as plt

#计算Computing the cost J(θ)
def compute_cost(theta, X, y, m):
    costs = X.dot(theta) - y # 得到h(x)与y的差值
    return float(costs.T.dot(costs) / (2*m)) #矩阵运算求平方和

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

    print(compute_cost(theta,X,y,m)) #计算结果为 32.07273387745567
    