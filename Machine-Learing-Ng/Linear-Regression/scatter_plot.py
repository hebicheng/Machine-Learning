import numpy as np 
import matplotlib.pyplot as plt
data = np.loadtxt("data/ex1data1.txt",dtype="float",delimiter=",")

#x,y分别保存population，profit
x = data[:,0]
y = data[:,1]

plt.scatter(x,y,marker='x')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Scatter plot of training data')
plt.show()