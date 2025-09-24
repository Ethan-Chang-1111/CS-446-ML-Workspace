import numpy as np


N = np.array([[1,1,1,1], [0,0,1,1], [0,0,0,1], [1,0,0,1], [0,1,0,1]])
y = np.array([1,1,-1,-1,-1])

def perceptron():
    # while True:
    w = np.array([0,0,0,0])
    while True:
        changed = False
        for n, i in zip(N, y):
            res = np.matmul(w.transpose(), n) * i
            # print(res)
            if res <= 0:
                w = w + (n*i)
                changed = True
            print(w)
        if not changed:
            print('done')
            break


perceptron()