import numpy as np
import matplotlib.pyplot as plt

n = 0
lr = 0.1

# input
X = np.array([[1,1,2,3],
              [1,1,4,5],
              [1,1,1,1],
              [1,1,5,3],
              [1,1,0,1]])

Y = np.array([1,1,-1,1,-1])

W = (np.random.random(X.shape[1])-0.5)*2

def get_show():

