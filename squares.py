import numpy as np
import torch
# # You only need to complete the two functions .
def numpy_squares ( k ) :
    """ return (1 , 4 , 9 , ... , k ^2) as a numpy array """
    arr = np.arange(1, k+1)
    return arr ** 2
# your code here
def torch_squares ( k ) :
    """ return (1 , 4 , 9 , ... , k ^2) as a torch array """
    tensor = torch.arange(1,k+1)
    return tensor **2