import numpy as np
import torch
from lib.CAModel import CAModel
from lib.CAModel2 import CAModel2

if __name__ == '__main__':
    m1 = CAModel(16, 0.2, 'cpu')
    m2 = CAModel2(16, 0.2, 'cpu')

    arr = np.random.random((8, 112, 112, 16))
    output1 = m1(torch.from_numpy(arr.astype(np.float32)), 1)
    output2 = m2(torch.from_numpy(arr.astype(np.float32)), 1)