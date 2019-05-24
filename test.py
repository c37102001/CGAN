import torch
import torch.nn as nn
import ipdb
from itertools import product
import numpy as np

pred = np.array([[1.3, 2.3, 3.6], [2.1,3.4,4.5]])
gt = np.array([[1, 2, 3], [3,4,5]])

ipdb.set_trace()
print(np.mean(pred == gt))