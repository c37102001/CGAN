import torch
import torch.nn as nn
import ipdb
from itertools import product
import numpy as np


SAMPLE_PATH = './sample_test/sample_human_testing_labels.txt'
labels = []
with open(SAMPLE_PATH, 'r') as f:
    count = 0
    for line in f:
        count += 1
        if count < 3 or count > 146:
            continue
        labels.append([torch.LongTensor([int(i)]) for i in line.split()])

labels = torch.LongTensor(labels)
