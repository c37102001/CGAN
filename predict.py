import argparse
import torch
import numpy as np
from torchvision.utils import save_image
from ACGAN import Generator
import ipdb

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

generator = Generator()
generator.load_state_dict(torch.load('./checkpoint/generator.ckpt'))


parser = argparse.ArgumentParser()
parser.add_argument("--label_path", type=str, default='./sample_test/my_sample_human_testing_labels.txt')
parser.add_argument("--output_path", type=str, default='./test_output')
opt = parser.parse_args()


def get_label(label_path):
    label_list = []
    with open(label_path, 'r') as f:
        count = 0
        for line in f:
            count +=1
            if not count >= 3:
                continue
            line = line.split()
            img_label = process_attr(line)
            label_list.append(img_label)
    return label_list


def process_attr(line):
    hair, eyes, face, glasses = line[0:6].index('1'), line[6:10].index('1'), line[10:13].index('1'), line[13:15].index('1')
    img_label = 24*hair + 6*eyes + 2*face + 1*glasses
    return img_label


labels = get_label(opt.label_path)
labels = LongTensor(labels)
if cuda:
    generator.cuda()

for i in range(len(labels)):
    z = FloatTensor(np.random.normal(0, 1, (1, 100)))
    img = generator.forward(z, labels[i])
    save_image(img.data, "%s/%d.png" % (opt.output_path, i))



