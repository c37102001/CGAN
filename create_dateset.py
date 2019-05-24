import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import os
import ipdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as trns
from PIL import Image

IMG_PATH = './selected_cartoonset100k/images'
LABEL_PATH = './selected_cartoonset100k/cartoon_attr.txt'
SAVE_PATH = './selected_cartoonset100k'


class CartoonDataset(Dataset):
    def __init__(self, img_path, label_path, image_transform=None):
        self.img_list = []
        self.label_list = []
        self.img_transform = image_transform
        with open(label_path, 'r') as f:
            count = 0
            for line in f:
                count += 1
                if not count >= 3:
                    continue
                line = line.split()
                if len(line) == 1:
                    self.total = int(line[0])
                elif len(line) == 15:
                    self.title = ' '.join(line) + '\n'
                else:
                    img_name, img_label = process_attr(line)
                    self.img_list.append(os.path.join(img_path, img_name))
                    self.label_list.append(img_label)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        # img = img_path
        # img = mpimg.imread(img_path)
        img = Image.open(img_path)
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label

    def __len__(self):
        return len(self.label_list)


def process_attr(line):
    img_name = line.pop(0)
    hair, eyes, face, glasses = line[0:6].index('1'), line[6:10].index('1'), line[10:13].index('1'), line[13:15].index('1')
    img_label = 24*hair + 6*eyes + 2*face + 1*glasses
    return img_name, img_label


if __name__ == '__main__':

    transform = trns.Compose([
        trns.Resize((512, 512)),
        trns.ToTensor(),
    ])

    ds = CartoonDataset(IMG_PATH, LABEL_PATH, image_transform=transform)
    img0, label0 = ds[0]
    img1, label1 = ds[1]
    img2, label2 = ds[2]
    img3, label3 = ds[3]
    imgs = torch.cat((img0.unsqueeze(0), img1.unsqueeze(0), img2.unsqueeze(0), img3.unsqueeze(0)), 0)
    # save_image(imgs.data, "my1.png", nrow=2, normalize=True)
    # ipdb.set_trace()
    # plt.imsave(os.path.join(SAVE_PATH, '1.png'), img1)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    for imgs, labels in loader:
        print(0)
        # imgs:(8, 3, 512, 512),  labels:(8)
        ipdb.set_trace()

