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
                img_name, img_label = line[0], [int(i) for i in (line[1:])]
                self.img_list.append(os.path.join(img_path, img_name))
                self.label_list.append(img_label)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = Image.open(img_path)
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, label

    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':

    ds = CartoonDataset(IMG_PATH, LABEL_PATH, image_transform=trns.Compose([trns.ToTensor()]))
    # img0, label0 = ds[0]
    # img1, label1 = ds[1]
    # img2, label2 = ds[2]
    # img3, label3 = ds[3]
    # imgs = torch.cat((img0.unsqueeze(0), img1.unsqueeze(0), img2.unsqueeze(0), img3.unsqueeze(0)), 0)
    # save_image(imgs.data, "my1.png", nrow=2, normalize=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    for imgs, labels in loader:
        print(0)
        # imgs:(8, 3, 512, 512),  labels:(15, 8)
        label = [*zip(*labels)]
        ipdb.set_trace()

