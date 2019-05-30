import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import os
import ipdb
import torchvision.transforms as trns
from PIL import Image
import matplotlib.image as mpimg


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

                img = mpimg.imread(os.path.join(img_path, img_name))
                if self.img_transform is not None:
                    img = self.img_transform(img)
                self.img_list.append(img)
                self.label_list.append(img_label)

    def __getitem__(self, index):
        data = dict()

        label = self.label_list[index]
        indexed = [label[0:6].index(1), label[6:10].index(1), label[10:13].index(1), label[13:15].index(1)]

        data['img'] = self.img_list[index]
        data['raw_label'] = label
        data['indexed_label'] = indexed

        return data

    def __len__(self):
        return len(self.label_list)

    def collate_fn(self, datas):
        batch = dict()

        batch['imgs'] = [data['img'] for data in datas]
        batch['raw_labels'] = [data['raw_label'] for data in datas]
        batch['indexed_labels'] = [[data['indexed_label'][i] for data in datas] for i in range(4)]

        return batch


if __name__ == '__main__':

    IMG_PATH = './selected_cartoonset100k/images'
    LABEL_PATH = './selected_cartoonset100k/cartoon_attr.txt'
    SAVE_PATH = './selected_cartoonset100k'

    ds = CartoonDataset(IMG_PATH, LABEL_PATH, image_transform=trns.Compose([trns.ToTensor()]))
    data = ds[0]
    # img1, label1 = ds[1]
    # img2, label2 = ds[2]
    # img3, label3 = ds[3]
    # imgs = torch.cat((img0.unsqueeze(0), img1.unsqueeze(0), img2.unsqueeze(0), img3.unsqueeze(0)), 0)
    # save_image(imgs.data, "my1.png", nrow=2, normalize=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=ds.collate_fn)

    for batch in loader:
        print(0)
        ipdb.set_trace()
        # imgs:(8, 3, 512, 512),  labels:(15, 8)
        # label = [*zip(*labels)]


