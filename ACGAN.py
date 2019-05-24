import argparse
import sys
import os
import numpy as np
import math

import torchvision.transforms as trns
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import ipdb
from itertools import product

from create_dateset import CartoonDataset

os.makedirs("images", exist_ok=True)
os.makedirs("checkpoint", exist_ok=True)
# IMG_PATH = './selected_cartoonset100k/images'
# LABEL_PATH = './selected_cartoonset100k/cartoon_attr.txt'
# SAVE_PATH = './checkpoint'

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=144, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=5, help="interval between image sampling")

parser.add_argument("--img_path", type=str, default='./selected_cartoonset100k/images')
parser.add_argument("--label_path", type=str, default='./selected_cartoonset100k/cartoon_attr.txt')
parser.add_argument("--save_path", type=str, default='./checkpoint')

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.hair_emb = nn.Embedding(6, opt.latent_dim // 4)
        self.eyes_emb = nn.Embedding(4, opt.latent_dim // 4)
        self.face_emb = nn.Embedding(3, opt.latent_dim // 4)
        self.glasses_emb = nn.Embedding(2, opt.latent_dim // 4)

        self.init_size = opt.img_size // 4  # Initial size before upsampling = 8
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim, 128 * self.init_size ** 2)        # (100, 128*8*8)
        )

        self.conv_blocks = nn.Sequential(                           # (64, 128, 8, 8)
            nn.BatchNorm2d(128),                                    # (64, 128, 8, 8)
            nn.Upsample(scale_factor=2),                            # (64, 128, 16, 16)
            nn.Conv2d(128, 128, 3, stride=1, padding=1),            # (64, 128, 16, 16)
            nn.BatchNorm2d(128, 0.8),                               # (64, 128, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),                        # (64, 128, 16, 16)
            nn.Upsample(scale_factor=2),                            # (64, 128, 32, 32)
            nn.Conv2d(128, 64, 3, stride=1, padding=1),             # (64, 64, 32, 32)
            nn.BatchNorm2d(64, 0.8),                                # (64, 64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),                        # (64, 64, 32, 32)
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),    # (64, 1, 32, 32)
            nn.Tanh(),                                              # (64, 1, 32, 32)
        )

    def forward(self, noise, labels):                               # (64, 100), (64)
        hair = self.hair_emb(labels[:, 0])                          # (64, 100)
        eyes = self.eyes_emb(labels[:, 1])
        face = self.face_emb(labels[:, 2])
        glasses = self.glasses_emb(labels[:, 3])
        gen_input = torch.cat((hair, eyes, face, glasses), 1) * noise        # (64, 100)

        out = self.l1(gen_input)                                    # (64, 128*8*8)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)       # (64, 128, 8, 8)
        img = self.conv_blocks(out)                                 # (64, 1, 32, 32)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(                           # (64, 1, 32, 32)
            *discriminator_block(opt.channels, 16, bn=False),       # (64, 16, 16, 16)  Conv stride=2
            *discriminator_block(16, 32),                           # (64, 32, 8, 8)
            *discriminator_block(32, 64),                           # (64, 64, 4, 4)
            *discriminator_block(64, 128),                          # (64, 128, 2, 2)
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 4))

    def forward(self, img):                         # (64, 1, 32, 32)
        out = self.conv_blocks(img)                 # (64, 128, 2, 2)
        out = out.view(out.shape[0], -1)            # (64, 128*2*2)
        validity = self.adv_layer(out)              # (64, 1)
        label = self.aux_layer(out)                 # (64, 4)

        return validity, label


def sample_image(n_row, batches_done, generator):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)))
    # Get labels ranging from 0 to n_classes for n rows
    # labels = np.array([num for num in range(n_row ** 2)])
    # labels = LongTensor(labels)

    hair = torch.arange(0, 6)
    eyes = torch.arange(0, 4)
    face = torch.arange(0, 3)
    glasses = torch.arange(0, 2)
    labels = LongTensor([label for label in product(hair, eyes, face, glasses)])
    gen_imgs = generator.forward(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row)


def save(save_path, generator, discriminator):
    print('save model to', save_path)
    torch.save(generator.state_dict(), './checkpoint/generator.ckpt')
    torch.save(discriminator.state_dict(), './checkpoint/discriminator.ckpt')


def main():
    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    # auxiliary_loss = torch.nn.CrossEntropyLoss()
    auxiliary_loss = torch.nn.L1Loss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    ds = CartoonDataset(opt.img_path, opt.label_path, image_transform=trns.Compose([trns.ToTensor()]))
    dataloader = DataLoader(ds, batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            labels = [*zip(*labels)]

            # imgs: (64, 1, 32, 32), labels: (64, 4)

            batch_size = imgs.shape[0]  # 64

            # Adversarial ground truths
            valid = FloatTensor(batch_size, 1).fill_(1.0)  # (64, 1)
            fake = FloatTensor(batch_size, 1).fill_(0.0)  # (64, 1)

            # Configure input
            real_imgs = imgs.type(FloatTensor)      # (64, 3, 128, 128)
            labels = LongTensor(labels)             # (64, 4)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            noise = torch.randn(batch_size, opt.latent_dim).type(FloatTensor)  # (64, 100)
            hair = torch.randint(0, 6, (batch_size, 1))
            eyes = torch.randint(0, 4, (batch_size, 1))
            face = torch.randint(0, 3, (batch_size, 1))
            glasses = torch.randint(0, 2, (batch_size, 1))
            gen_labels = torch.cat((hair, eyes, face, glasses), 1).type(LongTensor)  # (64, 4)

            # Generate a batch of images
            gen_imgs = generator.forward(noise, gen_labels)  # (64, 1, 32, 32)

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator.forward(gen_imgs)  # (64, 1),  (64, 4)
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels.type(FloatTensor)))
            #                                 (64,1)   (64,1)                   (64,4)       (64,4)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            # ipdb.set_trace()
            # Loss for real images
            real_pred, real_aux = discriminator.forward(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels.type(FloatTensor))) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator.forward(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels.type(FloatTensor))) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            pred = np.around(pred)
            gt = np.around(gt)
            d_acc = np.mean(pred == gt)

            d_loss.backward()
            optimizer_D.step()
            if i % 10 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
                )

        if epoch % opt.sample_interval == 0:
            sample_image(12, epoch, generator)
            save(opt.save_path, generator, discriminator)


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        main()
