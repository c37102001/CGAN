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
from acgan import Generator, Discriminator

os.makedirs("images", exist_ok=True)
os.makedirs("checkpoint", exist_ok=True)
SAMPLE_PATH = './sample_test/sample_human_testing_labels.txt'
# IMG_PATH = './selected_cartoonset100k/images'
# LABEL_PATH = './selected_cartoonset100k/cartoon_attr.txt'
# SAVE_PATH = './checkpoint'

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
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


def sample_image(n_row, batches_done, generator):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)))  # (144, 100)
    # Get labels ranging from 0 to n_classes for n rows
    # labels = np.array([num for num in range(n_row ** 2)])
    # labels = LongTensor(labels)

    labels = []
    with open(SAMPLE_PATH, 'r') as f:
        count = 0
        for line in f:
            count += 1
            if count < 3 or count > 146:
                continue
            line = line.split()
            labels.append([line[0:6].index('1'), line[6:10].index('1'), line[10:13].index('1'), line[13:15].index('1')])
    labels = LongTensor([*zip(*labels)])    # (4, 144)

    gen_imgs = generator.forward(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row)


def save(save_path, generator, discriminator):
    print('save model to', save_path)
    torch.save(generator.state_dict(), './checkpoint/generator.ckpt')
    torch.save(discriminator.state_dict(), './checkpoint/discriminator.ckpt')


def main():
    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()

    # Initialize generator and discriminator
    generator = Generator(opt.img_size, opt.latent_dim, opt.channels)
    discriminator = Discriminator(opt.img_size, opt.channels)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()

    ds = CartoonDataset(opt.img_path, opt.label_path)
    dataloader = DataLoader(ds, batch_size=opt.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    ds2 = CartoonDataset(opt.img_path, opt.label_path)
    dataloader2 = DataLoader(ds2, batch_size=opt.batch_size, shuffle=True, collate_fn=ds2.collate_fn)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        batch_num = 0
        for batch, batch2 in zip(dataloader, dataloader2):

            real_imgs = FloatTensor(batch['imgs']).permute(0, 3, 1, 2)  # (64, 3, 128, 128)
            batch_size = real_imgs.shape[0]                             # 64
            real_indexed_labels = LongTensor(batch['indexed_labels'])   # (4, 64)
            gen_indexed_labels = LongTensor(batch2['indexed_labels'])   # (4, 64)

            # Adversarial ground truths
            valid = FloatTensor(batch_size, 1).fill_(1.0)  # (64, 1)
            fake = FloatTensor(batch_size, 1).fill_(0.0)  # (64, 1)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            noise = torch.randn(batch_size, opt.latent_dim).type(FloatTensor)  # (64, 100)

            # Generate a batch of images (64, 3, 128, 128)
            gen_imgs = generator.forward(noise, gen_indexed_labels)

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator.forward(gen_imgs)  # (64, 1),  (64, 15)

            gen_hair_loss = auxiliary_loss(pred_label[:, 0:6], gen_indexed_labels[0])
            gen_eyes_loss = auxiliary_loss(pred_label[:, 6:10], gen_indexed_labels[1])
            gen_face_loss = auxiliary_loss(pred_label[:, 10:13], gen_indexed_labels[2])
            gen_glasses_loss = auxiliary_loss(pred_label[:, 13:15], gen_indexed_labels[3])
            gen_aux_loss = (gen_hair_loss + gen_eyes_loss + gen_face_loss + gen_glasses_loss) / 4
            g_loss = 0.5 * (adversarial_loss(validity, valid) + gen_aux_loss)
            #                                (64,1)   (64,1)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = discriminator.forward(real_imgs)

            real_hair_loss = auxiliary_loss(real_aux[:, 0:6], real_indexed_labels[0])
            real_eyes_loss = auxiliary_loss(real_aux[:, 6:10], real_indexed_labels[1])
            real_face_loss = auxiliary_loss(real_aux[:, 10:13], real_indexed_labels[2])
            real_glasses_loss = auxiliary_loss(real_aux[:, 13:15], real_indexed_labels[3])
            real_aux_loss = (real_hair_loss + real_eyes_loss + real_face_loss + real_glasses_loss) / 4

            d_real_loss = (adversarial_loss(real_pred, valid) + real_aux_loss) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator.forward(gen_imgs.detach())

            fake_hair_loss = auxiliary_loss(fake_aux[:, 0:6], gen_indexed_labels[0])
            fake_eyes_loss = auxiliary_loss(fake_aux[:, 6:10], gen_indexed_labels[1])
            fake_face_loss = auxiliary_loss(fake_aux[:, 10:13], gen_indexed_labels[2])
            fake_glasses_loss = auxiliary_loss(fake_aux[:, 13:15], gen_indexed_labels[3])
            fake_aux_loss = (fake_hair_loss + fake_eyes_loss + fake_face_loss + fake_glasses_loss) / 4

            d_fake_loss = (adversarial_loss(fake_pred, fake) + fake_aux_loss) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calculate discriminator accuracy
            r_aux = torch.split(real_aux, [6, 4, 3, 2], dim=1)
            f_aux = torch.split(fake_aux, [6, 4, 3, 2], dim=1)
            count_acc = lambda aux, labels: np.mean(aux.max(1)[1].data.cpu().numpy() == labels.data.cpu().numpy())
            r_acc = np.mean([count_acc(i, j) for i, j in zip(r_aux, real_indexed_labels)]) * 100
            f_acc = np.mean([count_acc(i, j) for i, j in zip(f_aux, gen_indexed_labels)]) * 100

            d_loss.backward()
            optimizer_D.step()
            if batch_num % 10 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, r_acc: %d%%, f_acc: %d%%] [G loss: %f]"
                    % (epoch, opt.n_epochs, batch_num, len(dataloader), d_loss.item(), r_acc, f_acc, g_loss.item())
                )
            batch_num += 1

        # if epoch % opt.sample_interval == 0:
        sample_image(12, epoch, generator)
        save(opt.save_path, generator, discriminator)


if __name__ == '__main__':
    # with ipdb.launch_ipdb_on_exception():
    #     sys.breakpointhook = ipdb.set_trace
    #     main()
    main()

