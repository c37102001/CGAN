import torch
import torch.nn as nn
import ipdb


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, channels):
        super(Generator, self).__init__()

        self.hair_emb = nn.Embedding(6, latent_dim)
        self.eyes_emb = nn.Embedding(4, latent_dim)
        self.face_emb = nn.Embedding(3, latent_dim)
        self.glasses_emb = nn.Embedding(2, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling = 32
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim * 5, 64 * self.init_size ** 2)        # (500, 64*32*32)
        )

        self.conv_blocks = nn.Sequential(                           # (64, 64, 32, 32)
            nn.BatchNorm2d(64),                                     # (64, 64, 32, 32)
            nn.Upsample(scale_factor=2),                            # (64, 64, 64, 64)
            nn.Conv2d(64, 64, 3, stride=1, padding=1),              # (64, 64, 64, 64
            nn.BatchNorm2d(64, 0.8),                                # (64, 64, 64, 64)
            nn.LeakyReLU(0.2, inplace=True),                        # (64, 64, 64, 64)
            nn.Upsample(scale_factor=2),                            # (64, 64, 128, 128)
            nn.Conv2d(64, 32, 3, stride=1, padding=1),              # (64, 32, 128, 128)
            nn.BatchNorm2d(32, 0.8),                                # (64, 32, 128, 128)
            nn.LeakyReLU(0.2, inplace=True),                        # (64, 32, 128, 128)
            nn.Conv2d(32, channels, 3, stride=1, padding=1),        # (64, 3, 128, 128)
            nn.Tanh(),                                              # (64, 3, 128, 128)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, noise, labels):                               # (64, 100), (4, 64)
        hair = self.hair_emb(labels[0])                             # (64, 100)
        eyes = self.eyes_emb(labels[1])
        face = self.face_emb(labels[2])
        glasses = self.glasses_emb(labels[3])
        gen_input = torch.cat((hair, eyes, face, glasses, noise), 1)            # (64, 500)

        out = self.l1(gen_input)                                                # (64, 64*32*32)
        out = out.view(out.shape[0], 64, self.init_size, self.init_size)        # (64, 64, 32, 32)
        img = self.conv_blocks(out)                                             # (64, 3, 128, 128)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(                   # (64, 3, 128, 128)
            *discriminator_block(channels, 16, bn=False),    # (64, 4, 64, 64)  Conv stride=2
            # *discriminator_block(4, 8),                     # (64, 8, 32, 32)
            # *discriminator_block(8, 16),                    # (64, 16, 16, 16)
            *discriminator_block(16, 32),                   # (64, 32, 8, 8)
            *discriminator_block(32, 64),                   # (64, 64, 4, 4)
            *discriminator_block(64, 128),                  # (64, 128, 2, 2)
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 15))

        self._initialize_weights()

    def forward(self, img):                         # (64, 1, 32, 32)
        out = self.conv_blocks(img)                 # (64, 128, 2, 2)
        out = out.view(out.shape[0], -1)            # (64, 128*2*2)
        validity = self.adv_layer(out)              # (64, 1)
        label = self.aux_layer(out)                 # (64, 15)

        return validity, label

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
