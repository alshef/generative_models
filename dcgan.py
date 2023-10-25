import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, num_feature_maps: int = 64) -> None:
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,
                               num_feature_maps * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(num_feature_maps * 8,
                               num_feature_maps * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(num_feature_maps * 4,
                               num_feature_maps * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(num_feature_maps * 2,
                               num_feature_maps,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feature_maps),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(num_feature_maps,
                               3,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.net(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_feature_maps: int = 64) -> None:
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, num_feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feature_maps, num_feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 2),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feature_maps * 2, num_feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feature_maps * 4, num_feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feature_maps * 8),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feature_maps * 8, 1, kernel_size=4, stride=1, padding=0),

            nn.Flatten()
        )

    def forward(self, img):
        logits = self.net(img)
        return logits


class DCGAN(nn.Module):
    def __init__(self, ) -> None:
        super(DCGAN, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

    def generator_forward(self, z):
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits


if __name__ == "__main__":
    net = DCGAN()
    print(net)