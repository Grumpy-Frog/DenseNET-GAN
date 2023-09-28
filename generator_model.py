import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
from torchvision import models

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels + growth_rate)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(torch.cat([x, out], 1)))
        out = self.conv2(out)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn(x))
        x = self.conv(x)
        return self.pool(x)

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, growth_rate=32):
        super(Generator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Dense blocks
        self.dense_blocks = nn.ModuleList([
            DenseBlock(2 * growth_rate, growth_rate) for _ in range(num_blocks)
        ])

        # Transition blocks
        self.transition_blocks = nn.ModuleList([
            TransitionBlock(3 * growth_rate, 2 * growth_rate) for _ in range(num_blocks - 1)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * growth_rate, out_channels, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        

        out = enc_out
        for dense_block, transition_block in zip(self.dense_blocks, self.transition_blocks):
            out = dense_block(out)
            out = transition_block(out)

        out = self.decoder(out)
        return out






def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))

    # Create the DenseNet-based generator
    in_channels = img_channels  # Input image channels (e.g., RGB)
    out_channels = 3  # Output image channels (e.g., RGB)
    num_blocks = 6  # Number of dense blocks in the generator
    growth_rate = 32  # Adjust this if needed

    generator = Generator(in_channels, out_channels, num_blocks, growth_rate)

    # Forward pass through the generator
    generated_images = generator(x)

    print(generated_images.shape)



if __name__ == "__main__":
    test()
