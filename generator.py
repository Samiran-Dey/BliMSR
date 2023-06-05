import torch.nn as nn
import torch
import torchvision.models as models

class MHCA(nn.Module):
    def __init__(self, n_feats, ratio):
        """
        MHCA spatial-channel attention module.
        Args:
         n_feats: The number of filter of the input.
         ratio: Channel reduction ratio.
        """
        super(MHCA, self).__init__()

        out_channels = int(n_feats // ratio)

        head_1 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=n_feats, kernel_size=1, padding=0, bias=True)
        ]

        kernel_size_sam = 3
        head_2 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam, padding=0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam, padding=0, bias=True)
        ]

        kernel_size_sam_2 = 5
        head_3 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam_2, padding=0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam_2, padding=0, bias=True)
        ]

        self.head_1 = nn.Sequential(*head_1)
        self.head_2 = nn.Sequential(*head_2)
        self.head_3 = nn.Sequential(*head_3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res_h1 = self.head_1(x)
        res_h2 = self.head_2(x)
        res_h3 = self.head_3(x)
        m_c = self.sigmoid(res_h1 + res_h2 + res_h3)
        res = x * m_c
        return res

class RMHAB(nn.Module):
    """Defines the Residual Multi-Head Attention Block
      Args:
        in_chhanels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
        )
        self.layer2 = MHCA(n_feats=in_channels,ratio=4)
        

    def forward(self, x):

        x_ = self.layer1(x)
        x__ = self.layer2(x_)

        x = x__ + x

        return x


class ShortResidualBlock(nn.Module):
    """Defines the Blocks with Short Residual Connections
      Args:
        in_chhanels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super().__init__()

        self.layers = nn.ModuleList([RMHAB(in_channels) for _ in range(16)])

    def forward(self, x):

        x_ = x.clone()

        for layer in self.layers:
            x_ = layer(x_)

        return x_ + x


class Generator(nn.Module):
    """Defines BliMSR generator architecture
      Args:
        in_chhanels (int): Number of input channels. Default: 1.
        blocks (int): Number of blocks with short residual connections. Default: 8.
    """
    def __init__(self, in_channels=1, blocks=8):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1)

        self.short_blocks = nn.ModuleList(
            [ShortResidualBlock(64) for _ in range(blocks)]
        )

        self.conv2 = nn.Conv2d(64, 64, (1, 1), stride=1, padding=0)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),  # Remove if output is 2x the input
            nn.Conv2d(64, in_channels, (1, 1), stride=1, padding=0),  # Change 64 -> 256
            nn.Sigmoid(),
        )
        

    def forward(self, x):

        x = self.conv(x)
        x_ = x.clone()

        for layer in self.short_blocks:
            x_ = layer(x_)

        x = torch.cat([self.conv2(x_), x], dim=1)
        x = self.conv3(x)
        return x
