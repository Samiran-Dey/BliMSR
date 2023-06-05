import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class ResBlock(nn.Module):
    """Defines the ResBlock with spectral normalization (SN)
    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Noumber of output channels.
        stride (int): Stride for the first convolution. Default: 2.
    """
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        norm=spectral_norm
        self.conv1 = norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1))
        self.Lrelu = nn.LeakyReLU()
        self.conv2 = norm(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.stride = stride

    def forward(self, x):
        identity = torch.clone(x)

        out = self.conv1(x)
        out = self.Lrelu(out)
        out = self.conv2(out)

        if out.shape != identity.shape:
          identity = F.interpolate(x, scale_factor=0.5, mode='bicubic', align_corners=False)
          identity = torch.cat([identity,identity], dim=1)
        out += identity
        
        return out


class DiscriminatorSN(nn.Module):
    """Defines a ResNet based discriminator with spectral normalization (SN)
    Args:
        num_in_ch (int): Channel number for inputs. Default: 1.
        num_feat (int): Channel number for base intermediate features. Default: 64.
    """

    def __init__(self, num_in_ch=1, num_feat=64):
      super(DiscriminatorSN, self).__init__()

      # the first convolution
      self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
      self.block1 = ResBlock(64,64,stride=1)
      self.block2 = ResBlock(64,128)
      self.block3 = ResBlock(128,128,stride=1)
      self.block4 = ResBlock(128,256)
      self.block5 = ResBlock(256,256,stride=1)
      self.block6 = ResBlock(256,512)
      self.block7 = ResBlock(512,512,stride=1)        
      self.flatten = nn.Flatten()
      self.fc = nn.Linear(2097152,1)
      self.LreLU = nn.LeakyReLU()


    def forward(self,x):      
      x0=self.conv0(x)
      x1=self.LreLU(x0)
      x1=self.block1(x1)
      x2=self.LreLU(x1)
      x2=self.block2(x2)
      x3=self.LreLU(x2)
      x3=self.block3(x3)
      x4=self.LreLU(x3)
      x4=self.block4(x4)
      x5=self.LreLU(x4)
      x5=self.block5(x5)
      x6=self.LreLU(x5)
      x6=self.block6(x6)
      x7=self.LreLU(x6)
      x7=self.block7(x7)
      x8=self.LreLU(x7)
      x=self.flatten(x8)
      x=self.fc(x)
      return x


