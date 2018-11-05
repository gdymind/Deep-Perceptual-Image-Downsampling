# full assembly of the sub-parts to form the complete net

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, out_channels)

    def forward(self, x):
        # b c w w
        x1 = self.inc(x) # b 64 w w
        x2 = self.down1(x1) # b 128 1/2w 1/2w
        x3 = self.down2(x2) # b 256 1/4w 1/4w
        x4 = self.down3(x3) # b 512 1/8w 1/8w
        x5 = self.down4(x4) # b 512 1/16w 1/16w
        x = self.up1(x5, x4) # b 256 1/8w 1/8w
        x = self.up2(x, x3) # b 128 1/4w 1/4w
        x = self.up3(x, x2) # b 64 1/2w 1/2w
        x = self.up4(x, x1) # b 64 w w
        x = self.outc(x)
        return x
