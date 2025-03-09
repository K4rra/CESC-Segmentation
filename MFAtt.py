class MFAtt(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(CSAtt, self).__init__()

        self.channel_attention = nn.Sequential(  # FC->act->FC
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(  # Conv->BN->act->Conv->BN
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

        self.frequency_attention = FcaLayer(in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x_branch = self.frequency_attention(x)
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        out = self.frequency_attention(out)
        out = torch.add(out, x_branch)