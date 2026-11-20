import torch
import torch.nn as nn

x = torch.randn(8, 64, 16, 16)

upsample = nn.Upsample(
    scale_factor=2,
    mode="nearest"
)

output = upsample(x)

print("Before:", x.shape)
print("After:", output.shape)


upsampled = torch.randn(8, 64, 32, 32)
encoder_features = torch.randn(8, 32, 32, 32)

combined = torch.cat(
    [upsampled, encoder_features],
    dim=1
)

print(combined.shape)


print("*"*50)

class MiniUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU()
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1),
            nn.ReLU()
        )

        # Final output
        self.output = nn.Conv2d(16, 3, 1)

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        print("Encoder 1:", e1.shape)

        e2 = self.enc2(self.pool(e1))
        print("Encoder 2:", e2.shape)

        # Bottleneck
        b = self.bottleneck(self.pool(e2))
        print("Bottleneck:", b.shape)

        # Decoder 2
        d2 = self.up(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        print("Decoder 2:", d2.shape)

        # Decoder 1
        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        print("Decoder 1:", d1.shape)

        # Output
        out = self.output(d1)
        return out


model = MiniUNet()

x = torch.randn(8, 3, 64, 64)

result = model(x)

print("Final output:", result.shape)
print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters())
)



