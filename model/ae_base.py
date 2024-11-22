import torch
import torch.nn as nn

class HSICMEncoder(nn.Module):
    def __init__(self):
        super(HSICMEncoder, self).__init__()
        
        # Conv2D Layer
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(7, 7), stride=(2, 2))
        
        # ResNet Blocks
        self.resnet_blocks = nn.Sequential(
            self._resnet_block(16, 16, kernel_size=(3, 3), stride=(2, 2)),
            self._resnet_block(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            self._resnet_block(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            self._resnet_block(64, 128, kernel_size=(3, 3), stride=(2, 2))
        )
        
        # MaxPooling Layer
        self.maxpool = nn.MaxPool2d(kernel_size=(10, 1), stride=(1, 1))
        
        # Flatten and BatchNorm Layer
        self.flatten_bn = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(2048)
        )
        
        # Dense Layers
        self.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.Linear(128, 16)
        )
        
    def _resnet_block(self, in_channels, out_channels, kernel_size, stride):
        """Helper function to create a ResNet block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv2d(x)               # Conv2D
        x = self.resnet_blocks(x)        # ResNet Blocks
        x = self.maxpool(x)              # MaxPooling
        x = self.flatten_bn(x)           # Flatten + BatchNorm
        x = self.fc(x)                   # Dense Layers
        return x


class HSICMDecoder(nn.Module):
    def __init__(self):
        super(HSICMDecoder, self).__init__()
        
        # Dense Layers (Reverse of Encoder's FC)
        self.fc = nn.Sequential(
            nn.Linear(16, 128),
            nn.Linear(128, 2048)
        )
        
        # Unflatten Layer (Reverse Flatten-BN)
        self.unflatten_bn = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Unflatten(dim=1, unflattened_size=(128, 2, 8))  # Assuming output spatial size (2, 8)
        )
        
        # Upsample (Reverse of MaxPooling)
        self.upsample = nn.Upsample(size=(10, 8))  # Matching the pooled size in encoder
        
        # ResNet Blocks (Reverse Order)
        self.resnet_blocks = nn.Sequential(
            self._resnet_block(128, 64, kernel_size=(3, 3), stride=(2, 2)),
            self._resnet_block(64, 32, kernel_size=(3, 3), stride=(2, 2)),
            self._resnet_block(32, 16, kernel_size=(3, 3), stride=(2, 2)),
            self._resnet_block(16, 16, kernel_size=(3, 3), stride=(2, 2))
        )
        
        # ConvTranspose2D (Reverse of Conv2D)
        self.deconv2d = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(7, 7), stride=(2, 2), output_padding=(1, 1))
        
    def _resnet_block(self, in_channels, out_channels, kernel_size, stride):
        """Helper function to create a ResNet block (same as encoder for symmetry)."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.fc(x)                   # Dense Layers
        x = self.unflatten_bn(x)         # Unflatten + BatchNorm
        x = self.upsample(x)             # Upsample (Reverse MaxPooling)
        x = self.resnet_blocks(x)        # ResNet Blocks
        x = self.deconv2d(x)             # ConvTranspose2D
        return x


# test:
# if __name__ == "__main__":
#    encoder = HSICMEncoder()
#    decoder = HSICMDecoder()

#    print(encoder)
#   print(decoder)

#    dummy_input = torch.randn(1, 1, 64, 64)  # Batch size = 1, 1 channel, 64x64 image
#    encoded = encoder(dummy_input)
#    print("Encoded shape:", encoded.shape)

#    decoded = decoder(encoded)
#    print("Decoded shape:", decoded.shape)
