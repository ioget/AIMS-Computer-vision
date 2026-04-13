import torch.nn as nn


class CNN1(nn.Module):
    """
    PyTorch CNN for Intel Image Classification.
    Input : (B, 3, 150, 150)  – RGB images
    Output: (B, 6)            – logits for 6 scene classes
    Architecture
    ────────────
    4 convolutional blocks (Conv → BN → ReLU → MaxPool):
        32 → 64 → 128 → 256 filters
    AdaptiveAvgPool2d(4,4) to decouple from input resolution
    Classifier: Dropout → FC(4096→512) → ReLU → Dropout → FC(512→6)
    """

    def __init__(self, num_classes: int = 6):
        super(CNN1, self).__init__()

        self.features = nn.Sequential(
            # Block 1  150×150 → 75×75
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2  75×75 → 37×37
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3  37×37 → 18×18
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4  18×18 → 9×9
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Fix spatial dims to 4×4 regardless of input resolution
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
