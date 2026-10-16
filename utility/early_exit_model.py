import torch.nn as nn


class EarlyExitResNet1D(nn.Module):

    def __init__(self, backbone, num_classes=10):
        super().__init__()

        # --------------------------------------------------
        # Shared pruned ResNet backbone

        self.stem = backbone.stem
        self.blocks = backbone.blocks
        self.pool = backbone.pool

        # Actual feature size of the pruned model.
        # Read from the last block's layers, since backbone.feature_dim
        # is a stale attribute that pruning does not update.
        self.feature_dim = self.blocks[-1].bn2.num_features

        # --------------------------------------------------
        # Exit 1
        # After Stage 1 = blocks 0 + 1
        # --------------------------------------------------

        stage1_channels = self.blocks[1].bn2.num_features

        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(stage1_channels, num_classes)
        )

        # --------------------------------------------------
        # Exit 2
        # After Stage 2 = blocks 2 + 3
        # --------------------------------------------------

        stage2_channels = self.blocks[3].bn2.num_features

        self.exit2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(stage2_channels, num_classes)
        )

        # --------------------------------------------------
        # Exit 3
        # Final exit after the complete backbone
        # --------------------------------------------------

        self.exit3 = backbone.classifier

    def forward(self, x):

        # --------------------------------------------------
        # Stem
        # --------------------------------------------------

        x = self.stem(x)

        # --------------------------------------------------
        # Stage 1
        # --------------------------------------------------

        x = self.blocks[0](x)
        x = self.blocks[1](x)

        exit1 = self.exit1(x)

        # --------------------------------------------------
        # Stage 2
        # --------------------------------------------------

        x = self.blocks[2](x)
        x = self.blocks[3](x)

        exit2 = self.exit2(x)

        # --------------------------------------------------
        # Stage 3
        # --------------------------------------------------

        x = self.blocks[4](x)
        x = self.blocks[5](x)

        # --------------------------------------------------
        # Stage 4
        # --------------------------------------------------

        x = self.blocks[6](x)
        x = self.blocks[7](x)

        # --------------------------------------------------
        # Final exit
        # --------------------------------------------------

        x = self.pool(x)
        x = x.flatten(1)

        exit3 = self.exit3(x)

        return exit1, exit2, exit3