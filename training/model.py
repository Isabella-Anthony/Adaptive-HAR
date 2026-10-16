import torch              
import torch.nn as nn     


# ---------------------------------------------------------------------------
# HELPER: round a number of channels to a multiple of 8, never below 8.
# Small chips process numbers in groups of 8, so this keeps them efficient
# when width_mult shrinks the model. Example: 64 * 0.35 = 22.4 -> 24.
# ---------------------------------------------------------------------------
def _round8(x):
    return max(8, int(round(x / 8)) * 8)


# ---------------------------------------------------------------------------
# RESIDUAL BLOCK: two pattern-detector layers plus a shortcut.
#
#   main path : Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
#   shortcut  : usually just the input, unchanged.
#               But if the block changes the shape (fewer moments because of
#               stride 2, or a different number of channels), the input can't
#               be added directly. Then the shortcut uses a cheap 1x1 conv to
#               reshape it so the two match.
#   output    : ReLU( main path + shortcut )
#
#   Conv      : slides short learned kernels along time to detect patterns.
#   BatchNorm : keeps numbers in a sensible range so learning is stable.
#   ReLU      : turns negatives into 0 (the "bend" that lets layers learn
#               complex patterns).
# ---------------------------------------------------------------------------
class ResidualBlock1D(nn.Module):

    # __init__ runs ONCE, when the block is built. It creates the layers.
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=5):
        super().__init__()                  # required PyTorch setup line
        pad = kernel_size // 2              # keeps the length unchanged

        # --- main path -----------------------------------------------------
        # First layer. If stride=2 it also halves the length.
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        # Second layer. Always stride 1.
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1,
                               padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # --- shortcut ------------------------------------------------------
        if stride != 1 or in_ch != out_ch:
            # Shape changes, so reshape the input with a 1x1 conv.
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            # Shape is the same: pass the input through untouched.
            self.shortcut = nn.Identity()

    # forward runs EVERY TIME data passes through. x is the incoming data.
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))   # first layer
        out = self.bn2(self.conv2(out))            # second layer (no ReLU yet)
        out = out + self.shortcut(x)               # add the shortcut
        return self.relu(out)                      # ReLU after adding


# ---------------------------------------------------------------------------
# THE COMPLETE MODEL
# ---------------------------------------------------------------------------
class ResNet1D(nn.Module):

    def __init__(
        self,
        in_channels=6,                     # signals per moment
        num_classes=6,                     # activities to choose between
        width_mult=1.0,                    # SIZE KNOB: 1.0 = full, 0.25 = small
        stage_channels=(64, 128, 256, 256),# channels per stage at full size
        blocks_per_stage=(2, 2, 2, 2),     # residual blocks in each stage
        kernel_size=5,                     # moments each detector looks at
        dropout=0.2,                       # fraction of features hidden in training
    ):
        super().__init__()                 # required PyTorch setup line

        # Scale every stage's channels by the size knob.
        chans = [_round8(c * width_mult) for c in stage_channels]

        # --- Stem: first layer. Turns the raw signals into learned channels
        # and halves the length (128 -> 64). Uses a wider kernel (7) so the
        # very first layer already sees a decent stretch of time.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, chans[0], kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm1d(chans[0]),
            nn.ReLU(inplace=True),
        )

        # --- Stages of residual blocks. -----------------------------------
        # The first stage keeps the length; every later stage halves it in
        # its first block (stride 2). Later blocks in a stage use stride 1.
        blocks = []
        prev = chans[0]
        for stage, (c, n) in enumerate(zip(chans, blocks_per_stage)):
            for i in range(n):
                stride = 2 if (stage > 0 and i == 0) else 1
                blocks.append(ResidualBlock1D(prev, c, stride, kernel_size))
                prev = c
        self.blocks = nn.Sequential(*blocks)

        # --- Pool: average each channel over the remaining moments. ---------
        # "Adaptive" = works for any recording length, not only 128.
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Size of the summary. Handy when adding a new classifier for
        # adaptation.
        self.feature_dim = prev

        # --- Classifier ---------------------------------------------------
        self.classifier = nn.Sequential(
            # Training only: randomly hide some summary numbers so the model
            # can't over-rely on any single one (reduces overfitting).
            nn.Dropout(dropout),
            # Turn the summary into one score per activity.
            nn.Linear(prev, num_classes),
        )

        self._init_weights()

    # -----------------------------------------------------------------------
    # Starting values for the weights, before any learning happens.
    # -----------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Random values scaled to suit ReLU networks.
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                # Start as "do nothing": multiply by 1, add 0.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Common ResNet trick: start each block's last BatchNorm at 0, so every
        # block begins as "output = input" (only the shortcut). The blocks
        # then gradually learn their corrections. Makes training smoother.
        for m in self.modules():
            if isinstance(m, ResidualBlock1D):
                nn.init.zeros_(m.bn2.weight)

    # -----------------------------------------------------------------------
    # Get the SUMMARY of each recording, without the final scores.
    # Used for adaptation: keep this fixed and retrain only the classifier,
    # or compare summaries between people.
    # -----------------------------------------------------------------------
    def forward_features(self, x):
        x = self.stem(x)          # first layer
        x = self.blocks(x)        # all residual blocks
        x = self.pool(x)          # average over time -> (batch, channels, 1)
        return x.flatten(1)       # drop the leftover 1 -> (batch, channels)

    # -----------------------------------------------------------------------
    # The normal path: recording in, scores out. model(x) calls this.
    # -----------------------------------------------------------------------
    def forward(self, x):
        return self.classifier(self.forward_features(x))


# ---------------------------------------------------------------------------
# Helper: how big is the model?
#   weights    : how many learnable numbers it has
#   float32_KB : storage as normal 32-bit numbers (4 bytes each)
#   int8_KB    : storage after 8-bit quantization (1 byte each)
# ---------------------------------------------------------------------------
def model_size(model):
    n = sum(p.numel() for p in model.parameters())
    return {
        "weights": n,
        "float32_KB": round(n * 4 / 1024, 1),
        "int8_KB": round(n / 1024, 1),
    }

