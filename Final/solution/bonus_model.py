"""Define your architecture here."""
import torch
from models import SimpleNet


class OptimizedSimpleNet(torch.nn.Module):
    def __init__(self, num_classes=2, in_chans=3, dropout_rate=0.3):
        super(OptimizedSimpleNet, self).__init__()

        base_filters = 8

        # Feature extractor
        self.features = torch.nn.Sequential(
            self._conv_block(in_chans, base_filters, kernel_size=3, stride=1, padding=1),  # Initial conv block
            self._conv_block(base_filters, base_filters, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample

            self._conv_block(base_filters, base_filters * 2, kernel_size=3, stride=1, padding=1),
            self._conv_block(base_filters * 2, base_filters * 2, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample

            self._conv_block(base_filters * 2, base_filters * 4, kernel_size=3, stride=1, padding=1),
            self._conv_block(base_filters * 4, base_filters * 4, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample

            self._conv_block(base_filters * 4, base_filters * 8, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Final downsample
        )

        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(int(64), 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, num_classes)
        )

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return torch.nn.Sequential(
            torch. nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = self.classifier(x)
        return x

def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """

    # initialize your model:
    model = OptimizedSimpleNet()

    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model
