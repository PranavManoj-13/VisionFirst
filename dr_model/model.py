import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
ARCHITECTURE = 'mobilenet_v3_large_ordered_v2'


class OrderedHead(nn.Module):
    """Shared severity score with strictly increasing cut points.

    Unlike independent binary heads, cumulative probabilities cannot cross.
    """
    def __init__(self, features):
        super().__init__()
        self.score = nn.Linear(features, 1, bias=False)
        self.first_cut = nn.Parameter(torch.tensor(-1.5))
        self.raw_gaps = nn.Parameter(torch.full((3,), 0.54132485))

    def forward(self, features):
        gaps = F.softplus(self.raw_gaps) + 1e-4
        cuts = torch.cat([self.first_cut.reshape(1), self.first_cut + torch.cumsum(gaps, 0)])
        return self.score(features) - cuts


class RetinalModel(nn.Module):
    """Input: RGB float32 [N,H,W,3] in [0,255]; output: four ordinal logits."""
    def __init__(self, pretrained=False):
        super().__init__()
        net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(net.classifier[0], net.classifier[1], nn.Dropout(0.2))
        self.head = OrderedHead(1280)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def forward(self, rgb):
        # NHWC -> NCHW permute is a strided view. Materialize standard NCHW
        # storage before convolutions to avoid MPS channels-last backward errors.
        x = rgb.permute(0, 3, 1, 2).contiguous() / 255.0
        x = (x - self.mean) / self.std
        return self.head(self.projection(self.pool(self.features(x)).flatten(1)))


class InferenceModel(nn.Module):
    """Return cumulative probabilities, preserving uncertainty for the app."""
    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.model = model
        self.register_buffer('temperature', torch.tensor(float(temperature)))

    def forward(self, rgb):
        return torch.sigmoid(self.model(rgb) / self.temperature)


def ordinal_loss(logits, labels, smoothing=0.0):
    targets = (labels[:, None] > torch.arange(4, device=logits.device)).float()
    targets = targets * (1 - smoothing) + smoothing * 0.5
    return F.binary_cross_entropy_with_logits(logits, targets)


def class_probabilities(cumulative):
    return torch.cat([1 - cumulative[:, :1], cumulative[:, :-1] - cumulative[:, 1:], cumulative[:, -1:]], dim=1)
