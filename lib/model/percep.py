import torchvision
from torch import nn
from torchvision.models import mobilenet_v3_small


class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        feature_layer = 34
        vgg = torchvision.models.vgg19(pretrained=True)
        vgg_layers = list(vgg.features[: feature_layer + 1])
        self.features = nn.Sequential(
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
            *vgg_layers,
        )
        self.requires_grad_(False)

    def forward(self, x):
        return self.features(x)


class VGGPercepLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extract = VGGFeatures()

    def forward(self, x, y):
        x = self.feature_extract(x)
        y = self.feature_extract(y.detach())
        return (x - y).abs().flatten(1, -1).mean(dim=1)
    


class MobilePercepLoss(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(pretrained=True)
        self.features = backbone.features[:10]  # 取中间层
        self.normalize = torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self.requires_grad_(False)
    
    def forward(self, x, y):
        x = self.features(self.normalize(x))
        y = self.features(self.normalize(y))
        z = (x - y).abs().flatten(1, -1).mean(dim=1)
        return z
