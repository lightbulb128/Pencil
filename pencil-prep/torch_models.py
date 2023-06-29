import torch
import torch.nn
import torchvision

class AlexNetFeatureExtractor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fe = torchvision.models.alexnet(True).features
        self.pool = torch.nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        y = self.pool(self.fe(x))
        return y

class ResNet50FeatureExtractor(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(True)
        self.fe = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )

    def forward(self, x):
        y = torch.flatten(self.fe(x), 1)
        return y

class DenseNet121FeatureExtractor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        model = torchvision.models.densenet121(True)
        self.fe = model.features

    def forward(self, x):
        y = torch.flatten(self.fe(x), 1)[:, :4096] # 50176 dims total. 
        return y


def load_model(preset: str):
    ret = None
    if preset == "alexnet-fe":
        ret = AlexNetFeatureExtractor()
    if preset == "resnet50-fe":
        return ResNet50FeatureExtractor()
    if preset == "densenet121-fe":
        return DenseNet121FeatureExtractor()
    ret.eval()
    return ret

class TorchNative(torch.nn.Module):
    
    def __init__(self, preset):
        super().__init__()
        self.preset = preset
        self.model = load_model(self.preset)
    
    def forward(self, x):
        return self.model(x)