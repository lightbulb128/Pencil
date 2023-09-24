import torch
from torch import nn
import torchvision
import torch_models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_alexnet(dropout=0.5, num_classes=10):
    return torch.nn.Sequential(
        # input (b, 3, 224, 224)
        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # mid (b, 128, 6, 6)
        nn.Flatten(),
        nn.Dropout(p=dropout),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )

def get_model_alexnet_classifier(dropout=0.5, num_classes=10):
    fe = torch_models.TorchNative("alexnet-fe")
    fe.requires_grad_(False)
    return torch.nn.Sequential(
        fe,
        nn.Flatten(),
        # nn.Dropout(p=dropout),
        nn.Linear(256 * 6 * 6, 512),
        nn.ReLU(inplace=True),
        # nn.Dropout(p=dropout),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes),
    )

def get_model_resnet50_classifier(dropout=0.5, num_classes=10):
    fe = torch_models.TorchNative("resnet50-fe")
    fe.requires_grad_(False)
    return torch.nn.Sequential(
        fe,
        nn.Flatten(),
        # nn.Dropout(p=dropout),
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        # nn.Dropout(p=dropout),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes),
    )

def get_model_resnet50_cifar100_classifier(dropout=0.5, num_classes=100):
    fe = torch_models.TorchNative("resnet50-fe")
    fe.requires_grad_(False)
    return torch.nn.Sequential(
        fe,
        nn.Flatten(),
        # nn.Dropout(p=dropout),
        nn.Linear(2048, num_classes),
    )

def get_model_densenet121_classifier(dropout=0.5, num_classes=10):
    fe = torch_models.TorchNative("densenet121-fe")
    fe.requires_grad_(False)
    return torch.nn.Sequential(
        fe,
        nn.Flatten(),
        # nn.Dropout(p=dropout),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        # nn.Dropout(p=dropout),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes),
    )

def get_model_mnist_aby3():
    return torch.nn.Sequential( # input (1, 28, 28)
        torch.nn.Flatten(),
        torch.nn.Linear(1*28*28, 128), # 100
        torch.nn.ReLU(), 
        torch.nn.Linear(128, 128), # 100
        torch.nn.ReLU(), 
        torch.nn.Linear(128, 10), # 10
    )

def get_model_mnist_chameleon():
    return torch.nn.Sequential( # input (1, 28, 28)
        torch.nn.Conv2d(1, 5, 5, stride=2, padding=2), # (5, 14, 14)
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(980, 100), # 100
        torch.nn.ReLU(), 
        torch.nn.Linear(100, 10), # 10
    )

def get_model_mnist_sphinx():
    return torch.nn.Sequential( # input (1, 28, 28)
        torch.nn.Conv2d(1, 16, 5), # (16, 24, 24)
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(2), # (16, 12, 12)
        torch.nn.Conv2d(16, 16, 5), # (16, 8, 8)
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(2), # 16 * 4 * 4
        torch.nn.Flatten(),
        torch.nn.Linear(256, 100), # 100
        torch.nn.ReLU(), 
        torch.nn.Linear(100, 10), # 10
    )

def get_model_cifar10_lenet5():
    return torch.nn.Sequential( # input (3, 32, 32)
        torch.nn.Conv2d(3, 6, 5), # (6, 28, 28)
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(2, 2), # (6, 14, 14)
        torch.nn.Conv2d(6, 16, 5), # (16, 10, 10)
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(2, 2), # (16, 5, 5)
        torch.nn.Conv2d(16, 120, 5), # (120, 1, 1)
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(120, 84), # 100
        torch.nn.ReLU(), 
        torch.nn.Linear(84, 10), # 10
    )

def get_model_cifar10_sphinx():
    return torch.nn.Sequential( # (3, 32, 32)
        torch.nn.Conv2d(3, 64, 5, padding=2), # (64, 32, 32)
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(2, 2), # (64, 16, 16)
        torch.nn.Conv2d(64, 64, 5, padding=2), # (64, 16, 16)
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(2, 2), # (64, 8, 8)
        torch.nn.Conv2d(64, 64, 3, padding=1), # (64, 8, 8)
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 1), # (64, 8, 8)
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 16, 1), # (16, 8, 8)
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(1024, 10)
    )

def get_model_mnist_quotient_2x128():
    return get_model_mnist_aby3()

def get_model_mnist_quotient_3x128():
    return torch.nn.Sequential( # input (1, 28, 28)
        torch.nn.Flatten(),
        torch.nn.Linear(1*28*28, 128), # 100
        torch.nn.ReLU(), 
        torch.nn.Linear(128, 128), # 100
        torch.nn.ReLU(), 
        torch.nn.Linear(128, 128), # 100
        torch.nn.ReLU(), 
        torch.nn.Linear(128, 10), # 10
    )

def get_model_mnist_quotient_2x512():
    return torch.nn.Sequential( # input (1, 28, 28)
        torch.nn.Flatten(),
        torch.nn.Linear(1*28*28, 512), # 100
        torch.nn.ReLU(), 
        torch.nn.Linear(512, 512), # 100
        torch.nn.ReLU(), 
        torch.nn.Linear(512, 10), # 100
    )

def get_model_agnews_mlp():
    return torch.nn.Sequential( # input (256, 64)
        torch.nn.Flatten(),
        torch.nn.Linear(16384, 128),
        torch.nn.ReLU(), 
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(), 
        torch.nn.Linear(128, 4),
    )

def get_model_agnews_cnn():
    return torch.nn.Sequential( # input (256, 64)
        torch.nn.Conv1d(256, 128, 5), # (128, 60)
        torch.nn.ReLU(), 
        torch.nn.AvgPool1d(2), # (128, 30)
        torch.nn.Conv1d(128, 128, 5), # (128, 26)
        torch.nn.ReLU(), 
        torch.nn.AvgPool1d(2), # (128, 13)
        torch.nn.Flatten(),
        torch.nn.Linear(128 * 13, 4),
    )

def get_model(name = "cifar10_lenet5"):
    model_funcs = {
        "mnist_aby3": get_model_mnist_aby3,
        "mnist_chameleon": get_model_mnist_chameleon,
        "mnist_sphinx": get_model_mnist_sphinx,
        "mnist_quotient_3x128": get_model_mnist_quotient_3x128,
        "mnist_quotient_2x512": get_model_mnist_quotient_2x512,
        "cifar10_lenet5": get_model_cifar10_lenet5,
        "cifar10_sphinx": get_model_cifar10_sphinx,
        "alexnet": get_model_alexnet,
        "alexnet_classifier": get_model_alexnet_classifier,
        "resnet50_classifier": get_model_resnet50_classifier,
        "densenet121_classifier": get_model_densenet121_classifier, 
        "agnews_mlp": get_model_agnews_mlp,
        "agnews_cnn": get_model_agnews_cnn,
        "resnet50_cifar100_classifier": get_model_resnet50_cifar100_classifier,
    }
    return name, model_funcs[name]()

def get_dataset(x):
    if "cifar100" in x: return "cifar100", (64, 3, 32, 32)
    if x == "alexnet": return "cifar10-224", (1, 3, 224, 224)
    if "agnews" in x: return "agnews", (32, 256, 64)
    if "classifier" in x: return "cifar10-224", (64, 3, 224, 224)
    if "cifar10" in x: return "cifar10-32", (64, 3, 32, 32)
    if "mnist" in x: return "mnist", (32, 1, 28, 28)
    return "mnist", (32, 1, 28, 28)
    raise Exception()