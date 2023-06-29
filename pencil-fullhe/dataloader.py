import torchvision
import torch.utils.data

def load_dataset(name, amount=None):
  name = name.lower()
  if name == "mnist":
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(0.5, 0.5)
    ])
    train_data_raw = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data_raw = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_data = []
    for t, i in enumerate(train_data_raw):
      if amount is not None and t > amount: break
      train_data.append((i[0], i[1]))
    test_data = []
    for t, i in enumerate(test_data_raw):
      if amount is not None and t > amount: break
      test_data.append((i[0], i[1]))
    return train_data, test_data # image shape: (1, 28, 28)
  elif name == "cifar10-224":
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
      torchvision.transforms.Resize((224, 224))
    ])
    train_data_raw = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data_raw = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_data = []
    for t, i in enumerate(train_data_raw):
      if amount is not None and t > amount: break
      train_data.append((i[0], i[1]))
    test_data = []
    for t, i in enumerate(test_data_raw):
      if amount is not None and t > amount: break
      test_data.append((i[0], i[1]))
    return train_data, test_data # image shape: (3, 224, 224)
  elif name == "cifar10-32":
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_data_raw = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data_raw = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_data = []
    for t, i in enumerate(train_data_raw):
      if amount is not None and t > amount: break
      train_data.append((i[0], i[1]))
    test_data = []
    for t, i in enumerate(test_data_raw):
      if amount is not None and t > amount: break
      test_data.append((i[0], i[1]))
    return train_data, test_data # image shape: (3, 32, 32)
  else:
    raise Exception("Unknown dataset")

class BatchLoader:
  
  def __init__(self, dataset, batchsize, shuffle):
    self.raw = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, drop_last=True)
    self.iterator = iter(self.raw)
  
  def get(self):
    try:
      result = next(self.iterator)
      return result
    except StopIteration:
      self.iterator = iter(self.raw)
      result = next(self.iterator)
      return result

  def __len__(self):
    return len(self.raw)
