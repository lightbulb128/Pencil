import torchvision
import torch.utils.data
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch

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
  elif name == "agnews":

    print("Loading agnews dataset")
    EMBEDDING_DIMS = 256
    TOKEN_COUNT = 64

    train_data_raw = torchtext.datasets.AG_NEWS(root="./data", split="train")
    test_data_raw = torchtext.datasets.AG_NEWS(root="./data", split="test")

    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_data_raw), specials=["<unk>"])
    default_unk_index = vocab["<unk>"]
    vocab.set_default_index(default_unk_index)

    # transform all data as indices, and then pad them to the same length as 64
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
    label_pipeline = lambda x: int(x) - 1

    # create a embedding into 256 dims
    embedding = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=EMBEDDING_DIMS)
    
    # initialize embedding weights with fixed random seed
    torch.manual_seed(42)
    embedding.weight.data.normal_(0, 1)

    embedding.eval()

    def encode(x):
      text = x[1]
      label = x[0]
      indices = torch.tensor(text_pipeline(text), dtype=torch.int64)

      if indices.shape[0] < TOKEN_COUNT:
        indices = torch.cat([indices, torch.zeros(TOKEN_COUNT - indices.shape[0], dtype=torch.int64)], dim=0)
      else:
        indices = indices[:TOKEN_COUNT]

      embedded = embedding(indices)
      embedded = embedded.transpose(0, 1)

      return (embedded.cpu().detach().numpy(), label_pipeline(label))

    print("Processing agnews train data")
    train_data_raw = list(map(encode, train_data_raw))
    print("Processing agnews test data")
    test_data_raw = list(map(encode, test_data_raw))

    train_data = []
    for t, i in enumerate(train_data_raw):
      if amount is not None and t > amount: break
      train_data.append((i[0], i[1]))
    test_data = []
    for t, i in enumerate(test_data_raw):
      if amount is not None and t > amount: break
      test_data.append((i[0], i[1]))
    print("Load agnews done")
    return train_data, test_data

  elif name == "cifar100":
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_data_raw = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_data_raw = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
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

if __name__ == "__main__":
  train_data, test_data = load_dataset("agnews")
  print(train_data[0][0].shape)
  print(train_data[1][0].shape)