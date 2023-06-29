import torch
import torch.nn
import torchvision
import math

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

class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, qkv):
        embed_dim = qkv.shape[-1] // 3
        q = qkv[:, :, :embed_dim]
        k = qkv[:, :, embed_dim:2*embed_dim]
        v = qkv[:, :, 2*embed_dim:]
        q = q / math.sqrt(q.shape[-1])
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = torch.nn.functional.softmax(qk, dim=-1)
        qk = self.dropout_layer(qk)
        qkv = torch.matmul(qk, v)
        return qkv
    
# The original torch.nn.MultiheadAttention require input to have q, k, v 
# and they are projected respectively.
# This edited version accepts x and make q=k=v=x as input
class MultiheadAttention(torch.nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.inner = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        ret = self.inner(x, x, x)[0]

        # print("--- torch ---")
        # batchsize = x.shape[1]
        # seqlen = x.shape[0]

        # in_linear = torch.nn.Linear(self.embed_dim, 3 * self.embed_dim)
        # in_linear.weight = self.inner.in_proj_weight
        # in_linear.bias = self.inner.in_proj_bias
        # qkv = in_linear(x)
        # print("qkv")
        # print(qkv)
        # q = qkv[:, :, :self.embed_dim]
        # k = qkv[:, :, self.embed_dim:2*self.embed_dim]
        # v = qkv[:, :, 2*self.embed_dim:]
        # head_dimension = self.embed_dim // self.num_heads
        # q = q.contiguous().view(seqlen, batchsize * self.num_heads, head_dimension).transpose(0, 1)
        # k = k.contiguous().view(seqlen, batchsize * self.num_heads, head_dimension).transpose(0, 1)
        # v = v.contiguous().view(seqlen, batchsize * self.num_heads, head_dimension).transpose(0, 1)
        # # concatenate qkv
        # qkv = torch.cat([q, k, v], dim=-1)
        # print("qkv concat")
        # print(qkv)

        # sdp = ScaledDotProductAttention(self.dropout)
        # y_sdp = sdp(qkv)
        # print("y_sdp")
        # print(y_sdp)
        
        # scale = math.sqrt(q.shape[-1])
        # q = q / scale
        # p = torch.matmul(q, k.transpose(-2, -1))
        # a = torch.nn.functional.softmax(p, dim=-1)
        # a = torch.nn.functional.dropout(a, p=self.dropout, training=self.training)
        # y = torch.matmul(a, v)
        # print("y before merge")
        # print(y)

        return ret
    

class Residual(torch.nn.Module):
    
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)
    
class Truncate(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class TorchNative(torch.nn.Module):
    
    def __init__(self, preset):
        super().__init__()
        self.preset = preset
        self.model = load_model(self.preset)
    
    def forward(self, x):
        return self.model(x)