import numpy as np
from crypto_switch import cryptography as crypto
import config
import communication_client
import sys
import hashlib
import os
import pickle
import math
import torch_models
import torch


def ceil_div(a, b):
  if a%b==0: return a//b
  return a//b+1

def random_tensor(shape, bound=config.RANDOM_TENSOR_BOUND):
  return np.random.uniform(-bound, bound, shape)
def random_scalar(bound = 1) -> float:
  return np.random.uniform(-bound, bound)


class ClientModule:

  # Common utilities
  def send_flatten_encrypt(self, x, double_scale=False):
    self.comm.send(self.crypto.serialize(self.crypto.flat_encrypt(x.flatten(), double_scale)))
  def recv_squeeze_decrypt(self, shape):
    x = self.crypto.flat_decrypt(self.crypto.deserialize(self.comm.recv()))
    return np.reshape(x, shape)
  def send_enc_scalar_lists(self, ks, inverse, layer=-2):
    if not inverse:
      encs = [self.crypto.encrypt_scalar(k, layer=layer) for k in ks]
    else:
      encs = [self.crypto.encrypt_scalar(1/k, layer=layer) for k in ks]
    encs = self.crypto.serialize(encs)
    self.send(encs)
  def send(self, x):
    self.comm.send(x)
  def recv(self):
    return self.comm.recv()

  def forward(self, x): pass
  def backward(self, partial_y): pass
  def prepare(self, input_shape): 
    raise Exception("Prepare in base class should never be called")

class ReLU(ClientModule):

  def __init__(self, crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication):
    self.comm = comm
    self.crypto = crypto
    self.static_forward = self.forward
    self.static_prepare = self.prepare

  def forward(self, x):
    self.x = x
    x, self.d = self.comm.relu(x)
    return x
  
  def backward(self, partial_y):
    partial_x = self.comm.drelumul(partial_y, self.d)
    return partial_x

  def prepare(self, input_shape): return input_shape


class GELU(ClientModule):

  def __init__(self, crypto: crypto.EncryptionUtils, communication: communication_client.ClientCommunication):
    super().__init__()
    self.comm = communication
    self.crypto = crypto

  def prepare(self, input_shape): return input_shape

  def forward(self, x):
    x = self.comm.truncate(x)
    x2 = self.comm.elementwise_multiply(x, x)
    x2 = self.comm.truncate(x2)
    x3 = self.comm.elementwise_multiply(x2, x)
    x3 = self.comm.truncate(x3)
    const1 = self.crypto.to_field(np.array(0.044715))
    x3c = self.crypto.field_mod(const1 * x3)
    x3c = self.comm.truncate(x3c)
    x3ca = self.crypto.field_add(x3c, x)
    const2 = self.crypto.to_field(np.array(0.7978845608028654))
    z = self.crypto.field_mod(const2 * x3ca)
    z = self.comm.truncate(z)
    z = self.comm.tanh(z)
    # Server add 1, client add 0.
    y = z
    const4 = self.crypto.to_field(np.array(0.5))
    y = self.crypto.field_mod(const4 * y)
    y = self.comm.truncate(y)
    y = self.comm.elementwise_multiply(x, y)
    y = self.comm.truncate(y)
    return y
  
  def backward(self, partial_y):
    assert False, "not implemented"

class Flatten(ClientModule):

  def __init__(self):
    self.static_forward = self.forward
    self.static_prepare = self.prepare

  def forward(self, x: np.ndarray):
    return x

  def backward(self, partial_y):
    return partial_y
      
  def prepare(self, input_shape): 
    return (input_shape[0], np.product(input_shape[1:]))

class Linear(ClientModule):

  def __init__(self, crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication, input_dims, output_dims):
    self.comm = comm
    self.crypto = crypto
    self.prepared = False
    self.input_dims = input_dims
    self.output_dims = output_dims

  def collaborate_matmul(self, helper, x_client):
    x_client_cipher = self.crypto.encrypt_cipher2d(self.crypto.matmul_encode_x(helper, x_client))
    self.send(self.crypto.serialize(x_client_cipher))
    y_client = self.crypto.matmul_decrypt_y(helper, self.crypto.matmul_deserialize_y(helper, self.recv()))
    return y_client

  def forward(self, x):
    self.x_client = x
    y = self.collaborate_matmul(self.helper_forward, x)
    return y

  def prepare(self, input_shape):
    self.prepared = True
    self.batchsize = input_shape[0]
    self.helper_forward = self.crypto.matmul_helper(input_shape[0], self.input_dims, self.output_dims, objective=0)
    self.helper_backward = self.crypto.matmul_helper(input_shape[0], self.output_dims, self.input_dims, objective=0)
    self.helper_weights = self.crypto.matmul_helper(self.output_dims, input_shape[0], self.input_dims, objective=2)
    return (input_shape[0], self.output_dims)

  def backward_calculate_partial_A(self, partial_y_client):
    # gW = (py^T) * x
    def linear_operation(y, x):
      x = np.reshape(x, (self.batchsize, self.input_dims))
      y = np.reshape(y, (self.output_dims, self.batchsize))
      return self.crypto.field_mod(np.matmul(y, x).flatten())

    helper = self.helper_weights
    
    partial_y_client = np.transpose(np.reshape(partial_y_client, (self.batchsize, self.output_dims)), (1, 0)).flatten()
    partial_y_client_encoded = self.crypto.matmul_encode_x(helper, partial_y_client)
    partial_y_client_cipher = self.crypto.encrypt_cipher2d(partial_y_client_encoded)
    self.send(self.crypto.serialize(partial_y_client_cipher))

    x_client_encoded = self.crypto.matmul_encode_w(helper, self.x_client)
    x_client_cipher = self.crypto.encrypt_cipher2d(x_client_encoded)
    self.send(self.crypto.serialize(x_client_cipher))

    partial_w = self.crypto.matmul_decrypt_y(helper, self.crypto.matmul_deserialize_y(helper, self.recv()))
    partial_w = self.crypto.field_add(partial_w, linear_operation(partial_y_client, self.x_client))
    self.send(partial_w)

  def backward(self, partial_y_client):

    # calculate partial_A
    self.backward_calculate_partial_A(partial_y_client)

    # to calculate partial_B
    self.comm.send(np.sum(np.reshape(partial_y_client, (self.batchsize, self.output_dims)), axis=0))
    
    # calculate partial_x shares
    partial_x = self.collaborate_matmul(self.helper_backward, partial_y_client)
    if config.TRUNCATE_BACKWARD:
      partial_x = self.comm.truncate(partial_x)
    else:
      print("WARNING: Backward gradient not truncated")
    return partial_x

  def export(self): pass

class Conv2d(ClientModule):
  def __init__(self, crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication, input_channels, output_channels, kernel_size, bias=True):
    self.comm = comm
    self.input_channels = input_channels
    self.output_channels = output_channels
    if isinstance(kernel_size, int):
      self.kernel_size = (kernel_size, kernel_size)
    else:
      self.kernel_size = kernel_size
    self.crypto = crypto
    self.prepared = False
    self.has_bias = bias

  def send_conv2d_encrypt(self, x, output_channels):
    self.comm.send(self.crypto.serialize(self.crypto.conv2d_encrypt(x, output_channels)))
  def recv_conv2d_decrypt(self):
    return self.crypto.conv2d_decrypt(self.crypto.deserialize(self.recv()))

  def collaborate_conv2d(self, helper, x_client, x_cipher_already_sent=False):
    if not x_cipher_already_sent:
      x_client_cipher = self.crypto.encrypt_cipher2d(self.crypto.conv2d_encode_x(helper, x_client))
      self.send(self.crypto.serialize(x_client_cipher))
    y_client = self.crypto.conv2d_decrypt_y(helper, self.crypto.conv2d_deserialize_y(helper, self.recv()))
    return y_client

  def collaborate_conv2d_accumulate_ciphers(self, helper, x_client):
    x_client_cipher = self.crypto.encrypt_cipher2d(self.crypto.conv2d_encode_x(helper, x_client))
    self.send(self.crypto.serialize(x_client_cipher))

  def forward(self, x):
    self.x_client = x
    y_client = self.collaborate_conv2d(self.helper_forward, x)
    return y_client

  def pad_size(self, x, size):
    pad_x = np.zeros((x.shape[0], x.shape[1], size[0], size[1]))
    pad_x[:x.shape[0], :x.shape[1], :x.shape[2], :x.shape[3]] = x
    return pad_x

  def forward_accumulate_ciphers(self, x):
    self.x_client = x
    self.collaborate_conv2d_accumulate_ciphers(self.helper_forward, x)

  def transpose4d(self, n):
    return np.transpose(n, (1,0,2,3))

  def conv2d_cut(self, x, k):
    return x[:, :, :k[0], :k[1]]

  def prepare(self, input_shape):
    self.input_shape = input_shape
    self.batchsize = input_shape[0]
    batchsize, channels, x_h, x_w = input_shape
    y_h = x_h - self.kernel_size[0] + 1
    y_w = x_w - self.kernel_size[1] + 1
    self.output_shape = (batchsize, self.output_channels, y_h, y_w)
    k_h = self.kernel_size[0]
    k_w = self.kernel_size[1]

    # y = Conv2d(x, w)
    self.helper_forward = self.crypto.conv2d_helper(
      self.batchsize, x_h, x_w, k_h, k_w, self.input_channels, self.output_channels, objective=0
    )
    # dx = Conv2d(Pad(dy), permute(flip(w, (2,3)), (1,0,2,3)))
    self.helper_backward = self.crypto.conv2d_helper(
      self.batchsize, x_h + k_h - 1, x_w + k_w - 1, k_h, k_w, self.output_channels, self.input_channels, objective=0
    )
    # dw = Conv2d^t(x^t, dy^t)
    self.helper_weights = self.crypto.conv2d_helper(
      self.input_channels,
      x_h, x_w, y_h, y_w, self.batchsize, self.output_channels, objective=2
    )

    self.prepared = True
    return (batchsize, self.output_channels, y_h, y_w)

  def conv2d_plain(self, x, weight):
    output_channels, _, k_h, k_w = weight.shape
    def to_tensor(x): 
      if x is None: return None
      return torch.tensor(x.astype(np.int64), dtype=torch.long)
    y = torch.conv2d(to_tensor(x), to_tensor(weight))
    return y.detach().cpu().numpy().astype(np.uint64)

  def backward_calculate_partial_A(self, partial_y_client, partial_y_already_sent=False):

    def linear_operation(x, y):
      return self.crypto.field_mod(self.conv2d_plain(x, y))

    helper = self.helper_weights
    
    partial_y_client = np.transpose(partial_y_client, (1, 0, 2, 3))
    if not partial_y_already_sent:
      partial_y_client_encoded = self.crypto.conv2d_encode_w(helper, partial_y_client.flatten())
      partial_y_client_cipher = self.crypto.encrypt_cipher2d(partial_y_client_encoded)
      self.send(self.crypto.serialize(partial_y_client_cipher))

    x_client = np.transpose(np.reshape(self.x_client, self.input_shape), (1, 0, 2, 3))
    x_client_encoded = self.crypto.conv2d_encode_x(helper, x_client.flatten())
    x_client_cipher = self.crypto.encrypt_cipher2d(x_client_encoded)
    self.send(self.crypto.serialize(x_client_cipher))

    partial_w = self.crypto.conv2d_decrypt_y(helper, self.crypto.conv2d_deserialize_y(helper, self.recv()))
    partial_w = np.reshape(partial_w, (self.input_channels, self.output_channels, self.kernel_size[0], self.kernel_size[1]))
    partial_w = self.crypto.field_add(partial_w, linear_operation(x_client, partial_y_client))
    self.send(partial_w)

  def backward(self, partial_y_client):

    batchsize, _, x_h, x_w = self.input_shape
    _, _, y_h, y_w = self.output_shape
    k_h, k_w = self.kernel_size
    partial_y_client = np.reshape(partial_y_client, self.output_shape)
    
    # calculate partial_A
    self.backward_calculate_partial_A(partial_y_client)

    # calculate partial_b
    if self.has_bias:
      partial_b = np.sum(np.sum(np.sum(partial_y_client, axis=0), axis=1), axis=1)
      self.comm.send(partial_b)

    # calculate partial_x
    p_h, p_w = k_h - 1, k_w - 1
    padded_partial_y = np.zeros((batchsize, self.output_channels, y_h + p_h*2, y_w + p_w*2), dtype=np.uint64)
    padded_partial_y[:,:,p_h:p_h+y_h,p_w:p_w+y_w] = partial_y_client
    padded_partial_y = padded_partial_y.flatten()
    partial_x_client = self.collaborate_conv2d(
      self.helper_backward,
      padded_partial_y
    )
    if config.TRUNCATE_BACKWARD:
      partial_x_client = self.comm.truncate(partial_x_client)
    else:
      print("WARNING: Backward gradient not truncated")
    return partial_x_client

  def backward_accumulate_ciphers(self, partial_y_client):

    batchsize, _, x_h, x_w = self.input_shape
    _, _, y_h, y_w = self.output_shape
    k_h, k_w = self.kernel_size
    partial_y_client = np.reshape(partial_y_client, self.output_shape)
    
    # calculate partial_A
    self.backward_calculate_partial_A(partial_y_client, True)

    # calculate partial_b
    if self.has_bias:
      partial_b = np.sum(np.sum(np.sum(partial_y_client, axis=0), axis=1), axis=1)
      self.comm.send(partial_b)

    # calculate partial_x
    p_h, p_w = k_h - 1, k_w - 1
    padded_partial_y = np.zeros((batchsize, self.output_channels, y_h + p_h*2, y_w + p_w*2), dtype=np.uint64)
    padded_partial_y[:,:,p_h:p_h+y_h,p_w:p_w+y_w] = partial_y_client
    padded_partial_y = padded_partial_y.flatten()
    partial_x_client = self.collaborate_conv2d(
      self.helper_backward,
      padded_partial_y,
      True
    )
    if config.TRUNCATE_BACKWARD:
      partial_x_client = self.comm.truncate(partial_x_client)
    else:
      print("WARNING: Backward gradient not truncated")
    return partial_x_client

  def export(self): pass

class Conv2dStrided(ClientModule):
  
  def __init__(self, 
    crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication,
    input_channels, output_channels, 
    kernel_size, stride, bias=True
  ):
    self.crypto = crypto
    self.comm = comm
    self.input_channels = input_channels
    self.output_channels = output_channels
    self.kernel_size = kernel_size
    self.stride = stride
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)
    self.kernel_size = kernel_size
    self.has_bias = bias
    self.create_convs()

  def create_convs(self):
    k_h, k_w = self.kernel_size
    d_h, d_w = ceil_div(k_h, self.stride), ceil_div(k_w, self.stride)
    self.sub_kernel_size = (d_h, d_w)
    convs = []
    for i in range(self.stride):
      convs_row = []
      for j in range(self.stride):
        has_bias = (i==0 and j==0 and self.has_bias)
        item = Conv2d(self.crypto, self.comm, self.input_channels, self.output_channels, self.sub_kernel_size, has_bias)
        convs_row.append(item)
      convs.append(convs_row)
    self.convs = convs

  def split(self, x):
    _, _, x_h, x_w = x.shape
    k_h, k_w = self.kernel_size
    x_h -= (x_h - k_h) % self.stride
    x_w -= (x_w - k_w) % self.stride
    x = x[:, :, :x_h, :x_w]
    n_h = ceil_div(x_h, self.stride)
    n_w = ceil_div(x_w, self.stride)
    nx = np.zeros((x.shape[0], x.shape[1], n_h * self.stride, n_w * self.stride), dtype=x.dtype)
    nx[:, :, :x_h, :x_w] = x
    x = nx
    split = []
    for i in range(self.stride):
      split_row = []
      for j in range(self.stride):
        split_row.append(x[:, :, i::self.stride, j::self.stride].copy())
      split.append(split_row)
    return split

  def merge(self, split):
    # deduce original shape
    input_channels, output_channels, d_h, d_w = split[-1][-1].shape
    m_h, m_w = 0, 0
    for i, row in enumerate(split):
      for j, item in enumerate(row):
        ic_, oc_, h, w = item.shape
        assert(ic_ == input_channels and oc_ == output_channels)
        if j == 0 and h > d_h: m_h += 1
        if i == 0 and w > d_w: m_w += 1
    x_h = d_h * self.stride + m_h
    x_w = d_w * self.stride + m_w
    # reconstruct
    merged = np.zeros((input_channels, output_channels, x_h, x_w))
    for i, row in enumerate(split):
      for j, item in enumerate(row):
        merged[:, :, i::self.stride, j::self.stride] = item
    return merged

  def forward(self, x_client):
    x_client = np.reshape(x_client, self.input_shape)
    x_client_split = self.split(x_client)
    for i, x_row in enumerate(x_client_split):
      for j, x_item in enumerate(x_row):
        self.convs[i][j].forward_accumulate_ciphers(x_item.flatten())
    # reconstruct
    helper = self.convs[0][0].helper_forward
    y_client = self.crypto.conv2d_decrypt_y(helper, self.crypto.conv2d_deserialize_y(helper, self.recv()))
    return y_client

  def static_forward(self, x_client):
    self.original_shape = x_client.shape
    x_client_split = self.split(x_client)
    y_client = None
    for i, x_row in enumerate(x_client_split):
      for j, x_item in enumerate(x_row):
        y_item = self.convs[i][j].static_forward(x_item)
        if y_client is None: y_client = y_item
        else: y_client += y_item
    return y_client

  def backward(self, partial_y_client):

    helper_backward = self.convs[0][0].helper_backward
    helper_weights = self.convs[0][0].helper_weights
    
    partial_y_client_reshaped = np.reshape(partial_y_client, self.output_shape)
    partial_y_client_encoded_for_weights = self.crypto.conv2d_encode_w(helper_weights, np.transpose(partial_y_client_reshaped, (1, 0, 2, 3)).flatten())
    partial_y_client_cipher_for_weights = self.crypto.encrypt_cipher2d(partial_y_client_encoded_for_weights)
    self.send(self.crypto.serialize(partial_y_client_cipher_for_weights))

    batchsize, _, y_h, y_w = self.convs[0][0].output_shape
    k_h, k_w = self.convs[0][0].kernel_size
    p_h, p_w = k_h - 1, k_w - 1
    padded_partial_y = np.zeros((batchsize, self.output_channels, y_h + p_h*2, y_w + p_w*2), dtype=np.uint64)
    padded_partial_y[:,:,p_h:p_h+y_h,p_w:p_w+y_w] = partial_y_client_reshaped
    padded_partial_y = padded_partial_y.flatten()
    padded_partial_y_client_cipher = self.crypto.encrypt_cipher2d(self.crypto.conv2d_encode_x(helper_backward, padded_partial_y))
    self.send(self.crypto.serialize(padded_partial_y_client_cipher))

    partial_x_split = []
    for i, conv_row in enumerate(self.convs):
      partial_x_row = []
      for j, conv_item in enumerate(conv_row):
        partial_x_item = conv_item.backward_accumulate_ciphers(partial_y_client)
        partial_x_item = np.reshape(partial_x_item, (self.batchsize, self.input_channels, self.sub_image_size[0], self.sub_image_size[1]))
        partial_x_row.append(partial_x_item)
      partial_x_split.append(partial_x_row)
    partial_x_client = self.merge(partial_x_split)
    if partial_x_client.shape != self.input_shape:
      _, _, x_h, x_w = self.input_shape
      return partial_x_client[:, :, :x_h, :x_w].flatten()
    else:
      return partial_x_client.flatten()

  def prepare(self, x_shape):
    self.input_shape = x_shape
    self.batchsize = x_shape[0]
    b, ic, x_h, x_w = x_shape
    assert(ic == self.input_channels)
    k_h, k_w = self.kernel_size
    x_h -= (x_h - k_h) % self.stride
    x_w -= (x_w - k_w) % self.stride
    assert((x_h - k_h) % self.stride == 0 and (x_w - k_w) % self.stride == 0)
    d_h, d_w = ceil_div(x_h, self.stride), ceil_div(x_w, self.stride)
    self.sub_image_size = (d_h, d_w)
    m_h, m_w = x_h % self.stride,  x_w % self.stride
    for i, conv_row in enumerate(self.convs):
      for j, item in enumerate(conv_row):
        # print("prepare", (ic, oc, sk_h, sk_w))
        item.prepare((b, ic, d_h, d_w))
    self.output_shape = (b, self.output_channels, (x_h-k_h)//self.stride+1, (x_w-k_w)//self.stride+1)
    return self.output_shape

  def static_prepare(self, x_shape):
    b, ic, x_h, x_w = x_shape
    assert(ic == self.input_channels)
    k_h, k_w = self.kernel_size
    x_h -= (x_h - k_h) % self.stride
    x_w -= (x_w - k_w) % self.stride
    assert((x_h - k_h) % self.stride == 0 and (x_w - k_w) % self.stride == 0)
    d_h, d_w = x_h // self.stride, x_w // self.stride
    m_h, m_w = x_h % self.stride,  x_w % self.stride
    for i, conv_row in enumerate(self.convs):
      for j, item in enumerate(conv_row):
        sk_h = d_h + (1 if i < m_h else 0)
        sk_w = d_w + (1 if j < m_w else 0)
        # print("prepare", (ic, oc, sk_h, sk_w))
        item.static_prepare((b, ic, sk_h, sk_w))
    return (b, self.output_channels, (x_h-k_h)//self.stride+1, (x_w-k_w)//self.stride+1)

class Conv2dPadded(ClientModule):
  
  def __init__(self, 
    crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication,
    input_channels, output_channels, 
    kernel_size, stride, padding, bias=True
  ):
    self.crypto = crypto
    self.comm = comm
    self.input_channels = input_channels
    self.output_channels = output_channels
    self.kernel_size = kernel_size
    self.stride = stride
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)
    self.kernel_size = kernel_size
    self.has_bias = bias
    self.padding = padding
    self.inner = Conv2dStrided(crypto, comm, input_channels, output_channels, kernel_size, stride, bias)

  def create_convs(self):
    self.inner.create_convs()

  def pad(self, x):
    i, o, h, w = x.shape
    nh = h + self.padding * 2
    nw = w + self.padding * 2
    new_x = np.zeros((i, o, nh, nw))
    new_x[:, :, self.padding:self.padding+h, self.padding:self.padding+w] = x
    return new_x

  def forward(self, x): 
    x = np.reshape(x, self.input_shape)
    x = self.pad(x).flatten()
    return self.inner.forward(x)
  
  def backward(self, partial_y):
    partial_x_padded = self.inner.backward(partial_y)
    nh, nw = self.input_shape[2] + self.padding * 2, self.input_shape[3] + self.padding * 2
    partial_x_padded = np.reshape(partial_x_padded, (self.batchsize, self.input_channels, nh, nw))
    partial_x_padded = partial_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
    return partial_x_padded.flatten()

  def prepare(self, input_shape): 
    self.input_shape = input_shape
    self.batchsize = input_shape[0]
    b, i, h, w = input_shape
    nh = h + self.padding * 2
    nw = w + self.padding * 2
    self.output_shape = self.inner.prepare((b, i, nh, nw))
    return self.output_shape


class Conv1d(ClientModule):

  def __init__(self, crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication, input_channels, output_channels, kernel_size, bias=True):
    super().__init__()
    self.crypto = crypto
    self.comm = comm
    self.input_channels = input_channels
    self.output_channels = output_channels
    self.kernel_size = kernel_size
    self.bias = bias
    self.inner = Conv2d(crypto, comm, input_channels, output_channels, (kernel_size, 1), bias)

  def prepare(self, input_shape):
    self.input_shape = input_shape
    batchsize, channels, input_features = input_shape
    output_shape = self.inner.prepare((batchsize, channels, input_features, 1))
    self.output_shape = (batchsize, self.output_channels, output_shape[2])
    return self.output_shape
  
  def forward(self, x):
    x = self.inner.forward(x)
    return x
  
  def backward(self, partial_y):
    partial_x = self.inner.backward(partial_y)
    return partial_x

class AvgPool2d(ClientModule):
  
  def __init__(self, crypto, comm, kernel_size, stride=None, padding=0):
    super().__init__()
    self.crypto = crypto
    self.comm = comm
    self.kernel_size = kernel_size
    self.padding = padding
    if stride is None: stride = kernel_size
    self.stride = stride
    self.static_prepare = self.prepare
    self.static_forward = self.forward

  def forward(self, x):
    x = np.reshape(x, self.input_shape)
    batchsize, channels, x_h, x_w = self.input_shape

    if self.padding != 0:
      pad = self.padding
      expanded_x = np.zeros((batchsize, channels, x_h + pad*2, x_w + pad*2), dtype=x.dtype)
      expanded_x[:, :, pad:pad+x_h, pad:pad+x_w] = x
      x = expanded_x
      batchsize, channels, x_h, x_w = x.shape

    kernel_size = self.kernel_size
    stride = self.stride

    y_h = (x_h - kernel_size) // stride + 1
    y_w = (x_w - kernel_size) // stride + 1
    
    y = np.zeros((batchsize, channels, y_h, y_w), dtype=x.dtype)
    for i in range(kernel_size):
      for j in range(kernel_size):
        ui = i + stride * y_h
        uj = j + stride * y_w
        y += x[:, :, i:ui:stride, j:uj:stride]

    y = self.crypto.field_mod(y.flatten())
    y = self.comm.divide(y, kernel_size * kernel_size)
    return y
  
  def backward(self, partial_y):
    partial_y = np.reshape(partial_y, self.output_shape)
    batchsize, channels, y_h, y_w = self.output_shape
    kernel_size = self.kernel_size
    _, _, x_h, x_w = self.input_shape
    pad = self.padding
    stride = self.stride
    px_h, px_w = x_h + pad*2, x_w + pad*2
    partial_x = np.zeros((batchsize, channels, px_h, px_w), dtype=partial_y.dtype)
    for i in range(kernel_size):
      for j in range(kernel_size):
        ui = i + stride * y_h
        uj = j + stride * y_w
        partial_x[:, :, i:ui:stride, j:uj:stride] += partial_y
    partial_x = partial_x[:, :, pad:pad+x_h, pad:pad+x_w]
    partial_x = self.crypto.field_mod(partial_x.flatten())
    partial_x = self.comm.divide(partial_x, kernel_size * kernel_size)
    return partial_x

  def prepare(self, input_shape):
    self.input_shape = input_shape
    k = self.kernel_size
    b, c, h, w = input_shape
    pad = self.padding
    stride = self.stride
    kernel_size = self.kernel_size
    y_h = (h + 2*pad - kernel_size) // stride + 1
    y_w = (w + 2*pad - kernel_size) // stride + 1
    output_shape = (b, c, y_h, y_w)
    self.output_shape = output_shape
    return output_shape

class AvgPool1d(ClientModule):
  
  def __init__(self, crypto, comm, kernel_size, stride=None, padding=0):
    super().__init__()
    self.crypto = crypto
    self.comm = comm
    self.kernel_size = kernel_size
    self.padding = padding
    if stride is None: stride = kernel_size
    self.stride = stride
    self.static_prepare = self.prepare
    self.static_forward = self.forward

  def forward(self, x):
    x = np.reshape(x, self.input_shape)
    batchsize, channels, x_features = self.input_shape

    if self.padding != 0:
      pad = self.padding
      expanded_x = np.zeros((batchsize, channels, x_features + pad*2), dtype=x.dtype)
      expanded_x[:, :, pad:pad+x_features] = x
      x = expanded_x
      batchsize, channels, x_features = x.shape

    kernel_size = self.kernel_size
    stride = self.stride

    y_features = (x_features - kernel_size) // stride + 1
    
    y = np.zeros((batchsize, channels, y_features), dtype=x.dtype)
    for i in range(kernel_size):
      ui = i + stride * y_features
      y += x[:, :, i:ui:stride]

    y = self.crypto.field_mod(y.flatten())
    y = self.comm.divide(y, kernel_size)
    return y
  
  def backward(self, partial_y):
    partial_y = np.reshape(partial_y, self.output_shape)
    batchsize, channels, y_features = self.output_shape
    kernel_size = self.kernel_size
    _, _, x_features = self.input_shape
    pad = self.padding
    stride = self.stride
    px_features = x_features + pad*2
    partial_x = np.zeros((batchsize, channels, px_features), dtype=partial_y.dtype)
    for i in range(kernel_size):
      ui = i + stride * y_features
      partial_x[:, :, i:ui:stride] += partial_y
    partial_x = partial_x[:, :, pad:pad+x_features]
    partial_x = self.crypto.field_mod(partial_x.flatten())
    partial_x = self.comm.divide(partial_x, kernel_size)
    return partial_x

  def prepare(self, input_shape):
    self.input_shape = input_shape
    k = self.kernel_size
    b, c, h = input_shape
    pad = self.padding
    stride = self.stride
    kernel_size = self.kernel_size
    y_h = (h + 2*pad - kernel_size) // stride + 1
    output_shape = (b, c, y_h)
    self.output_shape = output_shape
    return output_shape

class BatchNorm2d(ClientModule):

  def __init__(self,
    crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication, 
    channels, affine = True
  ):
    self.crypto = crypto
    self.comm = comm
    self.channels = channels
    self.affine = affine
    self.prepared = False

  def reshape4d(self, x):
    return np.reshape(x, (1, -1, 1, 1))

  def forward(self, x):
    self.x_client = x
    b, c, h, w = x.shape
    assert(c == self.channels)
    # calc stats
    mean = np.mean(x, (0, 2, 3))
    centered = x - self.reshape4d(mean)
    self.centered = centered
    self.comm.send(self.crypto.serialize(self.crypto.vector_encrypt(
      centered.flatten())))
    squared = self.crypto.vector_decrypt(self.crypto.deserialize(self.comm.recv()))
    squared = np.reshape(squared, x.shape)
    variance = np.mean(squared, (0, 2, 3))
    self.comm.send(variance)
    self.comm.send(mean)
    # calc normalized input
    k = self.comm.recv()
    y = x * self.reshape4d(k)
    return y

  def backward_calculate_partial_A(self, partial_y_client):

    assert(self.prepared)
    prep = self.preprocessed
    def linear_operation(x, y):
      return np.sum(x*y, (0,2,3))

    # 0. Receive standard deviation
    istd = self.comm.recv()
    x_hat_client = self.centered * self.reshape4d(istd)

    # 1. Client chooses scalar kx, ky, and send x0 - kx * rx, y0 - ky * ry, kx, ky
    kx, ky = random_scalar(), random_scalar()
    self.send(x_hat_client - prep["rx"] * kx)
    self.send(partial_y_client - prep["ry"] * ky)
    self.send((kx, ky))

    # 2c. Client receives five terms
    z_shape = (self.channels,)
    merged = self.recv()
    hx = self.recv_squeeze_decrypt(z_shape)
    hy = self.recv_squeeze_decrypt(z_shape)
    noised_x1 = self.recv()
    noised_y1 = self.recv()

    # 3. Client calculates x0∘y1 + x1∘y0 + x0∘y0
    #   = -m + x0 ∘ noised_y1 + (noised_x1 + x0) ∘ y0 - hx - hy
    denoised = (
      - merged 
      + linear_operation(x_hat_client, noised_y1)
      + linear_operation(noised_x1 + x_hat_client, partial_y_client)
      - hx - hy
    )
    self.send(denoised)

  def backward(self, partial_y_client):
    batchsize, _, h, w = self.x_client.shape
    n = batchsize * h * w

    assert(self.prepared)
    prep = self.preprocessed

    # partial A
    self.backward_calculate_partial_A(partial_y_client)

    # partial b
    partial_b = np.sum(partial_y_client, (0, 2, 3))
    self.comm.send(partial_b)

    # partial x
    def send_flatten(x, layer):
      self.comm.send(self.crypto.serialize(self.crypto.vector_encrypt(x.flatten())))
    send_flatten(-self.centered / n, -2)
    send_flatten(partial_y_client, -1)
    k = self.comm.recv()
    x_hat = self.crypto.vector_decrypt(self.crypto.deserialize(self.comm.recv()))
    x_hat = np.reshape(x_hat, self.x_client.shape)
    partial_x = self.reshape4d(k) * x_hat
    return partial_x

  def static_prepare(self, input_shape):
    batchsize, channels, h, w = input_shape
    shape = (input_shape[1], input_shape[2], input_shape[3])
    r = random_tensor(shape)
    r_enc = self.crypto.vector_encrypt(r.flatten(), layer=-2)
    self.comm.send(self.crypto.serialize(r_enc))
    p = self.crypto.vector_decrypt(self.crypto.deserialize(self.comm.recv()))
    p_enc = self.crypto.vector_encrypt(p, layer=-2)
    self.send(self.crypto.serialize(p_enc))
    self.delphi_r = r
    return input_shape

  def static_forward(self, x_client):
    batchsize, channels, x_h, x_w = x_client.shape
    r = np.reshape(self.delphi_r, (1, channels, x_h, x_w))
    # online
    # 1. For each sample, selects k. Client send x0 - r*k and k
    k = random_tensor(batchsize, 1)
    noised_x0 = x_client - np.repeat(r, batchsize, axis=0) * np.reshape(k, (batchsize, 1, 1, 1))
    self.send(noised_x0)
    self.send(k)

    # 3. Client gets and decrypts h
    h_encs = self.recv()
    z_shape = (channels, x_h, x_w)
    h = [np.reshape(self.crypto.vector_decrypt(self.crypto.deserialize(i)), z_shape) for i in h_encs]
    h = np.array(h)
    return h

  def prepare(self, input_shape):
    print(f"Preparing BatchNorm2d {input_shape}", flush=True)
    batchsize, channels, h, w = input_shape
    assert(channels == self.channels)

    describe = f"pd-bn2d-client-{batchsize}-{channels}-{h}-{w}"
    p = hashlib.sha256(describe.encode()).hexdigest()[:8]
    filename = f"preprocess/{p}.pickle"

    if os.path.exists(filename) and not config.FORCE_REGENERATE_PREPARE:

      print(f"-- loading {p}")
      f = open(filename, "rb")
      self.preprocessed = pickle.load(f)
      self.send_flatten_encrypt(self.preprocessed["px_share"])
      self.send_flatten_encrypt(self.preprocessed["py_share"])
      f.close()
    
    else:

      print(f"-- generating {p}")

      y_shape = input_shape
      x_shape = input_shape

      # 1. Client chooses rx ~ x, ry ~ y, and sends them
      rx = random_tensor(x_shape)
      ry = random_tensor(y_shape)
      self.send_flatten_encrypt(rx)
      self.send_flatten_encrypt(ry)

      # 5. Flatten noised px, py
      noised_px = self.recv_squeeze_decrypt(x_shape)
      noised_py = self.recv_squeeze_decrypt(y_shape)
      noised_px = np.sum(noised_px, (0,2,3))
      noised_py = np.sum(noised_py, (0,2,3))
      self.send_flatten_encrypt(noised_px)
      self.send_flatten_encrypt(noised_py)

      self.preprocessed = {
        "rx": rx,
        "ry": ry
      }
      
      file = open(filename, "wb")
      pickle.dump({
        "describe": describe,
        "rx": rx,
        "ry": ry,
        "px_share": noised_px.flatten(),
        "py_share": noised_py.flatten(),
      }, file)
      file.close()
    
    self.prepared = True
    return input_shape


class Softmax(ClientModule):

  def __init__(self, crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication):
    super().__init__()
    self.comm = comm
    self.crypto = crypto

  def prepare(self, s): 
    self.shape = s
    return s

  def forward(self, x):
    x = self.comm.softmax(x, self.shape[-1])
    x = self.comm.truncate(x)
    self.y = x.reshape(self.shape)
    return x
    # # This does no cryptographic operations, but just operate in plaintext
    # x = self.crypto.field_mod(self.recv() + x)
    # x = self.crypto.to_decimal(x, shape=self.shape)
    # x = torch.tensor(x)
    # x = torch.nn.functional.softmax(x, dim=-1)
    # self.attention = x
    # x = self.crypto.to_field(x.numpy())
    # x_c = self.crypto.field_random_mask(x.shape)
    # x_s = self.crypto.field_mod(x + x_c)
    # x_c = self.crypto.field_negate(x_c)
    # self.send(x_s)
    # return x_c
  
  def backward(self, dy):
    # # This does no cryptographic operations, but just operate in plaintext
    # dy = self.crypto.field_mod(self.recv() + dy)
    # dy = self.crypto.to_decimal(dy, shape=self.shape)
    # da = torch.tensor(dy)

    # a = self.attention
    # da = a * (da - torch.sum(da * a, dim=-1, keepdim=True))

    # dx = self.crypto.to_field(da.numpy())
    # dx_c = self.crypto.field_random_mask(dx.shape)
    # dx_s = self.crypto.field_mod(dx + dx_c)
    # dx_c = self.crypto.field_negate(dx_c)
    # self.send(dx_s)
    # return dx_c
    
    dy = dy.reshape(self.shape)
    prod = self.comm.elementwise_multiply(dy, self.y)
    prod = self.comm.truncate(prod)
    summed = self.crypto.field_mod(np.sum(prod, axis=-1, keepdims=True))
    sub = self.crypto.field_mod(dy - summed)
    dx = self.comm.elementwise_multiply(sub, self.y)
    dx = self.comm.truncate(dx)
    return dx.flatten()
  
class ScaledDotProductAttention(ClientModule):

  def __init__(self, crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication, dropout=0.1):
    super().__init__()
    self.comm = comm
    self.crypto = crypto
    self.softmax = Softmax(crypto, comm)
    self.dropout = Dropout(crypto, comm, dropout)

  def prepare(self, input_shape):
    self.shape = input_shape
    self.batchsize = input_shape[0]
    self.embed_dim = input_shape[2] // 3
    assert(self.embed_dim * 3 == input_shape[2])
    self.sequence_length = input_shape[1]

    self.helper_qkt = self.crypto.matmul_helper(self.sequence_length, self.embed_dim, self.sequence_length)
    self.helper_av = self.crypto.matmul_helper(self.sequence_length, self.sequence_length, self.embed_dim)

    self.helper_dv = self.helper_av # self.crypto.matmul_helper(self.sequence_length, self.sequence_length, self.embed_dim)
    self.helper_da = self.helper_qkt # self.crypto.matmul_helper(self.sequence_length, self.embed_dim, self.sequence_length)
    self.helper_dk = self.helper_av # self.crypto.matmul_helper(self.sequence_length, self.sequence_length, self.embed_dim)
    self.helper_dq = self.helper_av # self.crypto.matmul_helper(self.sequence_length, self.sequence_length, self.embed_dim)

    self.softmax.prepare((self.sequence_length, self.sequence_length))
    return input_shape
  
  def shares_matmul(self, helper, x_client, w_client, batchsize, input_dim, output_dim):
    x_client = x_client.copy()
    w_client = w_client.copy()
    x_client_encoded = self.crypto.matmul_encode_x(helper, x_client)
    w_client_encoded = self.crypto.matmul_encode_w(helper, w_client)
    x_client_encrypted = self.crypto.encrypt_cipher2d(x_client_encoded)
    w_client_encrypted = self.crypto.encrypt_cipher2d(w_client_encoded)
    self.send(self.crypto.serialize(x_client_encrypted))
    self.send(self.crypto.serialize(w_client_encrypted))
    y_cc = self.crypto.field_mod(
      np.matmul(
        x_client.reshape((batchsize, input_dim)), 
        w_client.reshape(input_dim, output_dim))
    ).flatten()
    y_c = self.crypto.matmul_decrypt_y(helper, self.crypto.matmul_deserialize_y(helper, self.recv()))
    y_c = self.crypto.field_add(y_c, y_cc)
    return y_c
  
  def forward(self, qkv):
    # qkv: [B, N, 3E]
    flatten = len(qkv.shape) == 1
    if flatten:
      qkv = np.reshape(qkv, (self.batchsize, self.sequence_length, 3 * self.embed_dim))
    q_client = qkv[:, :, :self.embed_dim].reshape((self.batchsize, -1))
    k_client = qkv[:, :, self.embed_dim:2*self.embed_dim].reshape((self.batchsize, -1))
    v_client = qkv[:, :, 2*self.embed_dim:].reshape((self.batchsize, -1))

    self.q_client = q_client
    self.k_client = k_client
    self.v_client = v_client

    a_client_list = []
    y_client_list = []
    for b in range(self.batchsize):
      k_transpose = np.transpose(k_client[b].reshape((self.sequence_length, self.embed_dim))).flatten()
      p_client = self.shares_matmul(self.helper_qkt, q_client[b], k_transpose, 
        self.sequence_length, self.embed_dim, self.sequence_length)

      p_client = self.comm.divide(p_client, math.floor(math.sqrt(self.embed_dim) * crypto.SCALE))

      a_client = self.softmax.forward(p_client)
      a_client = self.dropout.forward(a_client)
      a_client_list.append(a_client)

      y_client = self.shares_matmul(self.helper_av, a_client, v_client[b], 
        self.sequence_length, self.sequence_length, self.embed_dim)
      y_client_list.append(y_client)

    self.a_client = np.array(a_client_list)
    y_client = np.array(y_client_list)
    if flatten:
      y_client = y_client.flatten()
    else:
      y_client = y_client.reshape((self.batchsize, self.sequence_length, self.embed_dim))
    return y_client
    
  def backward(self, dy):
    flatten = len(dy.shape) == 1
    if flatten:
      dy = np.reshape(dy, (self.batchsize, self.sequence_length, self.embed_dim))
    dq_client_list = []
    dk_client_list = []
    dv_client_list = []
    for b in range(self.batchsize):
      dy_client = dy[b].flatten()
      a_client = self.a_client[b]

      # dv = a^T * dy
      a_transpose_client = np.transpose(a_client.reshape((self.sequence_length, self.sequence_length))).flatten()
      dv_client = self.shares_matmul(self.helper_dv, a_transpose_client, dy_client, self.sequence_length, self.sequence_length, self.embed_dim)
      dv_client = self.comm.truncate(dv_client)
      dv_client_list.append(dv_client)

      # da = dy * v^T
      v_transpose_client = np.transpose(self.v_client[b].reshape((self.sequence_length, self.embed_dim))).flatten()
      da_client = self.shares_matmul(self.helper_da, dy_client, v_transpose_client, self.sequence_length, self.embed_dim, self.sequence_length)
      da_client = self.comm.truncate(da_client)
      da_client = self.dropout.backward(da_client)

      # dp = softmax_backward(da)
      dp_client = self.softmax.backward(da_client)

      # dk = dp^T * q / sqrt(E)
      q_client = self.q_client[b]
      dp_transpose_client = np.transpose(dp_client.reshape((self.sequence_length, self.sequence_length))).flatten()
      dk_client = self.shares_matmul(self.helper_dk, dp_transpose_client, q_client, self.sequence_length, self.sequence_length, self.embed_dim)
      divisor = math.floor(math.sqrt(self.embed_dim) * crypto.SCALE)
      dk_client = self.comm.divide(dk_client, divisor)
      dk_client_list.append(dk_client)

      # dq = dp * k / sqrt(E)
      dq_client = self.shares_matmul(self.helper_dq, dp_client, self.k_client[b], self.sequence_length, self.sequence_length, self.embed_dim)
      dq_client = self.comm.divide(dq_client, divisor)
      dq_client_list.append(dq_client)

    dq_client = np.array(dq_client_list).reshape((self.batchsize, self.sequence_length, self.embed_dim)) # [B, N, E]
    dk_client = np.array(dk_client_list).reshape((self.batchsize, self.sequence_length, self.embed_dim)) 
    dv_client = np.array(dv_client_list).reshape((self.batchsize, self.sequence_length, self.embed_dim))
    dqkv = np.concatenate([dq_client, dk_client, dv_client], axis=2)
    if flatten:
      dqkv = dqkv.flatten()
    return dqkv

class MultiheadAttention(ClientModule):

  def __init__(self, crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication,
    embed_dim, num_heads, dropout=0.1
  ):
    super().__init__()
    assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.comm = comm
    self.crypto = crypto
    self.in_linear = Linear(crypto, comm, embed_dim, embed_dim*3)
    self.attention = ScaledDotProductAttention(crypto, comm, dropout)
    self.out_linear = Linear(crypto, comm, embed_dim, embed_dim)

  def prepare(self, input_shape):
    # input should be (sequence_length, batchsize, embed_dim)
    assert(len(input_shape) == 3)
    assert(input_shape[2] == self.embed_dim)
    self.batchsize = input_shape[1]
    self.sequence_length = input_shape[0]
    self.in_linear.prepare((self.batchsize * self.sequence_length, self.embed_dim))

    head_dimension = self.embed_dim // self.num_heads
    self.attention.prepare((self.batchsize * self.num_heads, self.sequence_length, head_dimension * 3))

    self.out_linear.prepare((self.batchsize * self.sequence_length, self.embed_dim))
    return (self.sequence_length, self.batchsize, self.embed_dim)
  
  def forward(self, x):
    # x: [N, B, E]
    qkv = self.in_linear.forward(x) # [N, B, 3E]
    qkv = self.comm.truncate(qkv)

    # self.send(qkv)

    # Reshape for feeding into scale-dot attention
    qkv = qkv.reshape((self.sequence_length, self.batchsize, self.embed_dim * 3)) # [N, B, 3E]
    q = qkv[:, :, :self.embed_dim] # [N, B, E]
    k = qkv[:, :, self.embed_dim:2*self.embed_dim] # [N, B, E]
    v = qkv[:, :, 2*self.embed_dim:] # [N, B, E]
    head_dimension = self.embed_dim // self.num_heads
    q = np.transpose(q.reshape((self.sequence_length, self.batchsize * self.num_heads, head_dimension)), (1, 0, 2)) # [B*H, N, E/H]
    k = np.transpose(k.reshape((self.sequence_length, self.batchsize * self.num_heads, head_dimension)), (1, 0, 2)) # [B*H, N, E/H]
    v = np.transpose(v.reshape((self.sequence_length, self.batchsize * self.num_heads, head_dimension)), (1, 0, 2)) # [B*H, N, E/H]
    qkv = np.concatenate([q, k, v], axis=2) # [B*H, N, 3E/H]

    # self.send(qkv)

    y = self.attention.forward(qkv) # [B*H, N, E/H]

    y = np.transpose(y, (1, 0, 2)).reshape((self.sequence_length, self.batchsize, self.embed_dim)) # [N, B, E]    
    y = self.comm.truncate(y.flatten())
    y = self.out_linear.forward(y) # [N, B, E]
    y = self.comm.truncate(y)
    return y
  
  def backward(self, dy):
    # dy: [N, B, E]
    dy = self.out_linear.backward(dy)

    head_dimension = self.embed_dim // self.num_heads
    dy = dy.reshape((self.sequence_length, self.batchsize * self.num_heads, head_dimension)) # [N, B*H, E/H]
    dy = np.transpose(dy, (1, 0, 2)) # [B*H, N, E/H]
    dqkv = self.attention.backward(dy) # [B*H, N, 3E/H]
    dq = dqkv[:, :, :head_dimension] # [B*H, N, E/H]
    dk = dqkv[:, :, head_dimension:2*head_dimension] # [B*H, N, E/H]
    dv = dqkv[:, :, 2*head_dimension:] # [B*H, N, E/H]
    dq = np.transpose(dq, (1, 0, 2)).reshape((self.sequence_length, self.batchsize, self.embed_dim)) # [N, B, E]
    dk = np.transpose(dk, (1, 0, 2)).reshape((self.sequence_length, self.batchsize, self.embed_dim)) # [N, B, E]
    dv = np.transpose(dv, (1, 0, 2)).reshape((self.sequence_length, self.batchsize, self.embed_dim)) # [N, B, E]
    dqkv = np.concatenate([dq, dk, dv], axis=2).flatten() # [N, B, 3E]

    dx = self.in_linear.backward(dqkv) # [N, B, E]
    return dx


class LayerNorm(ClientModule):

  def __init__(self,
    crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication, 
    dims, eps=1e-5, affine=True
  ):
    super().__init__()
    self.comm = comm
    self.crypto = crypto
    self.eps = eps
    self.affine = affine
    self.dims = dims

  def prepare(self, input_shape):
    assert(input_shape[-1] == self.dims)
    self.input_shape = input_shape
    return input_shape
  
  def forward(self, x_share):
    scale = self.crypto.default_scale()
    scale_double = scale ** 2

    # Reshape the input
    x_share = x_share.reshape((-1, self.dims)) # [N, D]

    # Reconstruct the sum
    x_sum = np.sum(x_share, axis=-1, keepdims=True) # [N, 1]

    # Calculate the mean
    mean = self.comm.divide(x_sum, self.dims) # [N, 1]

    # Calculate difference
    diff = x_share - mean # [N, D]
    self.x_diff = diff
    
    # Calculate the variance and standard deviation
    squared = self.comm.elementwise_multiply(diff, diff) # [N, D]
    squared_sum = np.sum(squared, axis=-1, keepdims=True) # [N, 1]
    variance = self.comm.divide(squared_sum, self.dims) # [N, 1]
    variance_eps = variance
    std_inverse = self.comm.sqrt(variance_eps, scale_double, scale, True) # [N, 1]
    self.std_inverse = std_inverse

    # Get normalized value
    std_inverse_repeated = np.repeat(std_inverse, self.dims, axis=1) # [N, D]
    y = self.comm.elementwise_multiply(diff, std_inverse_repeated) # [N, D]
    self.x_normalized = y

    if self.affine: # y = (x - mean) / std * weight + bias
      y = self.comm.truncate(y)
      weight_repeated = np.zeros_like(y)
      y = self.comm.elementwise_multiply(y, weight_repeated) # [N, D]

    return y.flatten()

  def backward(self, dy_share):
    scale = self.crypto.default_scale()
    scale_double = scale ** 2

    # Reshape dy
    dy_share = dy_share.reshape((-1, self.dims)) # [N, D]

    if self.affine:
      x_normalized = self.x_normalized
      x_normalized = self.comm.truncate(x_normalized)
      dweight = self.comm.elementwise_multiply(dy_share, x_normalized) # [N, D]
      dweight = np.sum(dweight, axis=0, keepdims=True).flatten() # [D]
      self.send(dweight)
      dbias = np.sum(dy_share, axis=0, keepdims=True).flatten() # [D]
      self.send(dbias)
      weight_repeated = np.zeros_like(dy_share)
      dy_share = self.comm.elementwise_multiply(dy_share, weight_repeated) # [N, D]
      dy_share = self.comm.truncate(dy_share)

    x_diff = self.x_diff

    # Calculate dy_div_std
    std_inverse = self.std_inverse
    std_inverse_repeated = np.repeat(std_inverse, self.dims, axis=1) # [N, D]
    dy_div_std = self.comm.elementwise_multiply(dy_share, std_inverse_repeated) # [N, D]
    dy_div_std = self.comm.truncate(dy_div_std)

    # Calculate 2 * dvariance = - sum(dy * x_diff) / (std ** 3)
    dvariance_double = self.comm.elementwise_multiply(dy_share, x_diff) # [N, D]
    dvariance_double = np.sum(dvariance_double, axis=-1, keepdims=True) # [N, 1]
    dvariance_double = self.comm.truncate(dvariance_double)  
    for i in range(3): 
      dvariance_double = self.comm.elementwise_multiply(dvariance_double, std_inverse)
      dvariance_double = self.comm.truncate(dvariance_double)  
    dvariance_double = self.crypto.field_negate(dvariance_double) # [N, 1]

    # Calculate dmean = - sum(dy) / std
    dmean = np.sum(dy_share, axis=-1, keepdims=True) # [N, 1]
    dmean = self.comm.elementwise_multiply(dmean, std_inverse)
    dmean = self.crypto.field_negate(dmean) # [N, 1]
  
    # Calculate dx = dy / std + (dvariance * 2 * x_diff + dmean) / N
    dvariance_double_repeated = np.repeat(dvariance_double, self.dims, axis=1) # [N, D]
    dx = self.comm.elementwise_multiply(dvariance_double_repeated, x_diff) # [N, D]
    dmean_repeated = np.repeat(dmean, self.dims, axis=1) # [N, D]
    dx = self.crypto.field_add(dx, dmean_repeated) # [N, D]
    dx = self.comm.truncate(dx)
    dx = self.comm.divide(dx, self.dims) # [N, D]
    dx = self.crypto.field_add(dx, dy_div_std) # [N, D]
    return dx.flatten()

class Dropout(ClientModule):

  def __init__(self, crypto, comm, rate=0.5):
    super().__init__()
    self.comm = comm
    self.crypto = crypto
    self.rate = rate

  def static_prepare(self, shape): return shape
  def static_forward(self, x): return x

  def forward(self, x):
    if self.rate == 0: return x
    p, r = self.comm.recv()
    if p:
      self.choices = r
      return x*r
    else:
      return x

  def backward(self, dy):
    if self.rate == 0: return dy
    return dy*self.choices

  def prepare(self, s): return s

class Residual(ClientModule):
  
  def __init__(self, crypto: crypto.EvaluationUtils, comm: communication_client.ClientCommunication, branch_module):
    super().__init__()
    self.comm = comm
    self.crypto = crypto
    self.branch_module = branch_module

  def prepare(self, input_shape):
    self.input_shape = input_shape
    output_shape = self.branch_module.prepare(input_shape)
    assert(output_shape == input_shape)
    return input_shape
  
  def forward(self, x):
    branch_out = self.branch_module.forward(x)
    return self.crypto.field_add(x, branch_out)
  
  def backward(self, dy):
    branch_dx = self.branch_module.backward(dy)
    return self.crypto.field_add(dy, branch_dx)

class Sequential(ClientModule):

  def __init__(self, crypto, comm, modules):
    self.items = modules
    self.crypto = crypto
    self.comm = comm

  def forward(self, x):
    px = 0
    for each in self.items:
      # before_x = x
      x = each.forward(x)
      # before_px = px
      # px = self.comm.recv()
      # print(each, np.max(np.abs(x + px)))
      # if (np.max(np.abs(x + px)) > 1000):
      #   np.set_printoptions(threshold=sys.maxsize)
      #   x = x.flatten()
      #   px = px.flatten()
      #   before_x = before_x.flatten()
      #   before_px = before_px.flatten()
      #   for i, (x0, px0, bx0, bpx0) in enumerate(zip(x, px, before_x, before_px)):
      #     if (np.abs(x0+px0) > 1000):
      #       print(i, x0, px0, bx0, bpx0)
      #   exit(1)
    return x

  def static_forward(self, x):
    px = 0
    for each in self.items:
      x = each.static_forward(x)
    return x

  def backward(self, partial_y):
    for i in range(len(self.items)):
      partial_y = self.items[len(self.items) - i - 1].backward(partial_y)
    return partial_y

  def prepare(self, input_shape):
    for each in self.items:
      input_shape = each.prepare(input_shape)
    return input_shape

  def static_prepare(self, input_shape):
    for each in self.items:
      input_shape = each.static_prepare(input_shape)
    return input_shape


class Truncate(ClientModule):

  def __init__(self, crypto, comm):
    super().__init__()
    self.crypto = crypto
    self.comm = comm

  def prepare(self, shape):
    self.shape = shape
    return shape

  def forward(self, x):
    return self.comm.truncate(x)
  
  def backward(self, partial_y):
    return partial_y
  
  def to_torch(self):
    return torch_models.Truncate()
  
  def describe(self):
    return {
      "name": "Truncate"
    }
  
  def parameters(self): return []

class TorchNative(ClientModule):
  
  def __init__(self, crypto, comm, preset, device):
    self.crypto = crypto
    self.comm = comm
    self.device = device
    self.model = torch_models.load_model(preset).to(device)
    self.model.eval()
    self.static_forward = self.forward
    self.static_prepare = self.prepare

  def forward(self, x):
    x = self.crypto.to_decimal(x, shape=self.x_shape)
    x = torch.tensor(x, dtype=torch.float32, device=self.device)
    y = self.model(x).detach().cpu().numpy().copy()
    y = self.crypto.to_field(y)
    return y

  def backward(self, dy):
    return None
    return np.zeros(self.x_shape, dtype=np.uint64)

  def prepare(self, input_shape):
    self.x_shape = input_shape
    x = torch.tensor(random_tensor(input_shape), 
      dtype=torch.float32, device=self.device)
    y = self.model(x)
    return y.shape


class TransformerEncoderLayer(ClientModule):

  def __init__(self, 
    crypto: crypto.EvaluationUtils, comm: communication_client.ClientCommunication,
    d_model: int, num_heads: int, dim_feedforward: int, activation: str = "relu", dropout: float = 0.1
  ):
    super().__init__()
    self.crypto = crypto
    self.comm = comm
    self.d_model = d_model
    self.num_heads = num_heads
    self.dim_feedforward = dim_feedforward
    self.activation_type = activation
    self.dropout = dropout

    self_attn = MultiheadAttention(crypto, comm, d_model, num_heads, dropout)
    dropout1 = Dropout(crypto, comm, dropout)
    attention_block = Sequential(crypto, comm, [self_attn, dropout1])
    self.res1 = Residual(crypto, comm, attention_block)
    self.norm1 = LayerNorm(crypto, comm, d_model)

    linear1 = Linear(crypto, comm, d_model, dim_feedforward)
    if activation == "relu":
      activation = ReLU(crypto, comm)
    else:
      activation = GELU(crypto, comm)
    linear2 = Linear(crypto, comm, dim_feedforward, d_model)
    dropout2 = Dropout(crypto, comm, dropout)
    tc = Truncate(crypto, comm)
    feedforward_block = Sequential(crypto, comm, [linear1, activation, linear2, dropout2, tc])
    self.norm2 = LayerNorm(crypto, comm, d_model)
    self.res2 = Residual(crypto, comm, feedforward_block)

  def prepare(self, s):
    self.tensor_shape = s
    s = self.res1.prepare(s)
    s = self.norm1.prepare(s)
    assert(len(s) == 3)
    original_s = s
    s = (s[0] * s[1], s[2])
    s = self.res2.prepare(s)
    s = self.norm2.prepare(s)
    assert(len(s) == 2)
    assert(s[0] * s[1] == original_s[0] * original_s[1] * original_s[2])
    return original_s
  
  def forward(self, x):
    x = self.res1.forward(x)
    x = self.norm1.forward(x)
    x = self.comm.truncate(x)
    x = self.res2.forward(x)
    x = self.norm2.forward(x)
    x = self.comm.truncate(x)
    return x
  
  def backward(self, dy):
    dy = self.norm2.backward(dy)
    dy = self.res2.backward(dy)
    dy = self.norm1.backward(dy)
    dy = self.res1.backward(dy)
    return dy

def client_model_from_description(des, crypto, comm):
  t = des["name"]

  if t == "ReLU":
    return ReLU(crypto, comm)

  if t == "GELU":
    return GELU(crypto, comm)

  elif t == "Flatten":
    return Flatten()

  elif t == "Linear":
    return Linear(crypto, comm, des["input_dims"], des["output_dims"])
  
  elif t == "Conv2d":
    return Conv2d(crypto, comm, des["input_channels"], des["output_channels"], des["kernel_size"])
 
  elif t == "Conv1d":
    return Conv1d(crypto, comm, des["input_channels"], des["output_channels"], des["kernel_size"])
 
  elif t == "Conv2dStrided":
    return Conv2dStrided(crypto, comm, 
      des["input_channels"], des["output_channels"], 
      des["kernel_size"], des["stride"])
  
  elif t == "Conv2dPadded":
    return Conv2dPadded(crypto, comm, 
      des["input_channels"], des["output_channels"], 
      des["kernel_size"], des["stride"],
      des["padding"])
  
  elif t == "Sequential":
    modules = []
    for each in des["modules"]:
      modules.append(client_model_from_description(each, crypto, comm))
    return Sequential(crypto, comm, modules)
  
  elif t == "AvgPool2d":
    return AvgPool2d(crypto, comm, 
      des["kernel_size"],
      des["stride"],
      des["padding"]
    )
  
  elif t == "AvgPool1d":
    return AvgPool1d(crypto, comm, 
      des["kernel_size"],
      des["stride"],
      des["padding"]
    )

  elif t == "BatchNorm2d":
    return BatchNorm2d(crypto, comm,
      des["channels"], des["affine"])

  elif t == "Dropout":
    return Dropout(crypto, comm, des["rate"])

  elif t == "Softmax":
    return Softmax(crypto, comm)

  elif t == "ScaledDotProductAttention":
    return ScaledDotProductAttention(crypto, comm, des["dropout"])
  
  elif t == "MultiheadAttention":
    return MultiheadAttention(crypto, comm, 
      des["embed_dim"], des["num_heads"], des["dropout"])
  
  elif t == "LayerNorm":
    return LayerNorm(crypto, comm, des["dims"], des["eps"], des["affine"])

  elif t == "TorchNative":
    return TorchNative(crypto, comm, des["preset"], des["device"])

  elif t == "Residual":
    branch = client_model_from_description(des["branch"], crypto, comm)
    return Residual(crypto, comm, branch)

  elif t == "TransformerEncoderLayer":
    return TransformerEncoderLayer(
      crypto, comm,
      des["d_model"], des["num_heads"], 
      des["dim_feedforward"], des["activation"], des["dropout"]
    )
  
  elif t == "Truncate":
    return Truncate(crypto, comm)

  else:
    raise Exception("invalid description: typename = " + t)

def create_shares(x: np.ndarray):
  r = np.random.rand(*(x.shape)) * 2 - 1
  return r, x-r