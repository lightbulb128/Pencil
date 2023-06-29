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
from utils import *

def to_torch_tensor(x):
  return torch.tensor(x.astype(np.int64), dtype=torch.int64, device=config.DEVICE)
def to_numpy_tensor(x):
  return x.detach().cpu().numpy().astype(np.uint64)

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

  class Helper:
  
    def __init__(self, 
      crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication,
      batchsize:int, input_dims:int, output_dims:int,
      trans: bool = False, objective: int = 0,
    ):
      self.cpp = crypto.matmul_helper(batchsize, input_dims, output_dims, objective)
      self.input_shape = (batchsize, input_dims)
      self.output_shape = (batchsize, output_dims)
      self.weight_shape = (input_dims, output_dims)
      self.input_size = np.product(self.input_shape)
      self.output_size = np.product(self.output_shape)
      self.weight_size = np.product(self.weight_shape)
      self.crypto = crypto
      self.comm = comm
      self.amount = 0
      self.trans = trans
      self.objective = objective

    def get_name(self):
      batchsize = self.input_shape[0]
      input_dims = self.input_shape[1]
      output_dims = self.output_shape[1]
      describe = f"tk-linear-client-{self.amount}-{batchsize}-{input_dims}-{output_dims}-{self.trans}-{self.objective}"
      return describe

    def send(self, x): self.comm.send(x)
    def recv(self): return self.comm.recv()

    def prepare_single(self, v):
      helper = self.cpp
      if not self.trans:
        v_encoded = self.crypto.matmul_encode_x(helper, v)
      else:
        v_encoded = self.crypto.matmul_encode_w(helper, v)
      v_cipher = self.crypto.encrypt_cipher2d(v_encoded)
      self.send(self.crypto.serialize(v_cipher))
      product = self.crypto.matmul_decrypt_y(helper, self.crypto.matmul_deserialize_y(helper, self.recv()))
      return product

    def prepare(self, amount=config.PREPARE_AMOUNT):
      self.amount = amount

      name = self.get_name()
      if config.PREPARE_FORCE_REGENERATE: loaded = None
      else: loaded = load_preprocess_file(name)

      if loaded is None:
        prepared_shares = [[] for _ in range(amount)]
        prepared_vs = []
        for _ in range(amount):
          shape = self.input_shape if not self.trans else self.weight_shape
          v = self.crypto.field_random_mask(shape).flatten()
          prepared_vs.append(v)
        for i in range(amount):
          for v in prepared_vs:
            prepared_shares[i].append(self.prepare_single(v))
        self.prepared_vs = prepared_vs
        self.prepared_shares = prepared_shares

        if config.PREPARE_CHECK:
          self.send(prepared_vs)
          self.send(prepared_shares)
        
        print(f"Prepared {name}")
        save_preprocess_file(name, {"vs": prepared_vs, "shares": prepared_shares})

      else:
        self.prepared_vs: list = loaded["vs"]
        self.prepared_shares: list = loaded["shares"]

      for i in range(self.amount):
        self.prepared_vs[i] = to_torch_tensor(self.prepared_vs[i])
        for j in range(self.amount):
          self.prepared_shares[i][j] = to_torch_tensor(self.prepared_shares[i][j])

    def execute_plain(self, u, v):
      if not self.trans:
        y = np.matmul(np.reshape(v, self.input_shape), np.reshape(u, self.weight_shape))
      else:
        y = np.matmul(np.reshape(u, self.input_shape), np.reshape(v, self.weight_shape))
      y = self.crypto.field_mod(y)
      return y.flatten()

    def mask(self):
      amount = self.amount
      k = self.crypto.field_random_mask(amount)
      k = to_torch_tensor(k)
      if not self.trans:
        mask = torch.zeros(self.input_size, dtype=torch.int64, device=config.DEVICE)
      else:
        mask = torch.zeros(self.weight_size, dtype=torch.int64, device=config.DEVICE)
      for ki, vi in zip(k, self.prepared_vs):
        mask += ki * vi
      return k, to_torch_tensor(self.crypto.field_mod(to_numpy_tensor(mask)))
    
    def execute(self, x):
      k, w_masked = self.recv()
      l, mask = self.mask()
      x_masked = self.crypto.field_mod(x - to_numpy_tensor(mask)) # x' = x - Σ (lj vj)
      self.send((l, x_masked))
      # [wv]^C_j = w'*vj + Σ ki [ui*vj]^C
      # y^C = Σ lj [wv]^C_j
      # ∴ y^C = w'*(Σlj vj) + Σlj ki [ui * vj]^C
      ljvj = torch.zeros_like(self.prepared_vs[0], dtype=torch.int64, device=config.DEVICE)
      for j in range(self.amount):
        ljvj += l[j] * self.prepared_vs[j]
      y_share = to_torch_tensor(self.execute_plain(w_masked, to_numpy_tensor(ljvj)))
      for i in range(self.amount):
        for j in range(self.amount):
          y_share += (l[j] * k[i]) * self.prepared_shares[i][j]
      y_share = self.crypto.field_mod(to_numpy_tensor(y_share))
      return y_share

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
    y = self.helper_forward.execute(x)
    return y

  def prepare(self, input_shape):
    self.prepared = True
    self.batchsize = input_shape[0]

    self.helper_forward = Linear.Helper(self.crypto, self.comm, self.batchsize, self.input_dims, self.output_dims, objective=0)
    self.helper_forward.prepare()
    
    self.helper_backward = Linear.Helper(self.crypto, self.comm, self.batchsize, self.output_dims, self.input_dims, objective=0)
    self.helper_backward.prepare()

    self.helper_weights_cis   = Linear.Helper(self.crypto, self.comm, self.output_dims, self.batchsize, self.input_dims, objective=0)
    self.helper_weights_cis.prepare()
    self.helper_weights_trans = Linear.Helper(self.crypto, self.comm, self.output_dims, self.batchsize, self.input_dims, trans=True, objective=1)
    self.helper_weights_trans.prepare()

    return (input_shape[0], self.output_dims)

  def backward_calculate_partial_A(self, partial_y_client):
    partial_y_client = np.transpose(np.reshape(partial_y_client, (self.batchsize, self.output_dims)), (1, 0)).flatten()
    
    # dW = dY^T * x
    w_cs = self.helper_weights_cis.execute(partial_y_client)
    w_sc = self.helper_weights_trans.execute(self.x_client)
    w_cc = self.helper_weights_cis.execute_plain(self.x_client, partial_y_client)
    w = self.crypto.field_mod(w_cs + w_sc + w_cc)
    self.send(w)

  def backward(self, partial_y_client):

    # calculate partial_A
    self.backward_calculate_partial_A(partial_y_client)

    # to calculate partial_B
    self.comm.send(np.sum(np.reshape(partial_y_client, (self.batchsize, self.output_dims)), axis=0))
    
    # calculate partial_x shares
    partial_x = self.helper_backward.execute(partial_y_client)
    if config.TRUNCATE_BACKWARD:
      partial_x = self.comm.truncate(partial_x)
    else:
      print("WARNING: backward truncated not enabled.")
    return partial_x

  def export(self): pass

class Conv2d(ClientModule):


  class Helper:
  
    def __init__(self, 
      crypto: crypto.EncryptionUtils, comm: communication_client.ClientCommunication,
      batchsize:int, image_height:int, image_width:int, 
      kernel_height:int, kernel_width:int, input_channels:int, output_channels:int,
      trans: bool = False, objective: int = 0
    ):
      self.cpp = crypto.conv2d_helper(batchsize, image_height, image_width, kernel_height, kernel_width, input_channels, output_channels, objective)
      self.input_shape = (batchsize, input_channels, image_height, image_width)
      self.output_shape = (batchsize, output_channels, image_height - kernel_height + 1, image_width - kernel_width + 1)
      self.weight_shape = (output_channels, input_channels, kernel_height, kernel_width)
      self.input_size = np.product(self.input_shape)
      self.output_size = np.product(self.output_shape)
      self.weight_size = np.product(self.weight_shape)
      self.crypto = crypto
      self.comm = comm
      self.amount = 0
      self.trans = trans
      self.objective = objective

    def get_name(self):
      batchsize, input_channels, image_height, image_width = self.input_shape
      output_channels, _, kernel_height, kernel_width = self.weight_shape
      describe = f"tk-conv2d-client-{self.amount}-{batchsize}-{image_height}-{image_width}-{kernel_height}-{kernel_width}-{input_channels}-{output_channels}-{self.trans}-{self.objective}"
      return describe

    def send(self, x): self.comm.send(x)
    def recv(self): return self.comm.recv()

    def prepare_single(self, v):
      helper = self.cpp
      if not self.trans:
        v_encoded = self.crypto.conv2d_encode_x(helper, v)
      else:
        v_encoded = self.crypto.conv2d_encode_w(helper, v)
      v_cipher = self.crypto.encrypt_cipher2d(v_encoded)
      self.send(self.crypto.serialize(v_cipher))
      product = self.crypto.conv2d_decrypt_y(helper, self.crypto.conv2d_deserialize_y(helper, self.recv()))
      return product

    def prepare(self, amount=config.PREPARE_AMOUNT):
      self.amount = amount

      name = self.get_name()
      if config.PREPARE_FORCE_REGENERATE: loaded = None
      else: loaded = load_preprocess_file(name)

      if loaded is None:
        prepared_shares = [[] for _ in range(amount)]
        prepared_vs = []
        for _ in range(amount):
          shape = self.input_shape if not self.trans else self.weight_shape
          v = self.crypto.field_random_mask(shape).flatten()
          prepared_vs.append(v)
        for i in range(amount):
          for v in prepared_vs:
            prepared_shares[i].append(self.prepare_single(v))
        self.prepared_vs = prepared_vs
        self.prepared_shares = prepared_shares

        if config.PREPARE_CHECK:
          self.send(prepared_vs)
          self.send(prepared_shares)
        
        print(f"Prepared {name}")
        save_preprocess_file(name, {"vs": prepared_vs, "shares": prepared_shares})

      else:
        self.prepared_vs: list = loaded["vs"]
        self.prepared_shares: list = loaded["shares"]

      for i in range(self.amount):
        self.prepared_vs[i] = to_torch_tensor(self.prepared_vs[i])
        for j in range(self.amount):
          self.prepared_shares[i][j] = to_torch_tensor(self.prepared_shares[i][j])

    def execute_plain(self, u, v):
      def to_tensor(x): 
        if x is None: return None
        return torch.tensor(x.astype(np.int64), dtype=torch.long)
      if not self.trans:
        v = np.reshape(v, self.input_shape)
        u = np.reshape(u, self.weight_shape)
        y = torch.conv2d(to_tensor(v), to_tensor(u))
      else:
        v = np.reshape(v, self.weight_shape)
        u = np.reshape(u, self.input_shape)
        y = torch.conv2d(to_tensor(u), to_tensor(v))
      y = y.detach().cpu().numpy().astype(np.uint64).flatten()
      y = self.crypto.field_mod(y)
      return y

    def mask(self):
      amount = self.amount
      k = self.crypto.field_random_mask(amount)
      k = to_torch_tensor(k)
      if not self.trans:
        mask = torch.zeros(self.input_size, dtype=torch.int64, device=config.DEVICE)
      else:
        mask = torch.zeros(self.weight_size, dtype=torch.int64, device=config.DEVICE)
      for ki, vi in zip(k, self.prepared_vs):
        mask += ki * vi
      return k, to_torch_tensor(self.crypto.field_mod(to_numpy_tensor(mask)))
    
    def execute(self, x):
      k, w_masked = self.recv()
      l, mask = self.mask()
      x_masked = self.crypto.field_mod(x - to_numpy_tensor(mask)) # x' = x - Σ (lj vj)
      self.send((l, x_masked))
      # [wv]^C_j = w'*vj + Σ ki [ui*vj]^C
      # y^C = Σ lj [wv]^C_j
      # ∴ y^C = w'*(Σlj vj) + Σlj ki [ui * vj]^C
      ljvj = torch.zeros_like(self.prepared_vs[0], dtype=torch.int64, device=config.DEVICE)
      for j in range(self.amount):
        ljvj += l[j] * self.prepared_vs[j]
      y_share = to_torch_tensor(self.execute_plain(w_masked, to_numpy_tensor(ljvj)))
      for i in range(self.amount):
        for j in range(self.amount):
          y_share += (l[j] * k[i]) * self.prepared_shares[i][j]
      y_share = self.crypto.field_mod(to_numpy_tensor(y_share))
      return y_share

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

  def forward(self, x):
    self.x_client = x
    y = self.helper_forward.execute(x)
    return y

  def forward_accumulate_ciphers(self, x):
    self.x_client = x
    self.collaborate_conv2d_accumulate_ciphers(self.helper_forward, x)

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
    self.helper_forward = Conv2d.Helper(self.crypto, self.comm, 
      self.batchsize, x_h, x_w, k_h, k_w, self.input_channels, self.output_channels, objective=0)
    self.helper_forward.prepare()

    # dx = Conv2d(Pad(dy), permute(flip(w, (2,3)), (1,0,2,3)))
    self.helper_backward = Conv2d.Helper(self.crypto, self.comm, 
      self.batchsize, x_h + k_h - 1, x_w + k_w - 1, k_h, k_w, self.output_channels, self.input_channels, objective=0)
    self.helper_backward.prepare()

    # dw = Conv2d^t(x^t, dy^t)
    self.helper_weights_cis   = Conv2d.Helper(self.crypto, self.comm, 
      self.input_channels, x_h, x_w, y_h, y_w, self.batchsize, self.output_channels, objective=0)
    self.helper_weights_cis.prepare()
    self.helper_weights_trans = Conv2d.Helper(self.crypto, self.comm, 
      self.input_channels, x_h, x_w, y_h, y_w, self.batchsize, self.output_channels, 
      trans=True, objective=1)
    self.helper_weights_trans.prepare()

    self.prepared = True
    return (batchsize, self.output_channels, y_h, y_w)

  def conv2d_plain(self, x, weight):
    def to_tensor(x): 
      if x is None: return None
      return torch.tensor(x.astype(np.int64), dtype=torch.long)
    y = torch.conv2d(to_tensor(x), to_tensor(weight))
    return y.detach().cpu().numpy().astype(np.uint64)

  def backward_calculate_partial_A(self, partial_y_client):
    partial_y_client = np.transpose(partial_y_client, (1, 0, 2, 3)).flatten()
    x_client = np.transpose(np.reshape(self.x_client, self.input_shape), (1, 0, 2, 3)).flatten()

    w_cs = self.helper_weights_trans.execute(partial_y_client)
    w_sc = self.helper_weights_cis.execute(x_client)
    w_cc = self.helper_weights_trans.execute_plain(x_client, partial_y_client)
    w = self.crypto.field_mod(w_cs + w_sc + w_cc)
    self.send(w)

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
    partial_x = self.helper_backward.execute(padded_partial_y)
    if config.TRUNCATE_BACKWARD:
      partial_x = self.comm.truncate(partial_x)
    else:
      print("WARNING: backward truncated not enabled.")
    return partial_x
    
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
    y_client = None
    for i, x_row in enumerate(x_client_split):
      for j, x_item in enumerate(x_row):
        y_item = self.convs[i][j].forward(x_item.flatten())
        if y_client is None: y_client = y_item
        else: y_client += y_item
    y_client = self.crypto.field_mod(y_client)
    return y_client

  def backward(self, partial_y_client):
    partial_x_split = []
    for i, conv_row in enumerate(self.convs):
      partial_x_row = []
      for j, conv_item in enumerate(conv_row):
        partial_x_item = conv_item.backward(partial_y_client)
        partial_x_item = np.reshape(partial_x_item, (self.batchsize, self.input_channels, self.sub_image_size[0], self.sub_image_size[1]))
        partial_x_row.append(partial_x_item)
      partial_x_split.append(partial_x_row)
    partial_x_client = self.merge(partial_x_split)
    if partial_x_client.shape != self.input_shape:
      _, _, x_h, x_w = self.input_shape
      return partial_x_client[:, :, :x_h, :x_w].flatten()
    else:
      return partial_x_client.flatten()

  def prepare(self, x_shape, trusted=False):
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
        if not trusted:
          item.prepare((b, ic, d_h, d_w))
        else:
          item.trusted_prepare((b, ic, d_h, d_w))
    self.output_shape = (b, self.output_channels, (x_h-k_h)//self.stride+1, (x_w-k_w)//self.stride+1)
    return self.output_shape

  def trusted_prepare(self, x_shape):
    self.prepare(x_shape, True)

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
    new_x = np.zeros((i, o, nh, nw), dtype=np.uint64)
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

  def prepare(self, input_shape, trusted=False): 
    self.input_shape = input_shape
    self.batchsize = input_shape[0]
    b, i, h, w = input_shape
    nh = h + self.padding * 2
    nw = w + self.padding * 2
    if not trusted:
      self.output_shape = self.inner.prepare((b, i, nh, nw))
    else:
      self.output_shape = self.inner.trusted_prepare((b, i, nh, nw))
    return self.output_shape

  def trusted_prepare(self, input_shape): self.prepare(input_shape, True)


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

class Dropout(ClientModule):

  def __init__(self, crypto, comm, rate=0.5):
    super().__init__()
    self.comm = comm
    self.crypto = crypto
    self.rate = rate

  def static_prepare(self, shape): return shape
  def static_forward(self, x): return x

  def forward(self, x):
    p, r = self.comm.recv()
    if p:
      self.choices = r
      return x*r
    else:
      return x

  def backward(self, dy):
    return dy*self.choices

  def prepare(self, s): return s

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

  def prepare(self, input_shape):
    self.x_shape = input_shape
    x = torch.tensor(np.zeros(input_shape), 
      dtype=torch.float32, device=self.device)
    y = self.model(x)
    return y.shape

def client_model_from_description(des, crypto, comm):
  t = des["name"]

  if t == "ReLU":
    return ReLU(crypto, comm)

  elif t == "Flatten":
    return Flatten()

  elif t == "Linear":
    return Linear(crypto, comm, des["input_dims"], des["output_dims"])
  
  elif t == "Conv2d":
    return Conv2d(crypto, comm, des["input_channels"], des["output_channels"], des["kernel_size"])
 
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

  elif t == "BatchNorm2d":
    return BatchNorm2d(crypto, comm,
      des["channels"], des["affine"])

  elif t == "Dropout":
    return Dropout(crypto, comm, des["rate"])

  elif t == "TorchNative":
    return TorchNative(crypto, comm, des["preset"], des["device"])

  else:
    raise Exception("invalid description: typename = " + t)

def create_shares(x: np.ndarray):
  r = np.random.rand(*(x.shape)) * 2 - 1
  return r, x-r