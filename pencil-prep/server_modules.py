import numpy as np
import config
from crypto_switch import cryptography as crypto
import communication_server
import math
import torch
import torch.nn
import os, hashlib, pickle
import torch_models
import time
from utils import *

# server: 0 with parameters
# client: 1 with data

def to_torch_tensor(x):
  return torch.tensor(x.astype(np.int64), dtype=torch.int64, device=config.DEVICE)
def to_numpy_tensor(x):
  return x.detach().cpu().numpy().astype(np.uint64)
def random_tensor(shape, bound=1):
  assert(False) # this should not be used
  return np.random.uniform(-bound, bound, shape)
def random_scalar(bound = 1) -> float:
  assert(False) # this should not be used
  return np.random.uniform(-bound, bound)


class Parameter:
  def __init__(self, value):
    self.value = value.copy()
    self.grad = np.zeros_like(value)
    self.shape = self.value.shape
  def set_grad(self, grad):
    assert(grad.shape == self.shape)
    self.grad += grad
  def zero_grad(self):
    self.grad = np.zeros_like(self.value)
    
class ServerModule:

  # Common utilities
  def send_ciphers(self, x):
    self.comm.send(self.crypto.serialize(x))
  def recv_ciphers(self):
    return self.crypto.deserialize(self.comm.recv())
  def send(self, x):
    self.comm.send(x)
  def recv(self):
    return self.comm.recv()

  def mul_scalar(self, x, s):
    return self.crypto.flat_mul_scalar(x, s)
  def mul_enc_scalar(self, s, k):
    return self.crypto.flat_mul_enc_scalar(s, k)
  def mul_enc_k(self, s, k):
    return self.crypto.flat_mul_enc(s, k)
  def add_vector_ciphers(self, x, s):
    self.crypto.flat_add(x, s)
  def add_plain_inplace(self, x, y):
    self.crypto.flat_add_plain(x, y)

  def __init__(self):
    self.is_training = True
  def forward(self, x): 
    raise Exception("not implemented")
  def forward_plain(self, x):
    raise Exception("not implemented")
  def backward_plain(self, partial_y):
    raise Exception("not implemented")
  def describe(self): 
    raise Exception("not implemented") 
  def backward(self, partial_y): 
    raise Exception("not implemented")
  def parameters(self):
    raise Exception("not implemented")
  def to_torch(self): 
    raise Exception("not implemented")
  def prepare(self, input_shape): 
    raise Exception("not implemented")
  def trusted_prepare(self, input_shape):
    raise Exception("not implemented:", self)
  def train(self): self.is_training = True
  def eval(self): self.is_training = False

class ReLU(ServerModule):

  def __init__(self, crypto: crypto.EvaluationUtils, communication: communication_server.ServerCommunication):
    super().__init__()
    self.comm = communication
    self.crypto = crypto

  def forward(self, x):
    self.x = x
    x, self.d = self.comm.relu(x)
    return x

  def forward_plain(self, x):
    self.x = x
    x = self.crypto.relu_plain(x)
    x = self.crypto.truncate_plain(x)
    return x

  def backward_plain(self, partial_y):
    return self.crypto.drelumul_plain(self.x, partial_y)
  
  def backward(self, partial_y):
    partial_x = self.comm.drelumul(partial_y, self.d)
    return partial_x

  def parameters(self): return []
  
  def describe(self):
    return {"name": "ReLU"}

  def to_torch(self): return torch.nn.ReLU()

  def prepare(self, input_shape): return input_shape
  def trusted_prepare(self, input_shape): 
    print(f"ReLU: {input_shape}")
    return input_shape

class Flatten(ServerModule):

  def __init__(self): 
    super().__init__()
  def forward(self, x: np.ndarray):
    return x
  def forward_plain(self, x):
    return x
  def backward_plain(self, partial_y):
    return partial_y
  def backward(self, partial_y):
    return partial_y
  def describe(self):
    return {"name": "Flatten"}
  def parameters(self): return []
  def to_torch(self): return torch.nn.Flatten()
  def prepare(self, input_shape): 
    return (input_shape[0], np.product(input_shape[1:]))
  def trusted_prepare(self, input_shape):
    print(f"Flatten: {input_shape}")
    return (input_shape[0], np.product(input_shape[1:]))


class Linear(ServerModule):

  class Helper:
  
    def __init__(self, 
      crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication,
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
      describe = f"tk-linear-server-{self.amount}-{batchsize}-{input_dims}-{output_dims}-{self.trans}-{self.objective}"
      return describe

    def send(self, x): self.comm.send(x)
    def recv(self): return self.comm.recv()

    def prepare_single(self, u):
      helper = self.cpp
      v_cipher = self.crypto.deserialize(self.recv())
      if not self.trans:
        u_encoded = self.crypto.matmul_encode_w(helper, u)
        product_cipher = self.crypto.matmul(helper, v_cipher, u_encoded)
      else:
        u_encoded = self.crypto.matmul_encode_x(helper, u)
        product_cipher = self.crypto.matmul(helper, u_encoded, v_cipher)
      output_size = np.product(self.output_shape)
      s = self.crypto.field_random_mask(output_size)
      s_encoded = self.crypto.matmul_encode_y(helper, s)
      self.crypto.add_plain_inplace(product_cipher, s_encoded)
      self.send(self.crypto.matmul_serialize_y(helper, product_cipher))
      return self.crypto.field_negate(s)

    def prepare(self, amount=config.PREPARE_AMOUNT):
      if self.comm is None: 
        print("Linear Helper: Omitting preparation.")
        return
      self.amount = amount

      name = self.get_name()
      if config.PREPARE_FORCE_REGENERATE: loaded = None
      else: loaded = load_preprocess_file(name)

      if loaded is None:
        prepared_shares = []
        prepared_us = []
        for _ in range(amount):
          shape = self.input_shape if self.trans else self.weight_shape
          u = self.crypto.field_random_mask(shape).flatten()
          prepared_us.append(u)
        for u in prepared_us:
          prepared_shares_for_this_u = []
          for _ in range(amount):
            prepared_shares_for_this_u.append(self.prepare_single(u))
          prepared_shares.append(prepared_shares_for_this_u)
        self.prepared_us = prepared_us
        self.prepared_shares = prepared_shares

        if config.PREPARE_CHECK:
          us = self.prepared_us
          vs = self.recv()
          other_shares = self.recv()
          for i in range(self.amount):
            for j in range(self.amount):
              p = self.execute_plain(us[i], vs[j])
              m = self.crypto.field_mod(prepared_shares[i][j] + other_shares[i][j])
              assert(np.max(p-m) == 0)
        
        print(f"Prepared {name}")
        save_preprocess_file(name, {"us": prepared_us, "shares": prepared_shares})

      else:
        self.prepared_us: list = loaded["us"]
        self.prepared_shares: list = loaded["shares"]

      for i in range(self.amount):
        self.prepared_us[i] = to_torch_tensor(self.prepared_us[i])
        for j in range(self.amount):
          self.prepared_shares[i][j] = to_torch_tensor(self.prepared_shares[i][j])

    def trusted_prepare(self):
      self.amount = config.PREPARE_AMOUNT
      amount = config.PREPARE_AMOUNT
      server_name = self.get_name()
      client_name = server_name.replace("server", "client")
      us = []
      vs = []
      server_shares = []
      client_shares = []
      for _ in range(amount):
        u_shape = self.input_shape if self.trans else self.weight_shape
        v_shape = self.input_shape if not self.trans else self.weight_shape
        u = self.crypto.field_random_mask(u_shape).flatten()
        us.append(u)
        v = self.crypto.field_random_mask(v_shape).flatten()
        vs.append(v)
      for i in range(self.amount):
        server_shares.append([])
        client_shares.append([])
        for j in range(self.amount):
          p = self.execute_plain(us[i], vs[j])
          sshare = self.crypto.field_random_mask(p.size)
          cshare = self.crypto.field_mod(p - sshare)
          server_shares[i].append(sshare)
          client_shares[i].append(cshare)
      save_preprocess_file(server_name, {"us": us, "shares": server_shares})
      save_preprocess_file(client_name, {"vs": vs, "shares": client_shares})

    def execute_plain(self, u, v):
      def to_tensor(x):
        return torch.tensor(x.astype(np.int64), dtype=torch.long)
      if not self.trans:
        v = to_tensor(np.reshape(v, self.input_shape))
        u = to_tensor(np.reshape(u, self.weight_shape))
        y = torch.matmul(v, u).detach().cpu().numpy().astype(np.uint64)
      else:
        v = to_tensor(np.reshape(v, self.weight_shape))
        u = to_tensor(np.reshape(u, self.input_shape))
        y = torch.matmul(u, v).detach().cpu().numpy().astype(np.uint64)
      y = self.crypto.field_mod(y)
      return y.flatten()

    def mask(self):
      amount = self.amount
      k = self.crypto.field_random_mask(amount)
      k = to_torch_tensor(k)
      if not self.trans:
        mask = torch.zeros(self.weight_size, dtype=torch.int64, device=config.DEVICE)
      else:
        mask = torch.zeros(self.input_size, dtype=torch.int64, device=config.DEVICE)
      for ki, ui in zip(k, self.prepared_us):
        mask += ki * ui
      return k, to_torch_tensor(self.crypto.field_mod(to_numpy_tensor(mask)))

    def execute(self, w: np.ndarray):
      k, mask = self.mask()
      w_masked = self.crypto.field_mod(w - to_numpy_tensor(mask)) # w' = w - Σ (ki ui)
      self.send((k, w_masked))
      wv_share = [] # [wv]^S_j = Σ ki [ui*vj]^S
      for j in range(self.amount):
        share = torch.zeros(self.output_size, dtype=torch.int64, device=config.DEVICE)
        for i in range(self.amount):
          share += k[i] * self.prepared_shares[i][j]
        wv_share.append(share)
      l, x_masked = self.recv()
      y_share = to_torch_tensor(self.execute_plain(w, x_masked)) # y^S = w * x' + Σ lj [wv]^S_j
      for j in range(self.amount):
        y_share += l[j] * wv_share[j]
      y_share = self.crypto.field_mod(to_numpy_tensor(y_share))
      return y_share

    def execute_xs(self, x_share: np.ndarray, w: np.ndarray):
      y_share = self.execute_plain(w, x_share) + self.execute(w)
      y_share = self.crypto.field_mod(y_share)
      return y_share


  def __init__(self, crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, input_dims, output_dims):
    super().__init__()
    self.input_dims = input_dims
    self.output_dims = output_dims
    self.weight = Parameter(np.random.uniform(-1, 1, (output_dims, input_dims)) / math.sqrt(input_dims))
    self.bias = Parameter(np.random.uniform(-1, 1, (output_dims,)) / math.sqrt(input_dims))
    self.comm = comm
    self.crypto = crypto
    self.prepared = False

  def forward(self, x):
    self.x_server = x
    weight = self.crypto.to_field(self.weight.value.transpose())
    y = self.helper_forward.execute_xs(x, weight)

    # add bias
    bias = self.crypto.to_field(self.bias.value, self.crypto.default_scale()**2)
    bias = np.reshape(bias, (1, self.output_dims))
    bias = np.repeat(bias, self.batchsize, axis=0).flatten()
    y = self.crypto.field_add(y, bias)
    return y

  def forward_plain(self, x):
    self.x_server = np.reshape(x, (self.batchsize, self.input_dims))
    w = self.crypto.to_field(self.weight.value.transpose())
    y = self.helper_forward.execute_plain(w, x)
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

  def trusted_prepare(self, input_shape):
    print(f"Linear: {input_shape}")
    self.prepared = True
    self.batchsize = input_shape[0]

    self.helper_forward = Linear.Helper(self.crypto, self.comm, self.batchsize, self.input_dims, self.output_dims, objective=0)
    self.helper_forward.trusted_prepare()

    self.helper_backward = Linear.Helper(self.crypto, self.comm, self.batchsize, self.output_dims, self.input_dims, objective=0)
    self.helper_backward.trusted_prepare()

    self.helper_weights_cis   = Linear.Helper(self.crypto, self.comm, self.output_dims, self.batchsize, self.input_dims, objective=0)
    self.helper_weights_cis.trusted_prepare()
    self.helper_weights_trans = Linear.Helper(self.crypto, self.comm, self.output_dims, self.batchsize, self.input_dims, trans=True, objective=1)
    self.helper_weights_trans.trusted_prepare()

    return (input_shape[0], self.output_dims)

  def backward_calculate_partial_A(self, partial_y_server):
    partial_y_server = np.transpose(np.reshape(partial_y_server, (self.batchsize, self.output_dims)), (1, 0)).flatten()
    
    # dW = dY^T * x
    w_cs = self.helper_weights_cis.execute(self.x_server)
    w_sc = self.helper_weights_trans.execute(partial_y_server)
    w_ss = self.helper_weights_cis.execute_plain(self.x_server, partial_y_server)
    w = self.crypto.field_mod(w_cs + w_sc + w_ss + self.recv())

    partial_w = self.crypto.to_decimal(w, self.crypto.default_scale()**2, (self.output_dims, self.input_dims))
    self.weight.set_grad(partial_w)

  def backward(self, partial_y_server):
    # calculate partial_A
    self.backward_calculate_partial_A(partial_y_server)

    # calculate partial_b
    partial_b = self.comm.recv() + np.sum(np.reshape(partial_y_server, (self.batchsize, self.output_dims)), axis=0)
    partial_b = self.crypto.to_decimal(self.crypto.field_mod(partial_b))
    self.bias.set_grad(partial_b)

    # calculate partial_x shares
    weight = self.crypto.to_field(self.weight.value)
    partial_x = self.helper_backward.execute_xs(partial_y_server, weight)
    if config.TRUNCATE_BACKWARD:
      partial_x = self.comm.truncate(partial_x)
    else:
      print("WARNING: backward truncated not enabled.")
    return partial_x

  def backward_plain(self, partial_y_server):
    partial_y = np.reshape(partial_y_server, (self.batchsize, self.output_dims))
    
    # calculate partial_A
    partial_w = self.helper_weights_cis.execute_plain(self.x_server.flatten(), np.transpose(partial_y, (1, 0)).flatten())
    partial_w = self.crypto.to_decimal(partial_w, self.crypto.default_scale()**2, (self.output_dims, self.input_dims))
    self.weight.set_grad(partial_w)

    # calculate partial_b
    partial_b = np.sum(partial_y, axis=0)
    partial_b = self.crypto.to_decimal(self.crypto.field_mod(partial_b))
    self.bias.set_grad(partial_b)
    
    # calculate partial_x
    w = self.crypto.to_field(self.weight.value)
    partial_x = self.helper_backward.execute_plain(w, partial_y_server)
    partial_x = self.crypto.truncate_plain(partial_x)
    return partial_x

  def describe(self):
    return {
      "name": "Linear",
      "input_dims": self.weight.shape[1],
      "output_dims": self.weight.shape[0]
    }
    
  def parameters(self):
    return [self.weight, self.bias]

  def to_torch(self): 
    ret = torch.nn.Linear(self.weight.shape[1], self.weight.shape[0])
    ret.weight.data = torch.tensor(self.weight.value, dtype=torch.float32)
    ret.bias.data = torch.tensor(self.bias.value, dtype=torch.float32)
    return ret

class Conv2d(ServerModule):


  class Helper:
  
    def __init__(self, 
      crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication,
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
      describe = f"tk-conv2d-server-{self.amount}-{batchsize}-{image_height}-{image_width}-{kernel_height}-{kernel_width}-{input_channels}-{output_channels}-{self.trans}-{self.objective}"
      return describe

    def send(self, x): self.comm.send(x)
    def recv(self): return self.comm.recv()

    def prepare_single(self, u):
      helper = self.cpp
      v_cipher = self.crypto.deserialize(self.recv())
      if not self.trans:
        u_encoded = self.crypto.conv2d_encode_w(helper, u)
        product_cipher = self.crypto.conv2d(helper, v_cipher, u_encoded)
      else:
        u_encoded = self.crypto.conv2d_encode_x(helper, u)
        product_cipher = self.crypto.conv2d(helper, u_encoded, v_cipher)
      output_size = np.product(self.output_shape)
      s = self.crypto.field_random_mask(output_size)
      s_encoded = self.crypto.conv2d_encode_y(helper, s)
      self.crypto.add_plain_inplace(product_cipher, s_encoded)
      self.send(self.crypto.conv2d_serialize_y(helper, product_cipher))
      return self.crypto.field_negate(s)

    def prepare(self, amount=config.PREPARE_AMOUNT):
      if self.comm is None: 
        print("Conv2d Helper: Omitting preparation.")
        return
      self.amount = amount

      name = self.get_name()
      if config.PREPARE_FORCE_REGENERATE: loaded = None
      else: loaded = load_preprocess_file(name)

      if loaded is None:
        prepared_shares = []
        prepared_us = []
        for _ in range(amount):
          shape = self.input_shape if self.trans else self.weight_shape
          u = self.crypto.field_random_mask(shape).flatten()
          prepared_us.append(u)
        for u in prepared_us:
          prepared_shares_for_this_u = []
          for _ in range(amount):
            prepared_shares_for_this_u.append(self.prepare_single(u))
          prepared_shares.append(prepared_shares_for_this_u)
        self.prepared_us = prepared_us
        self.prepared_shares = prepared_shares

        if config.PREPARE_CHECK:
          us = self.prepared_us
          vs = self.recv()
          other_shares = self.recv()
          for i in range(self.amount):
            for j in range(self.amount):
              p = self.execute_plain(us[i], vs[j])
              m = self.crypto.field_mod(prepared_shares[i][j] + other_shares[i][j])
              assert(np.max(p-m) == 0)
        
        print(f"Prepared {name}")
        save_preprocess_file(name, {"us": prepared_us, "shares": prepared_shares})

      else:
        self.prepared_us: list = loaded["us"]
        self.prepared_shares: list = loaded["shares"]

      for i in range(self.amount):
        self.prepared_us[i] = to_torch_tensor(self.prepared_us[i])
        for j in range(self.amount):
          self.prepared_shares[i][j] = to_torch_tensor(self.prepared_shares[i][j])

    def trusted_prepare(self):
      self.amount = config.PREPARE_AMOUNT
      amount = config.PREPARE_AMOUNT
      server_name = self.get_name()
      client_name = server_name.replace("server", "client")
      us = []
      vs = []
      server_shares = []
      client_shares = []
      for _ in range(amount):
        u_shape = self.input_shape if self.trans else self.weight_shape
        v_shape = self.input_shape if not self.trans else self.weight_shape
        u = self.crypto.field_random_mask(u_shape).flatten()
        us.append(u)
        v = self.crypto.field_random_mask(v_shape).flatten()
        vs.append(v)
      for i in range(self.amount):
        server_shares.append([])
        client_shares.append([])
        for j in range(self.amount):
          p = self.execute_plain(us[i], vs[j])
          sshare = self.crypto.field_random_mask(p.size)
          cshare = self.crypto.field_mod(p - sshare)
          server_shares[i].append(sshare)
          client_shares[i].append(cshare)
      save_preprocess_file(server_name, {"us": us, "shares": server_shares})
      save_preprocess_file(client_name, {"vs": vs, "shares": client_shares})

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
        mask = torch.zeros(self.weight_size, dtype=torch.int64, device=config.DEVICE)
      else:
        mask = torch.zeros(self.input_size, dtype=torch.int64, device=config.DEVICE)
      for ki, ui in zip(k, self.prepared_us):
        mask += ki * ui
      return k, to_torch_tensor(self.crypto.field_mod(to_numpy_tensor(mask)))

    def execute(self, w: np.ndarray):
      k, mask = self.mask()
      w_masked = self.crypto.field_mod(w - to_numpy_tensor(mask)) # w' = w - Σ (ki ui)
      self.send((k, w_masked))
      wv_share = [] # [wv]^S_j = Σ ki [ui*vj]^S
      for j in range(self.amount):
        share = torch.zeros(self.output_size, dtype=torch.int64, device=config.DEVICE)
        for i in range(self.amount):
          share += k[i] * self.prepared_shares[i][j]
        wv_share.append(share)
      l, x_masked = self.recv()
      y_share = to_torch_tensor(self.execute_plain(w, x_masked)) # y^S = w * x' + Σ lj [wv]^S_j
      for j in range(self.amount):
        y_share += l[j] * wv_share[j]
      y_share = self.crypto.field_mod(to_numpy_tensor(y_share))
      return y_share

    def execute_xs(self, x_share: np.ndarray, w: np.ndarray):
      y_share = self.execute(w) + self.execute_plain(w, x_share)
      y_share = self.crypto.field_mod(y_share)
      return y_share


  def __init__(self, crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, input_channels, output_channels, kernel_size, bias=True):
    super().__init__()
    self.comm = comm
    self.input_channels = input_channels
    self.output_channels = output_channels
    if isinstance(kernel_size, int):
      self.kernel_size = (kernel_size, kernel_size)
    else:
      self.kernel_size = kernel_size
    k_h, k_w = self.kernel_size
    # assert(k_h % 2 == 1)
    # assert(k_w % 2 == 1)
    factor = 1 / math.sqrt(input_channels * k_h * k_w)
    self.weight = Parameter(np.random.uniform(-1, 1, (output_channels, input_channels, k_h, k_w)) * factor)
    if bias:
      self.bias = Parameter(np.random.uniform(-1, 1, (output_channels,)) * factor)
    else:
      self.bias = None
    self.crypto = crypto
    self.prepared = False
    self.restrict_kernel_size = None

  def conv2d_plain(self, x, weight):
    def to_tensor(x): 
      if x is None: return None
      return torch.tensor(x.astype(np.int64), dtype=torch.long)
    y = torch.conv2d(to_tensor(x), to_tensor(weight))
    return y.detach().cpu().numpy().astype(np.uint64)

  def forward(self, x):
    self.x_server = x
    weight = self.crypto.to_field(self.weight.value)
    y_server = self.helper_forward.execute_xs(x, weight)

    if not (self.bias is None):
      bias = self.crypto.to_field(self.bias.value, self.crypto.default_scale()**2)
      batchsize, _, y_h, y_w = self.output_shape
      bias = np.reshape(bias, (1, self.output_channels, 1))
      bias = np.repeat(bias, batchsize, axis=0)
      bias = np.repeat(bias, y_h * y_w, axis=2).flatten()
      y_server = self.crypto.field_add(y_server, bias)

    return y_server

  def forward_plain(self, x):
    x = np.reshape(x, self.input_shape)
    # w_shape = (self.output_channels, self.input_channels, self.kernel_size[0], self.kernel_size[1])
    self.x_server = x
    w = self.crypto.to_field(self.weight.value, flatten=False)
    y = self.conv2d_plain(x, w).flatten()
    
    if not (self.bias is None):
      bias = self.crypto.to_field(self.bias.value, self.crypto.default_scale()**2)
      batchsize, _, y_h, y_w = self.output_shape
      bias = np.reshape(bias, (1, self.output_channels, 1))
      bias = np.repeat(bias, batchsize, axis=0)
      bias = np.repeat(bias, y_h * y_w, axis=2).flatten()
      y = self.crypto.field_add(y, bias)
    
    return y

  def restrict_kernel(self, restrict_size):
    self.restrict_kernel_size = restrict_size
    self.weight.value[:, :, restrict_size[0]:, :] = 0
    self.weight.value[:, :, :, restrict_size[1]:] = 0

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

  def trusted_prepare(self, input_shape):
    print("Conv2d:", input_shape)
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
    self.helper_forward.trusted_prepare()

    # dx = Conv2d(Pad(dy), permute(flip(w, (2,3)), (1,0,2,3)))
    self.helper_backward = Conv2d.Helper(self.crypto, self.comm, 
      self.batchsize, x_h + k_h - 1, x_w + k_w - 1, k_h, k_w, self.output_channels, self.input_channels, objective=0)
    self.helper_backward.trusted_prepare()

    # dw = Conv2d^t(x^t, dy^t)
    self.helper_weights_cis   = Conv2d.Helper(self.crypto, self.comm, 
      self.input_channels, x_h, x_w, y_h, y_w, self.batchsize, self.output_channels, objective=0)
    self.helper_weights_cis.trusted_prepare()
    self.helper_weights_trans = Conv2d.Helper(self.crypto, self.comm, 
      self.input_channels, x_h, x_w, y_h, y_w, self.batchsize, self.output_channels, 
      trans=True, objective=1)
    self.helper_weights_trans.trusted_prepare()

    self.prepared = True
    return (batchsize, self.output_channels, y_h, y_w)

  def backward_calculate_partial_A(self, partial_y_server):
    partial_y_server = np.transpose(partial_y_server, (1, 0, 2, 3)).flatten()
    x_server = np.transpose(np.reshape(self.x_server, self.input_shape), (1, 0, 2, 3)).flatten()

    w_cs = self.helper_weights_trans.execute(x_server)
    w_sc = self.helper_weights_cis.execute(partial_y_server)
    w_ss = self.helper_weights_trans.execute_plain(x_server, partial_y_server)
    w = self.crypto.field_mod(w_cs + w_sc + w_ss + self.recv())

    partial_w = self.crypto.to_decimal(w, self.crypto.default_scale()**2, 
      (self.input_channels, self.output_channels, self.kernel_size[0], self.kernel_size[1]))
    partial_w = np.transpose(partial_w, (1, 0, 2, 3))
    if not (self.restrict_kernel_size is None):
      partial_w[:, :, self.restrict_kernel_size[0]:, :] = 0
      partial_w[:, :, :, self.restrict_kernel_size[1]:] = 0
    self.weight.set_grad(partial_w)

  def backward(self, partial_y_server):

    batchsize, _, x_h, x_w = self.input_shape
    _, _, y_h, y_w = self.output_shape
    k_h, k_w = self.kernel_size
    partial_y_server = np.reshape(partial_y_server, self.output_shape)

    # calculate partial_A
    self.backward_calculate_partial_A(partial_y_server)

    # calculate partial_b
    if self.bias:
      partial_b = np.sum(np.sum(np.sum(partial_y_server, axis=0), axis=1), axis=1)
      partial_b += self.comm.recv()    
      partial_b = self.crypto.to_decimal(self.crypto.field_mod(partial_b))
      self.bias.set_grad(partial_b)

    # calculate partial_x
    p_h, p_w = k_h - 1, k_w - 1
    padded_partial_y = np.zeros((batchsize, self.output_channels, y_h + p_h*2, y_w + p_w*2), dtype=np.uint64)
    padded_partial_y[:,:,p_h:p_h+y_h,p_w:p_w+y_w] = partial_y_server
    padded_partial_y = padded_partial_y.flatten()
    weight = self.crypto.to_field(np.transpose(np.flip(self.weight.value, (2,3)), (1,0,2,3)))
    partial_x = self.helper_backward.execute_xs(padded_partial_y, weight)
    if config.TRUNCATE_BACKWARD:
      partial_x = self.comm.truncate(partial_x)
    else:
      print("WARNING: backward truncated not enabled.")
    return partial_x

  def backward_plain(self, partial_y_server):

    batchsize, _, x_h, x_w = self.input_shape
    _, _, y_h, y_w = self.output_shape
    k_h, k_w = self.kernel_size
    partial_y = np.reshape(partial_y_server, self.output_shape)

    # calculate partial_A
    partial_w = self.conv2d_plain(
      np.transpose(self.x_server, (1, 0, 2, 3)),
      np.transpose(partial_y, (1, 0, 2, 3))
    )
    partial_w = self.crypto.field_mod(partial_w)
    partial_w = np.transpose(partial_w, (1, 0, 2, 3))
    partial_w = self.crypto.to_decimal(partial_w, self.crypto.default_scale()**2)
    self.weight.set_grad(partial_w)

    # calculate partial_b
    if self.bias:
      partial_b = np.sum(np.sum(np.sum(partial_y, axis=0), axis=1), axis=1)
      partial_b = self.crypto.to_decimal(self.crypto.field_mod(partial_b))
      self.bias.set_grad(partial_b)

    # calculate partial_x
    p_h, p_w = k_h - 1, k_w - 1
    padded_partial_y = np.zeros((batchsize, self.output_channels, y_h + p_h*2, y_w + p_w*2), dtype=np.uint64)
    padded_partial_y[:,:,p_h:p_h+y_h,p_w:p_w+y_w] = partial_y
    w_transpose = np.transpose(np.flip(self.weight.value, (2,3)), (1,0,2,3))
    w_transpose = self.crypto.to_field(w_transpose, flatten=False)
    partial_x = self.conv2d_plain(padded_partial_y, w_transpose).flatten()
    partial_x = self.crypto.field_mod(partial_x)
    partial_x = self.crypto.truncate_plain(partial_x)
    return partial_x

  def describe(self):
    return {
      "name": "Conv2d",
      "input_channels": self.input_channels,
      "output_channels": self.output_channels,
      "kernel_size": self.kernel_size,
    }

  def parameters(self):
    if self.bias:
      return [self.weight, self.bias]
    else:
      return [self.weight,]

  def to_torch(self):
    has_bias = self.bias is not None
    ret = torch.nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, bias=has_bias)
    ret.weight.data = torch.tensor(self.weight.value, dtype=torch.float32)
    if has_bias:
      ret.bias.data = torch.tensor(self.bias.value, dtype=torch.float32)
    return ret

class Conv2dStrided(ServerModule):
  
  def __init__(self, 
    crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, 
    input_channels, output_channels, kernel_size, stride, bias=True,
    dont_create_conv = False
  ):
    super().__init__()
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
    if not dont_create_conv:
      self.create_convs()

  def create_convs(self, whole_weights=None, bias=None):
    k_h, k_w = self.kernel_size
    d_h, d_w = ceil_div(k_h, self.stride), ceil_div(k_w, self.stride)
    self.sub_kernel_size = (d_h, d_w)
    convs = []
    for i in range(self.stride):
      convs_row = []
      for j in range(self.stride):
        has_bias = (i==0 and j==0 and self.has_bias)
        item = Conv2d(self.crypto, self.comm, self.input_channels, self.output_channels, self.sub_kernel_size, has_bias)
        if whole_weights is not None:
          part_weights = whole_weights[:, :, i::self.stride, j::self.stride]
          pshape = part_weights.shape
          item.weight.value[:, :, :pshape[2], :pshape[3]] = part_weights
          item.restrict_kernel((pshape[2], pshape[3]))
        if (bias is not None) and (has_bias):
          item.bias.value = bias
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

  def forward(self, x_server):
    x_server = np.reshape(x_server, self.input_shape)
    x_server_split = self.split(x_server)
    y_server = None
    for i, x_row in enumerate(x_server_split):
      for j, x_item in enumerate(x_row):
        y_item = self.convs[i][j].forward(x_item.flatten())
        if y_server is None: y_server = y_item
        else: y_server += y_item
    y_server = self.crypto.field_mod(y_server)
    return y_server

  def forward_plain(self, x):
    x = np.reshape(x, self.input_shape)
    x_split = self.split(x)
    y = None
    for i, x_row in enumerate(x_split):
      for j, x_item in enumerate(x_row):
        y_item = self.convs[i][j].forward_plain(x_item.flatten())
        if y is None: y = y_item
        else: y += y_item
    y = self.crypto.field_mod(y.flatten())
    return y

  def describe(self):
    return {
      "name": "Conv2dStrided",
      "input_channels": self.input_channels,
      "output_channels": self.output_channels,
      "kernel_size": self.kernel_size,
      "stride": self.stride,
    }

  def backward(self, partial_y_server):
    partial_x_split = []
    for i, conv_row in enumerate(self.convs):
      partial_x_row = []
      for j, conv_item in enumerate(conv_row):
        partial_x_item = conv_item.backward(partial_y_server)
        partial_x_item = np.reshape(partial_x_item, (self.batchsize, self.input_channels, self.sub_image_size[0], self.sub_image_size[1]))
        partial_x_row.append(partial_x_item)
      partial_x_split.append(partial_x_row)
    partial_x_server = self.merge(partial_x_split)
    if partial_x_server.shape != self.input_shape:
      _, _, x_h, x_w = self.input_shape
      return partial_x_server[:, :, :x_h, :x_w].flatten()
    else:
      return partial_x_server.flatten()

  def backward_plain(self, partial_y_server):
    partial_x_split = []
    for i, conv_row in enumerate(self.convs):
      partial_x_row = []
      for j, conv_item in enumerate(conv_row):
        partial_x_item = conv_item.backward_plain(partial_y_server)
        partial_x_item = np.reshape(partial_x_item, (self.batchsize, self.input_channels, self.sub_image_size[0], self.sub_image_size[1]))
        partial_x_row.append(partial_x_item)
      partial_x_split.append(partial_x_row)
    partial_x_server = self.merge(partial_x_split)
    if partial_x_server.shape != self.input_shape:
      _, _, x_h, x_w = self.input_shape
      return partial_x_server[:, :, :x_h, :x_w].flatten()
    else:
      return partial_x_server.flatten()

  def parameters(self):
    ps = []
    for row in self.convs:
      for item in row:
        ps += item.parameters()
    return ps

  def merged_weight(self):
    collected_weights = [[item.weight.value for item in row] for row in self.convs]
    merged_weight = self.merge(collected_weights)
    merged_weight = merged_weight[:, :, :self.kernel_size[0], :self.kernel_size[1]]
    return merged_weight

  def merged_bias(self):
    if self.has_bias:
      return self.convs[0][0].bias.value
    else:
      return None

  def merged_weight_grad(self):
    collected_weights = [[item.weight.grad for item in row] for row in self.convs]
    merged_weight = self.merge(collected_weights)
    merged_weight = merged_weight[:, :, :self.kernel_size[0], :self.kernel_size[1]]
    return merged_weight

  def merged_bias_grad(self):
    if self.has_bias:
      return self.convs[0][0].bias.grad
    else:
      return None

  def to_torch(self):
    merged_weight = self.merged_weight()
    torch_module = torch.nn.Conv2d(self.input_channels, self.output_channels,
      self.kernel_size, self.stride, bias=self.has_bias)
    torch_module.weight.data = torch.tensor(merged_weight, dtype=torch.float32)
    if self.has_bias:
      torch_module.bias.data = torch.tensor(self.convs[0][0].bias.value, dtype=torch.float32)
    return torch_module

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
    print("Conv2dStrided:", x_shape)
    return self.prepare(x_shape, True)

  def train(self):
    self.is_training = True
    for x in self.convs:
      for y in x: y.train()
      
  def eval(self):
    self.is_training = False
    for x in self.convs:
      for y in x: y.eval()
      

class Conv2dPadded(ServerModule):
  
  def __init__(self, 
    crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, 
    input_channels, output_channels, kernel_size, stride, padding, bias=True,
    dont_create_conv = False
  ):
    super().__init__()
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
    self.inner = Conv2dStrided(crypto, comm, input_channels, output_channels, kernel_size, stride, bias, dont_create_conv)

  def create_convs(self, whole_weight=None, bias=None):
    self.inner.create_convs(whole_weight, bias)

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

  def forward_plain(self, x):
    x = np.reshape(x, self.input_shape)
    x = self.pad(x).flatten()
    return self.inner.forward_plain(x)


  def describe(self): 
    return {
      "name": "Conv2dPadded",
      "input_channels": self.input_channels,
      "output_channels": self.output_channels,
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding,
    }

  def backward(self, partial_y):
    partial_x_padded = self.inner.backward(partial_y)
    nh, nw = self.input_shape[2] + self.padding * 2, self.input_shape[3] + self.padding * 2
    partial_x_padded = np.reshape(partial_x_padded, (self.batchsize, self.input_channels, nh, nw))
    partial_x_padded = partial_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
    return partial_x_padded.flatten()

  def backward_plain(self, partial_y):
    partial_x_padded = self.inner.backward_plain(partial_y)
    nh, nw = self.input_shape[2] + self.padding * 2, self.input_shape[3] + self.padding * 2
    partial_x_padded = np.reshape(partial_x_padded, (self.batchsize, self.input_channels, nh, nw))
    partial_x_padded = partial_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
    return partial_x_padded.flatten()

  def parameters(self):
    return self.inner.parameters()

  def to_torch(self): 
    not_padded = self.inner.to_torch()
    torch_module = torch.nn.Conv2d(self.input_channels, self.output_channels,
      self.kernel_size, self.stride, self.padding)
    torch_module.weight.data = not_padded.weight.data
    torch_module.bias.data = not_padded.bias.data
    return torch_module

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

  def trusted_prepare(self, input_shape): 
    print("Conv2dPadded:", input_shape)
    return self.prepare(input_shape, True)

  def merged_weight(self): return self.inner.merged_weight() 
  def merged_bias(self): return self.inner.merged_bias() 
  def merged_weight_grad(self): return self.inner.merged_weight_grad() 
  def merged_bias_grad(self): return self.inner.merged_bias_grad() 

  def train(self): 
    self.is_training = True
    self.inner.train()
  def eval(self): 
    self.is_training = False
    self.inner.eval()

class Conv1d(ServerModule):

  def __init__(self, crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, input_channels, output_channels, kernel_size, bias=True):
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
  
  def trusted_prepare(self, input_shape): 
    print("Conv1d:", input_shape)
    self.input_shape = input_shape
    batchsize, channels, input_features = input_shape
    output_shape = self.inner.trusted_prepare((batchsize, channels, input_features, 1))
    self.output_shape = (batchsize, self.output_channels, output_shape[2])
    return self.output_shape

  def forward(self, x):
    x = self.inner.forward(x)
    return x
  
  def backward(self, partial_y):
    partial_x = self.inner.backward(partial_y)
    return partial_x
  
  def parameters(self):
    return self.inner.parameters()
  
  def describe(self):
    return {
      "name": "Conv1d",
      "input_channels": self.input_channels,
      "output_channels": self.output_channels,
      "kernel_size": self.kernel_size,
    }
  
  def to_torch(self):
    torch_module = torch.nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size)
    torch_module.weight.data = torch.tensor(self.inner.weight.value, dtype=torch.float32).reshape(torch_module.weight.data.shape)
    if self.bias:
      torch_module.bias.data = torch.tensor(self.inner.bias.value, dtype=torch.float32)
    return torch_module
  
class AvgPool2d(ServerModule):
  
  def __init__(self, crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, kernel_size, stride=None, padding=0):
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

  def forward_plain(self, x):
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
    y = self.crypto.divide_plain(y, kernel_size * kernel_size)
    return y
  
  def backward(self, partial_y, plain=False):
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
    if not plain:
      partial_x = self.comm.divide(partial_x, kernel_size * kernel_size)
    else:
      partial_x = self.crypto.divide_plain(partial_x, kernel_size * kernel_size)
    return partial_x
  
  def backward_plain(self, partial_y):
    return self.backward(partial_y, True)

  def parameters(self): return []

  def describe(self):
    return {
      "name": "AvgPool2d",
      "kernel_size": self.kernel_size,
      "padding": self.padding,
      "stride": self.stride
    }

  def to_torch(self):
    return torch.nn.AvgPool2d(self.kernel_size, self.stride, self.padding)

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

  def trusted_prepare(self, input_shape): 
    print("AvgPool2d:", input_shape)
    return self.prepare(input_shape)

class AvgPool1d(ServerModule):
  
  def __init__(self, crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, kernel_size, stride=None, padding=0):
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

  def parameters(self): return []

  def describe(self):
    return {
      "name": "AvgPool1d",
      "kernel_size": self.kernel_size,
      "padding": self.padding,
      "stride": self.stride
    }

  def to_torch(self):
    return torch.nn.AvgPool1d(self.kernel_size, self.stride, self.padding)

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

  def trusted_prepare(self, input_shape): 
    print("AvgPool1d:", input_shape)
    return self.prepare(input_shape)

class BatchNorm2d(ServerModule):

  def __init__(self,
    crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, 
    channels, eps=1e-5, momentum=0.1, affine=True
  ):
    super().__init__()
    assert(False) # TODO: this have not adapted flat_ series.
    self.crypto = crypto
    self.comm = comm
    self.channels = channels
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    if affine:
      self.weight = Parameter(np.ones((channels,)))
      self.bias = Parameter(np.zeros((channels,)))
    self.running_mean = np.zeros((channels,))
    self.running_var = np.ones((channels,))
    self.prepared = False

  def reshape4d(self, x):
    return np.reshape(x, (1, -1, 1, 1))

  def forward(self, x):
    self.x_server = x
    b, c, h, w = x.shape
    assert(c == self.channels)
    # calc stats
    mean = np.mean(x, (0, 2, 3))
    centered = x - self.reshape4d(mean)
    self.centered = centered
    centered_flattened = centered.flatten()
    centered_flattened_enc = self.crypto.deserialize(self.comm.recv())
    self.crypto.vector_add_plain(centered_flattened_enc, centered_flattened)
    self.crypto.vector_square(centered_flattened_enc)
    r = random_tensor(centered_flattened_enc.shape)
    self.crypto.vector_add_plain(centered_flattened_enc, -r)
    self.comm.send(self.crypto.serialize(centered_flattened_enc))
    squared = np.reshape(r, x.shape)
    variance = np.mean(squared, (0, 2, 3))
    variance += self.comm.recv() + self.eps
    mean += self.comm.recv()
    self.current_mean = mean
    self.current_var = variance
    # update running stats
    if self.is_training:
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.current_mean
      self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * self.current_var
    # calc normalized input
    mean = self.current_mean if self.is_training else self.running_mean
    var =  self.current_var  if self.is_training else self.running_var
    istd = 1 / np.sqrt(var + self.eps)
    if self.affine:
      k = self.weight.value * istd
      b = -mean * istd * self.weight.value + self.bias.value
    else:
      k = istd
      b = -mean * istd
    self.comm.send(k)
    y = x * self.reshape4d(k) + self.reshape4d(b)
    return y

  def forward_plain(self, x):
    if self.is_training:
      print("Warning: forward_plain called in training mode. Result may differ from torch.")
    # this won't update running stats
    istd = 1 / np.sqrt(self.eps + self.running_var)
    if self.affine:
      k = self.weight.value * istd
      b = -self.running_mean * istd * self.weight.value + self.bias.value
    else:
      k = istd
      b = -self.running_mean * istd
    y = x * self.reshape4d(k) + self.reshape4d(b)
    return y 
  
  def describe(self):
    return {
      "name": "BatchNorm2d",
      "channels": self.channels,
      "eps": self.eps,
      "momentum": self.momentum,
      "affine": self.affine
    }

  def backward_calculate_partial_A(self, partial_y_server, istd):
    assert(self.prepared)
    prep = self.preprocessed
    def linear_operation(x, y):
      return np.sum(x*y, (0,2,3))

    # 0. Send standard deviation
    self.comm.send(istd)
    x_hat_server = self.centered * self.reshape4d(istd)
    
    # 1. Receive noised x0 - kx * rx, y0 - ky * ry, kx, ky
    noised_x0 = self.recv()
    noised_y0 = self.recv()
    kx, ky = self.recv()

    # 2a. Server chooses scalar lx, ly, tensor sx, sy ~ z
    z_shape = (self.channels,)
    lx, ly = random_scalar(), random_scalar()
    sx = random_tensor(z_shape)
    sy = random_tensor(z_shape)

    # 2b: Server computes
    #   m = noised_x0 ∘ (Wx * lx) + sx + (Wy * ly) ∘ noised_y0 + sy
    #   [hx] = [px] * kx * lx - sx
    #   [hy] = [py] * ky * ly - sy
    #   noised_x1 = x1 + Wy * ly
    #   noised_y1 = y1 + Wx * lx
    merged = (
      + linear_operation(noised_x0, prep["Wx"] * lx)
      + linear_operation(prep["Wy"] * ly, noised_y0)
      + sx + sy
    )
    hx_enc = self.mul_scalar(prep["px"], kx * lx)
    hy_enc = self.mul_scalar(prep["py"], ky * ly)
    self.add_plain_inplace(hx_enc, -sx.flatten())
    self.add_plain_inplace(hy_enc, -sy.flatten())
    noised_x1 = x_hat_server + prep["Wy"] * ly
    noised_y1 = partial_y_server + prep["Wx"] * lx
    
    # 2c: Server sends these five terms
    self.send(merged)
    self.send_ciphers(hx_enc)
    self.send_ciphers(hy_enc)
    self.send(noised_x1)
    self.send(noised_y1)
    
    # 4. Server finished by calculating x ∘ y = denoised + x1 ∘ y1
    denoised = self.recv()
    partial_A = denoised + linear_operation(x_hat_server, partial_y_server)

    if self.affine:
      self.weight.set_grad(partial_A)

    return partial_A

  def backward(self, partial_y_server):
    batchsize, _, h, w = self.x_server.shape
    n = batchsize * h * w
    istd = 1 / np.sqrt(self.eps + self.current_var)

    # partial_A
    partial_A = self.backward_calculate_partial_A(partial_y_server, istd)
    
    # partial_b
    partial_b = np.sum(partial_y_server, (0, 2, 3))
    partial_b += self.comm.recv()
    if self.affine:
      self.bias.set_grad(partial_b)

    # https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html

    # partial_x
    k = istd
    if self.affine: k *= self.weight.value
    x_hat = self.crypto.deserialize(self.comm.recv())
    partial_y = self.crypto.deserialize(self.comm.recv())
    self.crypto.vector_add_plain(x_hat, -self.centered.flatten() / n)
    self.crypto.vector_add_plain(partial_y, partial_y_server.flatten())
    def repeat_n(x):
      x = self.reshape4d(x)
      x = np.repeat(x, batchsize, axis=0)
      x = np.repeat(x, h, axis=2)
      x = np.repeat(x, w, axis=3)
      return x
    partial_A_repeated = repeat_n(partial_A * istd).flatten()
    self.crypto.vector_mul_plain(x_hat, partial_A_repeated)
    self.crypto.vector_add(x_hat, partial_y)
    r = random_tensor(x_hat.shape)
    self.crypto.vector_add_plain(x_hat, r)
    self.comm.send(k)
    self.comm.send(self.crypto.serialize(x_hat))
    partial_x = -repeat_n(partial_b) / n - np.reshape(r, self.x_server.shape)
    partial_x = self.reshape4d(k) * partial_x
    return partial_x


  def parameters(self):
    if self.affine: return [self.weight, self.bias]
    else: return []
  
  def to_torch(self):
    torch_module = torch.nn.BatchNorm2d(self.channels, self.eps, self.momentum, self.affine)
    if self.affine:
      torch_module.weight.data = torch.tensor(self.weight.value, dtype=torch.float32)
      torch_module.bias.data = torch.tensor(self.bias.value, dtype=torch.float32)
    torch_module.running_mean = torch.tensor(self.running_mean, dtype=torch.float32)
    torch_module.running_var = torch.tensor(self.running_var, dtype=torch.float32)
    return torch_module 


  def static_prepare(self, input_shape):
    batchsize, channels, h, w = input_shape
    istd = 1 / np.sqrt(self.eps + self.running_var)
    k = self.weight.value / istd
    b = -k * self.running_mean + self.bias.value
    b = self.reshape4d(b)
    self.encoded_b = b
    shape = (input_shape[1], input_shape[2], input_shape[3])
    r_enc = self.crypto.deserialize(self.comm.recv())
    k = np.reshape(self.weight.value / istd, (-1, 1, 1))
    k = np.repeat(k, h, axis=1)
    k = np.repeat(k, w, axis=2).flatten()
    self.crypto.vector_mul_plain(r_enc, k)
    s = random_tensor(shape)
    self.crypto.vector_add_plain(r_enc, s.flatten())
    self.comm.send(self.crypto.serialize(r_enc))
    
    p = self.crypto.deserialize(self.comm.recv())
    self.crypto.vector_add_plain(p, -s.flatten())
    self.encoded_k = self.reshape4d(self.weight.value / istd)
    self.delphi_p = p
    return input_shape


  def static_forward(self, x_server):
    noised_x0 = self.recv()
    k = self.recv()
    
    # 2. Server picks s ~ y for each sample and calculates
    #   [h] = [p] * k + s
    #   y1 = W (noised_x0 + x1) - s + b
    y_shape = x_server.shape
    s = random_tensor(y_shape)
    h_encs = []
    for i, ki in enumerate(k):
      h_enc = self.mul_scalar(self.delphi_p, ki)
      self.add_plain_inplace(h_enc, s[i].flatten())
      h_encs.append(h_enc)
    y1 = (noised_x0 + x_server) * self.encoded_k + self.encoded_b - s
    
    # 3. Server send [h]
    h_encs = [self.crypto.serialize(i) for i in h_encs]
    self.send(h_encs)
    return y1

  def prepare(self, input_shape):
    print(f"Preparing BatchNorm2d {input_shape}", flush=True)
    batchsize, channels, h, w = input_shape
    assert(channels == self.channels)

    describe = f"pd-bn2d-server-{batchsize}-{channels}-{h}-{w}"
    p = hashlib.sha256(describe.encode()).hexdigest()[:8]
    filename = f"preprocess/{p}.pickle"
    
    if os.path.exists(filename) and not config.FORCE_REGENERATE_PREPARE:

      print(f"-- loading {p}")
      f = open(filename, "rb")
      self.preprocessed = pickle.load(f)
      px = self.recv_ciphers()
      py = self.recv_ciphers()
      self.crypto.vector_add_plain(px, self.preprocessed["px_share"])
      self.crypto.vector_add_plain(py, self.preprocessed["py_share"])
      self.preprocessed["px"] = px
      self.preprocessed["py"] = py
      f.close()
    
    else:

      print(f"-- generating {p}")

      y_shape = input_shape
      x_shape = input_shape
      z_shape = input_shape

      # 2. Server chooses Wx ~ y, Wy ~ x
      Wx = random_tensor(y_shape)
      Wy = random_tensor(x_shape)
      
      # 3. Server computes px = x ∘ Wx, py = Wy ∘ y
      rx_enc = self.recv_ciphers()
      ry_enc = self.recv_ciphers()
      self.crypto.vector_mul_plain(rx_enc, Wx.flatten())
      self.crypto.vector_mul_plain(ry_enc, Wy.flatten())
      px_enc = rx_enc 
      py_enc = ry_enc

      # 4. Add noise to px, py
      sx = random_tensor(x_shape)
      sy = random_tensor(y_shape)
      self.crypto.vector_add_plain(px_enc, sx.flatten())
      self.crypto.vector_add_plain(py_enc, sy.flatten())
      self.send_ciphers(px_enc)
      self.send_ciphers(py_enc)

      # 6. Reconstruct px, py as flattened ciphertexts
      px = self.recv_ciphers()
      py = self.recv_ciphers()
      sx = np.sum(sx, axis=(0,2,3))
      sy = np.sum(sy, axis=(0,2,3))
      self.crypto.vector_add_plain(px, -sx.flatten())
      self.crypto.vector_add_plain(py, -sy.flatten())
      self.preprocessed = {
        "px": px, # [C]
        "py": py, # [C]
        "Wx": Wx, # [Inputshape]
        "Wy": Wy, # [Inputshape]
      }

      file = open(filename, "wb")
      pickle.dump({
        "describe": describe,
        "Wx": Wx, # [Inputshape]
        "Wy": Wy, # [Inputshape]
        "px_share": -sx.flatten(),
        "py_share": -sy.flatten(),
      }, file)
      file.close()

    self.prepared = True
    return input_shape

  def trusted_prepare(self, input_shape, save_file=True):    

    batchsize, channels, h, w = input_shape
    assert(channels == self.channels)
    y_shape = input_shape
    x_shape = input_shape
    z_shape = input_shape
    Wx = random_tensor(y_shape)
    Wy = random_tensor(x_shape)
    sx = np.sum(random_tensor(z_shape), (0,2,3))
    sy = np.sum(random_tensor(z_shape), (0,2,3))
    rx = random_tensor(x_shape)
    ry = random_tensor(y_shape)
    hx = np.sum(Wx * rx, (0,2,3)) + sx
    hy = np.sum(ry * Wy, (0,2,3)) + sy

    describe = f"pd-bn2d-server-{batchsize}-{channels}-{h}-{w}"
    p = hashlib.sha256(describe.encode()).hexdigest()[:8]
    filename1 = f"preprocess/{p}.pickle"
    preprocessed = {
      "describe": describe,
      "px_share": -sx.flatten(), 
      "py_share": -sy.flatten(), 
      "Wx": Wx,  
      "Wy": Wy   
    }
    if save_file:
      file = open(filename1, "wb")
      pickle.dump(preprocessed, file)
      file.close()
    
    describe = f"pd-bn2d-client-{batchsize}-{channels}-{h}-{w}"
    p = hashlib.sha256(describe.encode()).hexdigest()[:8]
    filename2 = f"preprocess/{p}.pickle"
    preprocessed = {
      "describe": describe,
      "rx": rx,
      "ry": ry,
      "px_share": hx.flatten(),
      "py_share": hy.flatten()
    }
    if save_file:
      file = open(filename2, "wb")
      pickle.dump(preprocessed, file)
      file.close()    
    
    print(
      f"BatchNorm2d: {input_shape}"
    )
    if save_file:
      print(f"   Server {filename1}, Client {filename2}")

    return input_shape

class Dropout(ServerModule):
  
  def __init__(self, crypto, comm, rate=0.5):
    super().__init__()
    self.comm = comm
    self.crypto = crypto
    self.rate = rate
    
  def static_prepare(self, shape): return shape
  def static_forward(self, x): return x

  def forward(self, x):
    if self.is_training:
      self.choices = np.random.choice([0, 1], x.shape, p=[self.rate, 1-self.rate])
      self.comm.send([True, self.choices])
      return x*self.choices
    else:
      self.comm.send([False, None])
      return x

  def forward_plain(self, x):
    if self.is_training:
      choices = np.random.choice([0, 1], x.shape, p=[self.rate, 1-self.rate])
      return x*choices
    else:
      return x

  def backward(self, dy):
    return dy * self.choices
  
  def describe(self):
    return {
      "name": "Dropout",
      "rate": self.rate
    }

  def parameters(self): return []
  def prepare(self, s): return s
  def trusted_prepare(self, s): return s
  def to_torch(self): return torch.nn.Dropout(self.rate)

class Sequential(ServerModule):

  def __init__(self, crypto, comm, modules):
    super().__init__()
    self.items = modules
    self.crypto = crypto
    self.comm = comm

  def forward(self, x):
    n = len(self.items)
    print("Forward:  " + "." * n, end="")
    for i, each in enumerate(self.items):
      x = each.forward(x)
      print("\rForward:  " + "-" * (i+1) + "." * (n-i-1), end="")
      # self.comm.send(x)
    print("")
    return x

  def static_forward(self, x):
    n = len(self.items)
    print("Forward:  " + "." * n, end="")
    for i, each in enumerate(self.items):
      x = each.static_forward(x)
      print("\rForward:  " + "-" * (i+1) + "." * (n-i-1), end="")
    print("")
    return x

  def forward_plain(self, x):
    n = len(self.items)
    # print("Forward:  " + "." * n, end="")
    for i, each in enumerate(self.items):
      x = each.forward_plain(x)
      # print("\rForward:  " + "-" * (i+1) + "." * (n-i-1), end="")
    # print("")
    return x

  def backward(self, partial_y):
    n = len(self.items)
    print("Backward: " + "." * n, end="")
    for i in range(n):
      partial_y = self.items[n - i - 1].backward(partial_y)
      print("\rBackward: " + "." * (n-i-1) + "-" * (i+1), end="")
    print("")
    return partial_y

  def backward_plain(self, partial_y):
    n = len(self.items)
    # print("Backward plain: " + "." * n, end="")
    for i in range(n):
      partial_y = self.items[n - i - 1].backward_plain(partial_y)
      # print("\rBackward plain: " + "." * (n-i-1) + "-" * (i+1), end="")
    # print("")
    return partial_y


  def describe(self):
    return {
      "name": "Sequential",
      "modules": [item.describe() for item in self.items]
    }

  def parameters(self):
    ret = []
    for each in self.items:
      ret += each.parameters()
    return ret

  def to_torch(self):
    items = [i.to_torch() for i in self.items]
    return torch.nn.Sequential(*items)

  def prepare(self, input_shape):
    for each in self.items:
      # print(input_shape, end="")
      input_shape = each.prepare(input_shape)
      # print(" ->", input_shape)
    return input_shape

  def trusted_prepare(self, input_shape):
    original_shape = input_shape
    for each in self.items:
      # print(input_shape)
      input_shape = each.trusted_prepare(input_shape)
    print(f"Sequential: {original_shape} -> {input_shape}")
    return input_shape

  def train(self): 
    self.is_training = True
    for each in self.items: each.train()
  
  def eval(self):
    self.is_training = False
    for each in self.items: each.eval()

class TorchNative(ServerModule):

  def __init__(self, crypto, comm, preset, device):
    super().__init__()
    self.crypto = crypto
    self.comm = comm
    self.device = device
    self.model = torch_models.load_model(preset).to(device)
    self.model.eval()
    self.preset = preset
    self.static_forward = self.forward
    self.static_prepare = self.prepare

  def forward(self, x): 
    # x must be 0
    return np.zeros(self.y_shape, dtype=np.uint64).flatten()

  def forward_plain(self, x): 
    x = self.crypto.to_decimal(x, shape=self.x_shape)
    x = torch.tensor(x, dtype=torch.float32, device=self.device)
    y = self.model(x).detach().cpu().numpy().copy()
    y = self.crypto.to_field(y)
    return y

  def describe(self): 
    return {
      "name": "TorchNative",
      "preset": self.preset, 
      "device": self.device
    }

  def backward(self, partial_y): 
    return None

  def backward_plain(self, partial_y):
    return None

  def parameters(self):
    return []

  def to_torch(self): 
    return self.model

  def prepare(self, input_shape): 
    self.x_shape = input_shape
    x = torch.tensor(np.zeros(input_shape), 
      dtype=torch.float32, device=self.device)
    y = self.model(x)
    self.y_shape = y.shape
    return y.shape

  def trusted_prepare(self, input_shape):
    x = torch.tensor(np.zeros(input_shape), 
      dtype=torch.float32, device=self.device)
    y = self.model(x)
    self.y_shape = y.shape
    return y.shape

def server_model_from_pytorch(torch_model, crypto, comm):

  if isinstance(torch_model, torch.nn.ReLU):
    return ReLU(crypto, comm)

  elif isinstance(torch_model, torch.nn.Flatten):
    return Flatten()

  elif isinstance(torch_model, torch.nn.Linear):
    ret = Linear(crypto, comm, 
      torch_model.weight.shape[1], torch_model.weight.shape[0])
    ret.weight.value = torch_model.weight.detach().cpu().numpy().copy()
    ret.bias.value = torch_model.bias.detach().cpu().numpy().copy()
    return ret

  elif isinstance(torch_model, torch.nn.Conv2d):
    stride = torch_model.stride[0]
    pad = torch_model.padding[0]
    if pad > 0:
      ret = Conv2dPadded(crypto, comm,
        torch_model.weight.shape[1],
        torch_model.weight.shape[0],
        (torch_model.weight.shape[2], torch_model.weight.shape[3]),
        stride, pad,
        dont_create_conv=True)
      ret.create_convs(
        torch_model.weight.detach().cpu().numpy().copy(),
        torch_model.bias.detach().cpu().numpy().copy()
      )
      return ret
    elif stride > 1:
      ret = Conv2dStrided(crypto, comm,
        torch_model.weight.shape[1],
        torch_model.weight.shape[0],
        (torch_model.weight.shape[2], torch_model.weight.shape[3]),
        stride,
        dont_create_conv=True)
      ret.create_convs(
        torch_model.weight.detach().cpu().numpy().copy(),
        torch_model.bias.detach().cpu().numpy().copy()
      )
      return ret
    else:
      ret = Conv2d(crypto, comm,
        torch_model.weight.shape[1],
        torch_model.weight.shape[0],
        (torch_model.weight.shape[2], torch_model.weight.shape[3])
      )
      ret.weight.value = torch_model.weight.detach().cpu().numpy().copy()
      ret.bias.value = torch_model.bias.detach().cpu().numpy().copy()
      return ret

  elif isinstance(torch_model, torch.nn.Conv1d):
    stride = torch_model.stride[0]
    pad = torch_model.padding[0]
    assert(pad == 0)
    assert(stride == 1)

    input_channels = torch_model.weight.shape[1]
    output_channels = torch_model.weight.shape[0]
    kernel_size = torch_model.weight.shape[2]

    ret = Conv1d(crypto, comm,
      input_channels,
      output_channels,
      kernel_size,
    )

    ret.inner.weight.value = (
      torch_model.weight.detach().cpu().numpy().copy()
      .reshape((output_channels, input_channels, kernel_size, 1))
    )
    ret.inner.bias.value = torch_model.bias.detach().cpu().numpy().copy()
    return ret

  elif isinstance(torch_model, torch.nn.Sequential):
    modules = []
    for each in torch_model.children():
      modules.append(server_model_from_pytorch(each, crypto, comm))
    return Sequential(crypto, comm, modules)

  elif isinstance(torch_model, torch.nn.AvgPool2d):
    return AvgPool2d(crypto, comm, 
      torch_model.kernel_size,
      torch_model.stride,
      torch_model.padding
    )

  elif isinstance(torch_model, torch.nn.MaxPool2d):
    print("Warning: MaxPool2d converted to AvgPool2d")
    return AvgPool2d(crypto, comm, 
      torch_model.kernel_size,
      torch_model.stride,
      torch_model.padding
    )
  
  elif isinstance(torch_model, torch.nn.AvgPool1d):
    return AvgPool1d(crypto, comm, 
      torch_model.kernel_size[0],
      torch_model.stride[0],
      torch_model.padding[0]
    )

  elif isinstance(torch_model, torch.nn.MaxPool1d):
    print("Warning: MaxPool1d converted to AvgPool1d")
    return AvgPool1d(crypto, comm, 
      torch_model.kernel_size[0],
      torch_model.stride[0],
      torch_model.padding[0]
    )

  elif isinstance(torch_model, torch.nn.BatchNorm2d):
    ret = BatchNorm2d(crypto, comm, 
      torch_model.num_features,
      torch_model.eps,
      torch_model.momentum,
      torch_model.affine)
    if torch_model.affine:
      ret.weight.value = torch_model.weight.detach().cpu().numpy().copy()
      ret.bias.value = torch_model.bias.detach().cpu().numpy().copy()
    ret.running_mean = torch_model.running_mean.detach().cpu().numpy().copy()
    ret.running_var = torch_model.running_var.detach().cpu().numpy().copy()
    return ret

  elif isinstance(torch_model, torch.nn.Dropout):
    return Dropout(crypto, comm, torch_model.p)

  elif isinstance(torch_model, torch_models.TorchNative):
    return TorchNative(crypto, comm, torch_model.preset, config.DEVICE)

  else:
    raise Exception("cannot convert from pytorch -- unsupported layer type")