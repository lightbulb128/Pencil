import numpy as np
import config
from crypto_switch import cryptography as crypto
import communication_server
import math
import torch
import torch.nn
import torch.nn.functional
import os, hashlib, pickle
import torch_models
import time

# server: 0 with parameters
# client: 1 with data

def ceil_div(a, b):
  if a%b==0: return a//b
  return a//b+1

def random_tensor(shape, bound=config.RANDOM_TENSOR_BOUND):
  return np.random.uniform(-bound, bound, shape)
def random_scalar(bound = 1) -> float:
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
  def trusted_prepare(self, input_shape, save_file=True):
    raise Exception("not implemented")
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
  def trusted_prepare(self, input_shape, save_file=True): 
    print(f"ReLU: {input_shape}")
    return input_shape

class GELU(ServerModule):

  def __init__(self, crypto: crypto.EvaluationUtils, communication: communication_server.ServerCommunication):
    super().__init__()
    self.comm = communication
    self.crypto = crypto

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
    const3 = self.crypto.to_field(np.array(1))
    y = self.crypto.field_add(z, const3)
    const4 = self.crypto.to_field(np.array(0.5))
    y = self.crypto.field_mod(const4 * y)
    y = self.comm.truncate(y)
    y = self.comm.elementwise_multiply(x, y)
    y = self.comm.truncate(y)
    return y
  
  def backward(self, partial_y):
    assert False, "not implemented"

  def parameters(self): return []
  
  def describe(self):
    return {"name": "GELU"}

  def to_torch(self): return torch.nn.GELU()

  def prepare(self, input_shape): return input_shape

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
    return (input_shape[0], np.prod(input_shape[1:]))

class Linear(ServerModule):

  def __init__(self, crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, input_dims, output_dims):
    super().__init__()
    self.input_dims = input_dims
    self.output_dims = output_dims
    self.weight = Parameter(np.random.uniform(-1, 1, (output_dims, input_dims)) / math.sqrt(input_dims))
    self.bias = Parameter(np.random.uniform(-1, 1, (output_dims,)) / math.sqrt(input_dims))
    self.comm = comm
    self.crypto = crypto
    self.prepared = False

  def collaborate_matmul(self, helper, x_server, weight, output_size):
    weight = self.crypto.to_field(weight)
    x_server_encoded = self.crypto.matmul_encode_x(helper, x_server)
    w_encoded = self.crypto.matmul_encode_w(helper, weight)
    x_cipher = self.crypto.deserialize(self.recv())
    self.crypto.add_plain_inplace(x_cipher, x_server_encoded)
    y_client_cipher = self.crypto.matmul(helper, x_cipher, w_encoded)
    s = self.crypto.field_random_mask(output_size)
    s_encoded = self.crypto.matmul_encode_y(helper, s)
    self.crypto.add_plain_inplace(y_client_cipher, s_encoded)
    self.send(self.crypto.matmul_serialize_y(helper, y_client_cipher))
    return self.crypto.field_negate(s)

  def forward(self, x):
    self.x_server = x
    y = self.collaborate_matmul(self.helper_forward, x, self.weight.value.transpose(), self.batchsize * self.output_dims)
    
    # add bias
    bias = self.crypto.to_field(self.bias.value, self.crypto.default_scale()**2)
    bias = np.reshape(bias, (1, self.output_dims))
    bias = np.repeat(bias, self.batchsize, axis=0).flatten()
    y = self.crypto.field_add(y, bias)
    return y

  def forward_plain(self, x):
    x = np.reshape(x, (self.batchsize, self.input_dims))
    self.x_server = x
    w = self.crypto.to_field(self.weight.value.transpose(), flatten=False)
    y = self.crypto.field_mod(np.matmul(x, w))
    return y.flatten()

  def prepare(self, input_shape):
    self.prepared = True
    self.batchsize = input_shape[0]
    self.helper_forward = self.crypto.matmul_helper(input_shape[0], self.input_dims, self.output_dims, objective=0)
    self.helper_backward = self.crypto.matmul_helper(input_shape[0], self.output_dims, self.input_dims, objective=0)
    self.helper_weights = self.crypto.matmul_helper(self.output_dims, input_shape[0], self.input_dims, objective=2)
    return (input_shape[0], self.output_dims)

  def backward_calculate_partial_A(self, partial_y_server):
    def linear_operation(y, x):
      x = np.reshape(x, (self.batchsize, self.input_dims))
      y = np.reshape(y, (self.output_dims, self.batchsize))
      return self.crypto.field_mod(np.matmul(y, x).flatten())

    helper = self.helper_weights

    partial_y_server = np.transpose(np.reshape(partial_y_server, (self.batchsize, self.output_dims)), (1, 0)).flatten()
    partial_y_server_encoded = self.crypto.matmul_encode_x(helper, partial_y_server)
    partial_y_client_cipher = self.crypto.deserialize(self.recv())

    x_server_encoded = self.crypto.matmul_encode_w(helper, self.x_server)
    x_client_cipher = self.crypto.deserialize(self.recv())

    w = self.crypto.matmul(helper, partial_y_client_cipher, x_server_encoded)
    w2 = self.crypto.matmul(helper, partial_y_server_encoded, x_client_cipher)
    self.crypto.add_inplace(w, w2)
    s = self.crypto.field_random_mask(self.output_dims * self.input_dims)
    s_encoded = self.crypto.matmul_encode_y(helper, s)
    self.crypto.add_plain_inplace(w, s_encoded)
    self.send(self.crypto.matmul_serialize_y(helper, w))

    partial_w = self.recv()
    partial_w = self.crypto.field_add(partial_w, self.crypto.field_negate(s))
    partial_w = self.crypto.field_add(partial_w, linear_operation(partial_y_server, self.x_server))
    partial_w = self.crypto.to_decimal(partial_w, self.crypto.default_scale()**2, (self.output_dims, self.input_dims))
    self.weight.set_grad(partial_w)

  def backward(self, partial_y_server):
    # calculate partial_A
    self.backward_calculate_partial_A(partial_y_server)

    # calculate partial_b
    partial_b = self.comm.recv() + np.sum(np.reshape(partial_y_server, (self.batchsize, self.output_dims)), axis=0)
    partial_b = self.crypto.to_decimal(self.crypto.field_mod(partial_b))
    self.bias.set_grad(partial_b)

    # calculate partial_x shares
    # print(self.batchsize, self.output_dims, len(partial_y_server))
    partial_x = self.collaborate_matmul(self.helper_backward, partial_y_server, self.weight.value, self.batchsize * self.input_dims)
    if config.TRUNCATE_BACKWARD:
      partial_x = self.comm.truncate(partial_x)
    else:
      print("WARNING: Backward gradient not truncated")
    return partial_x

  def backward_plain(self, partial_y_server):
    partial_y = np.reshape(partial_y_server, (self.batchsize, self.output_dims))
    
    # calculate partial_A
    partial_w = self.crypto.field_mod(np.matmul(np.transpose(partial_y, (1, 0)), self.x_server))
    partial_w = self.crypto.to_decimal(partial_w, self.crypto.default_scale()**2)
    self.weight.set_grad(partial_w)

    # calculate partial_b
    partial_b = np.sum(partial_y, axis=0)
    partial_b = self.crypto.to_decimal(self.crypto.field_mod(partial_b))
    self.bias.set_grad(partial_b)
    
    # calculate partial_x
    partial_x = self.crypto.field_mod(np.matmul(partial_y, self.crypto.to_field(self.weight.value, flatten=False))).flatten()
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

  def collaborate_conv2d(self, helper, x_server, weight, output_size, x_cipher=None):
    weight = self.crypto.to_field(weight)
    w_encoded = self.crypto.conv2d_encode_w(helper, weight)

    if x_cipher is None:
      x_server_encoded = self.crypto.conv2d_encode_x(helper, x_server)
      x_cipher = self.crypto.deserialize(self.recv())
      self.crypto.add_plain_inplace(x_cipher, x_server_encoded)

    y_client_cipher = self.crypto.conv2d(helper, x_cipher, w_encoded)
    s = self.crypto.field_random_mask(output_size)
    s_encoded = self.crypto.conv2d_encode_y(helper, s)
    self.crypto.add_plain_inplace(y_client_cipher, s_encoded)
    self.send(self.crypto.conv2d_serialize_y(helper, y_client_cipher))
    return self.crypto.field_negate(s)

  def collaborate_conv2d_accumulate_ciphers(self, helper, x_server, weight):
    weight = self.crypto.to_field(weight)
    x_server_encoded = self.crypto.conv2d_encode_x(helper, x_server)
    w_encoded = self.crypto.conv2d_encode_w(helper, weight)
    x_cipher = self.crypto.deserialize(self.recv())
    self.crypto.add_plain_inplace(x_cipher, x_server_encoded)
    y_client_cipher = self.crypto.conv2d(helper, x_cipher, w_encoded)
    return y_client_cipher

  def forward(self, x):
    self.x_server = x
    y_shape = self.output_shape
    s_size = y_shape[0] * y_shape[1] * y_shape[2] * y_shape[3]
    y_server = self.collaborate_conv2d(self.helper_forward, x, self.weight.value, s_size)
    
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

  def forward_accumulate_ciphers(self, x):
    self.x_server = x
    y_server = self.collaborate_conv2d_accumulate_ciphers(self.helper_forward, x, self.weight.value)
    return y_server

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

  def backward_calculate_partial_A(self, partial_y_server, partial_y_server_encoded=None, partial_y_client_cipher=None):
    def linear_operation(x, y):
      return self.crypto.field_mod(self.conv2d_plain(x, y))
      
    helper = self.helper_weights

    partial_y_server = np.transpose(partial_y_server, (1, 0, 2, 3))
    if partial_y_server_encoded is None:
      partial_y_server_encoded = self.crypto.conv2d_encode_w(helper, partial_y_server.flatten())
      partial_y_client_cipher = self.crypto.deserialize(self.recv())

    x_server = np.transpose(np.reshape(self.x_server, self.input_shape), (1, 0, 2, 3))
    x_server_encoded = self.crypto.conv2d_encode_x(helper, x_server.flatten())
    x_client_cipher = self.crypto.deserialize(self.recv())

    w = self.crypto.conv2d(helper, x_server_encoded, partial_y_client_cipher)
    w2 = self.crypto.conv2d(helper, x_client_cipher, partial_y_server_encoded)
    self.crypto.add_inplace(w, w2)
    s = self.crypto.field_random_mask(self.output_channels * self.input_channels * self.kernel_size[0] * self.kernel_size[1])
    s_encoded = self.crypto.conv2d_encode_y(helper, s)
    self.crypto.add_plain_inplace(w, s_encoded)
    self.send(self.crypto.conv2d_serialize_y(helper, w))

    partial_w = self.recv()
    s = np.reshape(s, (self.input_channels, self.output_channels, self.kernel_size[0], self.kernel_size[1]))
    partial_w = self.crypto.field_add(partial_w, self.crypto.field_negate(s))
    partial_w = self.crypto.field_add(partial_w, linear_operation(x_server, partial_y_server))
    partial_w = self.crypto.to_decimal(partial_w, self.crypto.default_scale()**2)
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
    s_size = batchsize * self.input_channels * x_h * x_w
    partial_x_server = self.collaborate_conv2d(
      self.helper_backward,
      padded_partial_y,
      np.transpose(np.flip(self.weight.value, (2,3)), (1,0,2,3)),
      s_size
    )
    if config.TRUNCATE_BACKWARD:
      partial_x_server = self.comm.truncate(partial_x_server)
    else:
      print("WARNING: Backward gradient not truncated")
    return partial_x_server

  def backward_accumulate_ciphers(self, 
    partial_y_server,
    partial_y_server_encoded_for_weights,
    partial_y_client_cipher_for_weights,
    partial_y_cipher_for_x
  ):

    batchsize, _, x_h, x_w = self.input_shape
    _, _, y_h, y_w = self.output_shape
    k_h, k_w = self.kernel_size
    partial_y_server = np.reshape(partial_y_server, self.output_shape)

    # calculate partial_A
    self.backward_calculate_partial_A(partial_y_server, 
      partial_y_server_encoded_for_weights, 
      partial_y_client_cipher_for_weights)

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
    s_size = batchsize * self.input_channels * x_h * x_w
    partial_x_server = self.collaborate_conv2d(
      self.helper_backward,
      padded_partial_y,
      np.transpose(np.flip(self.weight.value, (2,3)), (1,0,2,3)),
      s_size,
      partial_y_cipher_for_x
    )
    if config.TRUNCATE_BACKWARD:
      partial_x_server = self.comm.truncate(partial_x_server)
    else:
      print("WARNING: Backward gradient not truncated")
    return partial_x_server


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
  
  def conv2d_clip(self, x, margin):
    _,_, c, d = x.shape
    k_h, k_w = margin
    return x[:, :, :c-k_h+1, :d-k_w+1].copy()

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
        y_item = self.convs[i][j].forward_accumulate_ciphers(x_item.flatten())
        if y_server is None:
          y_server = y_item
        else: 
          self.crypto.add_inplace(y_server, y_item)
    # send ciphers
    yshape = self.output_shape
    s_size = yshape[0] * yshape[1] * yshape[2] * yshape[3]

    helper = self.convs[0][0].helper_forward
    
    s = self.crypto.field_random_mask(s_size)
    s_encoded = self.crypto.conv2d_encode_y(helper, s)
    self.crypto.add_plain_inplace(y_server, s_encoded)
    self.send(self.crypto.conv2d_serialize_y(helper, y_server))
    s = self.crypto.field_negate(s)

    if self.has_bias:
      bias = self.crypto.to_field(self.convs[0][0].bias.value, self.crypto.default_scale()**2)
      batchsize, _, y_h, y_w = self.output_shape
      bias = np.reshape(bias, (1, self.output_channels, 1))
      bias = np.repeat(bias, batchsize, axis=0)
      bias = np.repeat(bias, y_h * y_w, axis=2).flatten()
      s = self.crypto.field_add(s, bias)
    return s

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

    helper_backward = self.convs[0][0].helper_backward
    helper_weights = self.convs[0][0].helper_weights
    
    partial_y_server_reshaped = np.reshape(partial_y_server, self.output_shape)
    partial_y_server_encoded_for_weights = self.crypto.conv2d_encode_w(helper_weights, np.transpose(partial_y_server_reshaped, (1, 0, 2, 3)).flatten())
    partial_y_client_cipher_for_weights = self.crypto.deserialize(self.recv())

    batchsize, _, y_h, y_w = self.convs[0][0].output_shape
    k_h, k_w = self.convs[0][0].kernel_size
    p_h, p_w = k_h - 1, k_w - 1
    padded_partial_y = np.zeros((batchsize, self.output_channels, y_h + p_h*2, y_w + p_w*2), dtype=np.uint64)
    padded_partial_y[:,:,p_h:p_h+y_h,p_w:p_w+y_w] = partial_y_server_reshaped
    padded_partial_y = padded_partial_y.flatten()
    padded_partial_y_server_encoded = self.crypto.conv2d_encode_x(helper_backward, padded_partial_y)
    padded_partial_y_cipher_for_x = self.crypto.deserialize(self.recv())
    self.crypto.add_plain_inplace(padded_partial_y_cipher_for_x, padded_partial_y_server_encoded)

    partial_x_split = []
    for i, conv_row in enumerate(self.convs):
      partial_x_row = []
      for j, conv_item in enumerate(conv_row):
        partial_x_item = conv_item.backward_accumulate_ciphers(
          partial_y_server,
          partial_y_server_encoded_for_weights,
          partial_y_client_cipher_for_weights,
          padded_partial_y_cipher_for_x
        )
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
    new_x = np.zeros((i, o, nh, nw))
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

  def prepare(self, input_shape): 
    self.input_shape = input_shape
    self.batchsize = input_shape[0]
    b, i, h, w = input_shape
    nh = h + self.padding * 2
    nw = w + self.padding * 2
    self.output_shape = self.inner.prepare((b, i, nh, nw))
    return self.output_shape

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
  
  def backward_plain(self, partial_y):
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
    partial_x = self.crypto.divide_plain(partial_x, kernel_size * kernel_size)
    return partial_x

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

class Softmax(ServerModule):

  def __init__(self, crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication):
    super().__init__()
    self.comm = comm
    self.crypto = crypto

  def prepare(self, input_shape):
    self.shape = input_shape
    return input_shape
  
  def forward(self, x):
    x = self.comm.softmax(x, self.shape[-1])
    x = self.comm.truncate(x)
    self.y = x.reshape(self.shape)
    return x
    # self.send(x)
    # return self.recv()
  
  def forward_plain(self, x):
    x = torch.tensor(x)
    x = torch.nn.functional.softmax(x, dim=-1)
    return x
  
  def backward(self, dy):
    # self.send(dy)
    # return self.recv()

    dy = dy.reshape(self.shape)
    prod = self.comm.elementwise_multiply(dy, self.y)
    prod = self.comm.truncate(prod)
    summed = self.crypto.field_mod(np.sum(prod, axis=-1, keepdims=True))
    sub = self.crypto.field_mod(dy - summed)
    dx = self.comm.elementwise_multiply(sub, self.y)
    dx = self.comm.truncate(dx)
    return dx.flatten()
    


  def describe(self): return {"name": "Softmax"}
  def parameters(self): return []
  def to_torch(self): return torch.nn.Softmax(dim=-1)

class ScaledDotProductAttention(ServerModule):
  
  def __init__(self, crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, dropout=0.1):
    super().__init__()
    self.comm = comm
    self.crypto = crypto
    self.softmax = Softmax(crypto, comm)
    self.dropout = Dropout(crypto, comm, dropout)

  # Input shape must be [B, N, 3E].
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
  
  def shares_matmul(self, helper, x_server, w_server, batchsize, input_dim, output_dim):
    x_server = x_server.copy()
    w_server = w_server.copy()
    x_server_encoded = self.crypto.matmul_encode_x(helper, x_server)
    w_server_encoded = self.crypto.matmul_encode_w(helper, w_server)
    s = self.crypto.field_random_mask(batchsize * output_dim)
    s_encoded = self.crypto.matmul_encode_y(helper, s)
    x_client_encrypted = self.crypto.deserialize(self.recv())
    w_client_encrypted = self.crypto.deserialize(self.recv())
    y_sc = self.crypto.matmul(helper, x_server_encoded, w_client_encrypted)
    y_cs = self.crypto.matmul(helper, x_client_encrypted, w_server_encoded)
    y_ss = self.crypto.field_mod(
      np.matmul(
        x_server.reshape((batchsize, input_dim)), 
        w_server.reshape(input_dim, output_dim))
    ).flatten()
    self.crypto.add_inplace(y_cs, y_sc)
    self.crypto.add_plain_inplace(y_cs, s_encoded)
    self.send(self.crypto.matmul_serialize_y(helper, y_cs))
    y_s = self.crypto.field_add(y_ss, self.crypto.field_negate(s))
    return y_s
  
  def forward(self, qkv):
    # qkv: [B, N, 3E]
    flatten = len(qkv.shape) == 1
    if flatten:
      qkv = np.reshape(qkv, (self.batchsize, self.sequence_length, 3 * self.embed_dim))
    q_server = qkv[:, :, :self.embed_dim].reshape((self.batchsize, -1))
    k_server = qkv[:, :, self.embed_dim:2*self.embed_dim].reshape((self.batchsize, -1))
    v_server = qkv[:, :, 2*self.embed_dim:].reshape((self.batchsize, -1))

    self.q_server = q_server
    self.k_server = k_server
    self.v_server = v_server

    a_server_list = []
    y_server_list = []
    for b in range(self.batchsize):
      k_transpose = np.transpose(k_server[b].reshape((self.sequence_length, self.embed_dim))).flatten()
      p_server = self.shares_matmul(self.helper_qkt, q_server[b], k_transpose, 
        self.sequence_length, self.embed_dim, self.sequence_length)

      p_server = self.comm.divide(p_server, math.floor(math.sqrt(self.embed_dim) * crypto.SCALE))

      a_server = self.softmax.forward(p_server)
      a_server = self.dropout.forward(a_server)
      a_server_list.append(a_server)

      y_server = self.shares_matmul(self.helper_av, a_server, v_server[b], 
        self.sequence_length, self.sequence_length, self.embed_dim)
      y_server_list.append(y_server)
    
    self.a_server = np.array(a_server_list)
    y_server = np.array(y_server_list) # [B, N, E]
    if flatten:
      y_server = y_server.flatten()
    else:
      y_server = y_server.reshape((self.batchsize, self.sequence_length, self.embed_dim))
    return y_server

  def backward(self, dy):
    flatten = len(dy.shape) == 1
    if flatten:
      dy = np.reshape(dy, (self.batchsize, self.sequence_length, self.embed_dim))
    dq_server_list = []
    dk_server_list = []
    dv_server_list = []
    for b in range(self.batchsize):
      dy_server = dy[b].flatten()
      a_server = self.a_server[b]

      # dv = a^T * dy
      a_transpose_server = np.transpose(a_server.reshape((self.sequence_length, self.sequence_length))).flatten()
      dv_server = self.shares_matmul(self.helper_dv, a_transpose_server, dy_server, self.sequence_length, self.sequence_length, self.embed_dim)
      dv_server = self.comm.truncate(dv_server)
      dv_server_list.append(dv_server)

      # da = dy * v^T
      v_transpose_server = np.transpose(self.v_server[b].reshape((self.sequence_length, self.embed_dim))).flatten()
      da_server = self.shares_matmul(self.helper_da, dy_server, v_transpose_server, self.sequence_length, self.embed_dim, self.sequence_length)
      da_server = self.comm.truncate(da_server)
      da_server = self.dropout.backward(da_server)

      # dp = softmax_backward(da)
      dp_server = self.softmax.backward(da_server)

      # dk = dp^T * q / sqrt(E)
      q_server = self.q_server[b]
      dp_transpose_server = np.transpose(dp_server.reshape((self.sequence_length, self.sequence_length))).flatten()
      dk_server = self.shares_matmul(self.helper_dk, dp_transpose_server, q_server, self.sequence_length, self.sequence_length, self.embed_dim)
      divisor = math.floor(math.sqrt(self.embed_dim) * crypto.SCALE)
      dk_server = self.comm.divide(dk_server, divisor)
      dk_server_list.append(dk_server)

      # dq = dp * k / sqrt(E)
      dq_server = self.shares_matmul(self.helper_dq, dp_server, self.k_server[b], self.sequence_length, self.sequence_length, self.embed_dim)
      dq_server = self.comm.divide(dq_server, divisor)
      dq_server_list.append(dq_server)

    dq_server = np.array(dq_server_list).reshape((self.batchsize, self.sequence_length, self.embed_dim)) # [B, N, E]
    dk_server = np.array(dk_server_list).reshape((self.batchsize, self.sequence_length, self.embed_dim)) 
    dv_server = np.array(dv_server_list).reshape((self.batchsize, self.sequence_length, self.embed_dim))
    dqkv = np.concatenate([dq_server, dk_server, dv_server], axis=2)
    if flatten:
      dqkv = dqkv.flatten()
    return dqkv
  
  def describe(self):
    return {
      "name": "ScaledDotProductAttention",
      "dropout": self.dropout.rate,
    }
  def parameters(self): return []
  def to_torch(self): return torch_models.ScaledDotProductAttention()

class MultiheadAttention(ServerModule):

  def __init__(self, crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, 
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

    # qkv_plain = self.crypto.to_decimal(self.crypto.field_mod(qkv + self.recv()))
    # print("qkv")
    # print(qkv_plain)

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

    # qkv_plain = self.crypto.to_decimal(self.crypto.field_mod(qkv + self.recv()))
    # print("qkv concat")
    # print(qkv_plain)

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
  
  def describe(self):
    return {
      "name": "MultiheadAttention",
      "embed_dim": self.embed_dim,
      "num_heads": self.num_heads,
      "dropout": self.dropout,
    }
  
  def parameters(self): return self.in_linear.parameters() + self.out_linear.parameters()

# Inputs stats are calculated on the last dimension
class LayerNorm(ServerModule):

  def __init__(self,
    crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, 
    dims, eps=1e-5, affine=True
  ):
    super().__init__()
    self.comm = comm
    self.crypto = crypto
    self.eps = eps
    self.eps_double_scaled = self.crypto.to_field(np.array(self.eps), self.crypto.default_scale() ** 2)
    self.affine = affine
    self.dims = dims
    if affine:
      self.weight = Parameter(np.ones((dims,)))
      self.bias = Parameter(np.zeros((dims,)))

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
    variance_eps = self.crypto.field_add(variance, self.eps_double_scaled) # [N, 1]
    std_inverse = self.comm.sqrt(variance_eps, scale_double, scale, True) # [N, 1]
    self.std_inverse = std_inverse

    # Get normalized value
    std_inverse_repeated = np.repeat(std_inverse, self.dims, axis=1) # [N, D]
    y = self.comm.elementwise_multiply(diff, std_inverse_repeated) # [N, D]
    self.x_normalized = y

    if self.affine: # y = (x - mean) / std * weight + bias
      y = self.comm.truncate(y)
      weight = self.crypto.to_field(self.weight.value, scale).reshape((1, -1))
      weight_repeated = np.repeat(weight, y.shape[0], axis=0) # [N, D]
      y = self.comm.elementwise_multiply(y, weight_repeated) # [N, D]
      bias = self.crypto.to_field(self.bias.value, scale_double).reshape((1, -1))
      bias_repeated = np.repeat(bias, y.shape[0], axis=0) # [N, D]
      y = self.crypto.field_add(y, bias_repeated) # [N, D]

    return y.flatten()

  def backward(self, dy_share):
    scale = self.crypto.default_scale()
    scale_double = scale ** 2

    # Reshape dy
    dy_share = dy_share.reshape((-1, self.dims)) # [N, D]

    if self.affine:
      weight = self.crypto.to_field(self.weight.value, scale).reshape((1, -1))
      weight_repeated = np.repeat(weight, dy_share.shape[0], axis=0) # [N, D]
      x_normalized = self.x_normalized
      x_normalized = self.comm.truncate(x_normalized)
      dweight = self.comm.elementwise_multiply(dy_share, x_normalized) # [N, D]
      dweight = np.sum(dweight, axis=0, keepdims=True).flatten() # [D]
      dweight = self.crypto.field_mod(dweight + self.recv())
      dweight = self.crypto.to_decimal(dweight, scale_double)
      self.weight.set_grad(dweight)
      dbias = np.sum(dy_share, axis=0, keepdims=True).flatten() # [D]
      dbias = self.crypto.field_mod(dbias + self.recv())
      dbias = self.crypto.to_decimal(dbias, scale)
      self.bias.set_grad(dbias)
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

  def describe(self):
    return {
      "name": "LayerNorm",
      "dims": self.dims,
      "eps": self.eps,
      "affine": self.affine,
    }
  
  def parameters(self): return [self.weight, self.bias] if self.affine else []

class Dropout(ServerModule):
  
  def __init__(self, crypto, comm, rate=0.5):
    super().__init__()
    self.comm = comm
    self.crypto = crypto
    self.rate = rate
    
  def static_prepare(self, shape): return shape
  def static_forward(self, x): return x

  def forward(self, x):
    if self.rate == 0: return x
    if self.is_training:
      self.choices = np.random.choice([0, 1], x.shape, p=[self.rate, 1-self.rate]).astype(np.uint64)
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
    if self.rate == 0: return dy
    return dy * self.choices
  
  def describe(self):
    return {
      "name": "Dropout",
      "rate": self.rate
    }

  def parameters(self): return []
  def prepare(self, s): return s
  def trusted_prepare(self, s, sv=True): return s
  def to_torch(self): return torch.nn.Dropout(self.rate)

class Residual(ServerModule):
  
  def __init__(self, crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication, branch_module):
    super().__init__()
    self.comm = comm
    self.crypto = crypto
    self.branch_module = branch_module

  def prepare(self, input_shape):
    self.input_shape = input_shape
    output_shape = self.branch_module.prepare(input_shape)
    assert output_shape == input_shape, f"shape mismatch: {input_shape}, {output_shape}"
    return input_shape
  
  def forward(self, x):
    branch_out = self.branch_module.forward(x)
    return self.crypto.field_add(x, branch_out)
  
  def backward(self, dy):
    branch_dx = self.branch_module.backward(dy)
    return self.crypto.field_add(dy, branch_dx)

  def to_torch(self): 
    return torch_models.Residual(self.branch_module.to_torch())
  
  def parameters(self):
    return self.branch_module.parameters()
  
  def describe(self):
    return {
      "name": "Residual",
      "branch": self.branch_module.describe()
    }

class Sequential(ServerModule):

  def __init__(self, crypto, comm, modules, verbose=True):
    super().__init__()
    self.items = modules
    self.crypto = crypto
    self.comm = comm
    self.verbose = verbose

  def forward(self, x):
    n = len(self.items)
    if self.verbose: print("Forward:  " + "." * n, end="")
    for i, each in enumerate(self.items):
      x = each.forward(x)
      if self.verbose: print("\rForward:  " + "-" * (i+1) + "." * (n-i-1), end="")
      # self.comm.send(x)
    if self.verbose: print("")
    return x

  def forward_plain(self, x):
    n = len(self.items)
    # print("Forward:  " + "." * n, end="")
    for i, each in enumerate(self.items):
      x = each.forward_plain(x)
      # print("\rForward:  " + "-" * (i+1) + "." * (n-i-1), end="")
      # self.comm.send(x)
    # print("")
    return x

  def backward(self, partial_y):
    n = len(self.items)
    if self.verbose: print("Backward: " + "." * n, end="")
    for i in range(n):
      partial_y = self.items[n - i - 1].backward(partial_y)
      if self.verbose: print("\rBackward: " + "." * (n-i-1) + "-" * (i+1), end="")
    if self.verbose: print("")
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

  def static_prepare(self, input_shape):
    for each in self.items:
      input_shape = each.static_prepare(input_shape)
    return input_shape

  def trusted_prepare(self, input_shape, save_file=True):
    original_shape = input_shape
    for each in self.items:
      # print(input_shape)
      input_shape = each.trusted_prepare(input_shape, save_file)
    print(f"Sequential: {original_shape} -> {input_shape}")
    return input_shape

  def train(self): 
    self.is_training = True
    for each in self.items: each.train()
  
  def eval(self):
    self.is_training = False
    for each in self.items: each.eval()

class Truncate(ServerModule):

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
    return np.zeros(self.x_shape, dtype=np.uint64).flatten()

  def backward_plain(self, partial_y):
    return None

  def parameters(self):
    return []

  def to_torch(self): 
    return self.model.to("cpu")

  def prepare(self, input_shape): 
    self.x_shape = input_shape
    x = torch.tensor(random_tensor(input_shape), 
      dtype=torch.float32, device=self.device)
    y = self.model(x)
    self.y_shape = y.shape
    return y.shape

  def trusted_prepare(self, input_shape, save_file=True):
    x = torch.tensor(random_tensor(input_shape), 
      dtype=torch.float32, device=self.device)
    y = self.model(x)
    self.y_shape = y.shape
    return y.shape

class TransformerEncoderLayer(ServerModule):

  def __init__(self, 
    crypto: crypto.EvaluationUtils, comm: communication_server.ServerCommunication,
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
    attention_block = Sequential(crypto, comm, [self_attn, dropout1], False)
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
    feedforward_block = Sequential(crypto, comm, [linear1, activation, linear2, dropout2, tc], False)
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
    # print("Forward res1")
    x = self.res1.forward(x)
    # print("Forward norm1")
    x = self.norm1.forward(x)
    x = self.comm.truncate(x)
    # print("Forward res2")
    x = self.res2.forward(x)
    # print("Forward norm2")
    x = self.norm2.forward(x)
    x = self.comm.truncate(x)
    return x
  
  def backward(self, dy):
    dy = self.norm2.backward(dy)
    dy = self.res2.backward(dy)
    dy = self.norm1.backward(dy)
    dy = self.res1.backward(dy)
    return dy
  
  def parameters(self):
    return self.res1.parameters() + self.norm1.parameters() + self.res2.parameters() + self.norm2.parameters()
  
  def describe(self):
    return {
      "name": "TransformerEncoderLayer",
      "d_model": self.d_model,
      "num_heads": self.num_heads,
      "dim_feedforward": self.dim_feedforward,
      "activation": self.activation_type,
      "dropout": self.dropout
    }
  


def server_model_from_pytorch(torch_model, crypto, comm):

  if isinstance(torch_model, torch.nn.ReLU):
    return ReLU(crypto, comm)
  
  if isinstance(torch_model, torch.nn.GELU):
    return GELU(crypto, comm)

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
  
  elif isinstance(torch_model, torch.nn.Softmax):
    print("Warning: Using mocked softmax.")
    return Softmax(crypto, comm)
  
  elif isinstance(torch_model, torch_models.ScaledDotProductAttention):
    return ScaledDotProductAttention(crypto, comm, torch_model.dropout)

  elif isinstance(torch_model, torch_models.MultiheadAttention):
    ret = MultiheadAttention(crypto, comm, torch_model.embed_dim, torch_model.num_heads, torch_model.dropout)
    ret.in_linear.weight.value = torch_model.inner.in_proj_weight.detach().cpu().numpy().copy()
    ret.in_linear.bias.value = torch_model.inner.in_proj_bias.detach().cpu().numpy().copy()
    ret.out_linear.weight.value = torch_model.inner.out_proj.weight.detach().cpu().numpy().copy()
    ret.out_linear.bias.value = torch_model.inner.out_proj.bias.detach().cpu().numpy().copy()
    return ret

  elif isinstance(torch_model, torch.nn.MultiheadAttention):
    # Note: the original torch.nn.MultiheadAttention's forward
    # method accepts separated Q,K,V but the torch_models.MultiheadAttention
    # wrapper accepts the same tensor to project to Q,K,V
    assert(torch_model._qkv_same_embed_dim)
    ret = MultiheadAttention(crypto, comm, torch_model.embed_dim, torch_model.num_heads, torch_model.dropout)
    ret.in_linear.weight.value = torch_model.in_proj_weight.detach().cpu().numpy().copy()
    ret.in_linear.bias.value = torch_model.in_proj_bias.detach().cpu().numpy().copy()
    ret.out_linear.weight.value = torch_model.out_proj.weight.detach().cpu().numpy().copy()
    ret.out_linear.bias.value = torch_model.out_proj.bias.detach().cpu().numpy().copy()
    return ret

  elif isinstance(torch_model, torch_models.TorchNative):
    return TorchNative(crypto, comm, torch_model.preset, config.DEVICE)
  
  elif isinstance(torch_model, torch.nn.LayerNorm):
    assert(len(torch_model.normalized_shape) == 1)
    dim = torch_model.normalized_shape[0]
    ret = LayerNorm(crypto, comm, dim, torch_model.eps, torch_model.elementwise_affine)
    if torch_model.elementwise_affine:
      ret.weight.value = torch_model.weight.detach().cpu().numpy().copy()
      ret.bias.value = torch_model.bias.detach().cpu().numpy().copy()
    return ret

  elif isinstance(torch_model, torch_models.Residual):
    module = server_model_from_pytorch(torch_model.module, crypto, comm)
    return Residual(crypto, comm, module)

  elif isinstance(torch_model, torch.nn.TransformerEncoderLayer):
    mha: MultiheadAttention = server_model_from_pytorch(torch_model.self_attn, crypto, comm)
    d_model = mha.embed_dim
    num_heads = mha.num_heads
    dim_feedforward = torch_model.linear1.out_features
    dropout = mha.dropout
    if torch_model.activation == torch.nn.functional.relu:
      act = "relu"
    else:
      act = "gelu"
    ret = TransformerEncoderLayer(crypto, comm, 
      d_model, num_heads, dim_feedforward, act, dropout)
    ret.res1.branch_module.items[0] = mha
    ret.norm1.weight.value = torch_model.norm1.weight.detach().cpu().numpy().copy()
    ret.norm1.bias.value = torch_model.norm1.bias.detach().cpu().numpy().copy()
    ret.res2.branch_module.items[0].weight.value = torch_model.linear1.weight.detach().cpu().numpy().copy()
    ret.res2.branch_module.items[0].bias.value = torch_model.linear1.bias.detach().cpu().numpy().copy()
    ret.res2.branch_module.items[2].weight.value = torch_model.linear2.weight.detach().cpu().numpy().copy()
    ret.res2.branch_module.items[2].bias.value = torch_model.linear2.bias.detach().cpu().numpy().copy()
    ret.norm2.weight.value = torch_model.norm2.weight.detach().cpu().numpy().copy()
    ret.norm2.bias.value = torch_model.norm2.bias.detach().cpu().numpy().copy()
    return ret
  
  elif isinstance(torch_model, torch_models.Truncate):
    return Truncate(crypto, comm)

  else:
    raise Exception("cannot convert from pytorch -- unsupported layer type")