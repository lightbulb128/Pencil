import numpy as np

def serialize(x): return x
def deserialize(x): return x

class EvaluationUtils:

  def __init__(self):
    pass

  def serialize(self, x): return x
  def deserialize(self, x): return x

  def fmod(self, ans):
    # ans = np.fmod(ans, self.modulus)
    # ans = np.where(ans > self.modulus / 2, ans - self.modulus, ans)
    # ans = np.where(ans < -self.modulus / 2, ans + self.modulus, ans)
    return ans

  def receive_keys(self, any):
    pass

  def encrypt_scalar(self, k):
    return k

  def flat_encrypt(self, x, double_scale=False):
    return x.copy()

  def flat_add_plain(self, y, x):
    y += x

  def flat_mul_scalar(self, y, s):
    return y * s

  def flat_mul_enc_scalar(self, y, s):
    return y * s

  def flat_add(self, y, x):
    y += x

  def vector_encrypt(self, x, layer=0):
    return x.copy()

  def vector_add_plain(self, y, x):
    y += x

  def vector_add(self, y, x):
    y += x

  def vector_mul_plain(self, y, x):
    y *= x

  def vector_mul_scalar(self, y, s):
    return y*s

  def vector_square(self, y):
    y *= y

  def vector_mul(self, x, y):
    return x*y

  def linear_encrypt(self, x, layer=-3, info=None):
    return x.copy()

  def linear_add_plain(self, y, x, info=None):
    y += x

  def linear_encode_weights_small(self, b, parms=-2):
    return b

  def linear_find_split(self, b_shape):
    height, width = b_shape
    return height, width

  def linear_get_enc_info(self, b_shape):
    bh, bw = self.linear_find_split(b_shape)
    return (bh, bw, b_shape[0], b_shape[1])

  def linear_encode_weights(self, b, parms=-2):
    height, width = b.shape
    h, w = self.linear_find_split(b.shape)
    return {
      "enc": b, 
      "info": (h, w, height, width)
    }

  def linear_matmul_cp_encoded(self, a, encoded):
    return np.matmul(a, encoded["enc"])

  def linear_matmul_cp(self, a, b):
    return np.matmul(a, b)
  
  def linear_add_constant_inplace_cp(self, a, b):
    a += b

  def conv2d_encrypt(self, x, output_channels = None):
    return x.copy()

  def conv2d_encrypt_natural(self, x):
    return self.conv2d_encrypt(x)

  def conv2d_add_plain(self, y, x):
    y += x

  def conv2d_add_plain_natural(self, y, x):
    self.conv2d_add_plain(y, x)

  def conv2d_conv(self, x, weight, bias):
    batchsize, _, x_h, x_w = x.shape
    output_channels, _, k_h, k_w = weight.shape
    y = np.zeros((batchsize, output_channels, x_h, x_w), dtype=np.float32)
    y_h = x_h - k_h + 1
    y_w = x_w - k_w + 1
    for b in range(batchsize):
      for oc in range(output_channels):
        for i in range(k_h):
          for j in range(k_w):
            y[b, oc, :y_h, :y_w] += np.sum(
              x[b, :, i:i+y_h, j:j+y_w] * weight[oc, :, i, j].reshape((-1, 1, 1)),
              axis=0)
        if bias is not None: y[b, oc, :y_h, :y_w] += bias[oc]
    return y

  def conv2d_largek_cp(self, x, weight):
    batchsize, _, x_h, x_w = x.shape
    output_channels, input_channels, k_h, k_w = weight.shape
    y = np.zeros((batchsize, output_channels, x_h, x_w), dtype=np.float32)
    y_h = x_h - k_h + 1
    y_w = x_w - k_w + 1
    for b in range(batchsize):
      for oc in range(output_channels):
        for i in range(y_h):
          for j in range(y_w):
            y[b, oc, i, j] = np.sum(x[b, :, i:i+k_h, j:j+k_w] * weight[oc, :, :, :])
    return y

  def conv2d_largek_pc(self, x, weight, k_h, k_w):
    return self.conv2d_largek_cp(x, weight[:, :, :k_h, :k_w])
    

class EncryptionUtils(EvaluationUtils):

  def __init__(self):
    super().__init__()

  def generate_keys(self):
    return 0

  def linear_decrypt(self, x, info):
    return x.copy()

  def conv2d_decrypt(self, x):
    return x.copy()

  def conv2d_decrypt_natural(self, x):
    return x.copy()

  def vector_decrypt(self, x):
    return x.copy()

  def flat_decrypt(self, x):
    return x.copy()