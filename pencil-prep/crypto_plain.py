import numpy as np
import math
import time

PLAIN_MODULUS = 1 << 59
SCALE_BITS = 25
SCALE = 1<<SCALE_BITS
MOD_SWITCH_TO_NEXT = False
# If we use the preprocessing method, using mod_switch_to_next would 
# induce a small error. See seal-cuda/test/linear's testMatmulInt for a test.

# If we naively use HE in every execution, we need only (60, 60, 60) and MOD_SWITCH_TO_NEXT could be set to True
# see branch archive04 for this naive implementation.

def initialize_kernel():
  pytroy.initialize_kernel()

def get2e(p):
  i=1
  while i<p: i*=2
  return i


class EvaluationUtils:

  def __init__(self):
    pass

  def receive_keys(self, params_set):
    pass

  def to_field(self, a: np.ndarray, scale=SCALE, flatten=True):
    if flatten: a = a.flatten()
    a = a * scale
    a = np.where(a < 0, PLAIN_MODULUS + a, a).astype(np.uint64)
    return a

  def field_random_mask(self, size) -> np.ndarray:
    return np.random.randint(0, PLAIN_MODULUS, size, dtype=np.uint64)

  def field_negate(self, x) -> np.ndarray:
    return np.mod(PLAIN_MODULUS - x, PLAIN_MODULUS)

  def field_mod(self, x) -> np.ndarray:
    return np.mod(x, PLAIN_MODULUS)

  def field_add(self, x, y) -> np.ndarray:
    return np.mod(x + y, PLAIN_MODULUS)

  def to_decimal(self, a: np.ndarray, scale=SCALE, shape=None) -> np.ndarray:
    a = a.astype(np.float64)
    a = np.where(a > PLAIN_MODULUS // 2, a - PLAIN_MODULUS, a) / scale
    if shape is not None:
      a = np.reshape(a, shape)
    return a

  def default_scale(self) -> int: return SCALE

  def serialize(self, c):
    return c.save()

  def deserialize(self, c):
    return c

  def matmul_helper(self, batchsize, input_dims, output_dims):
    # print("MatmulHelper: ", batchsize, input_dims, output_dims)
    ret = MatmulHelper(batchsize, input_dims, output_dims)
    # print("Helper ok")
    return ret

  def matmul_encode_x(self, helper, x):
    return helper.encode_inputs(self.encoder, x)

  def matmul_encode_y(self, helper, y):
    return helper.encode_outputs(self.encoder, y)

  def matmul_encode_w(self, helper, w):
    return helper.encode_weights(self.encoder, w)

  def matmul(self, helper, x, w):
    ret = helper.matmul(self.evaluator, x, w)
    # if MOD_SWITCH_TO_NEXT: ret.mod_switch_to_next(self.evaluator)
    return ret

  def matmul_serialize_y(self, helper, y):
    return helper.serialize_outputs(self.evaluator, y)

  def matmul_deserialize_y(self, helper, s):
    return helper.deserialize_outputs(self.evaluator, s)

  def conv2d_helper(self, batchsize, image_height, image_width, kernel_height, kernel_width, input_channels, output_channels):
    # print("Conv2dHelper: ", batchsize, batchsize, image_height, image_width, kernel_height, kernel_width, input_channels, output_channels)
    ret = Conv2dHelper(
      batchsize, image_height, image_width, kernel_height, kernel_width,
      input_channels, output_channels, self.slot_count
    )
    # print("Helper ok")
    return ret

  def conv2d_encode_x(self, helper, x):
    return helper.encode_inputs(self.encoder, x)

  def conv2d_encode_y(self, helper, y):
    return helper.encode_outputs(self.encoder, y)

  def conv2d_encode_w(self, helper, w):
    return helper.encode_weights(self.encoder, w)

  def conv2d(self, helper, x, w):
    ret = helper.conv2d(self.evaluator, x, w)
    if MOD_SWITCH_TO_NEXT: ret.mod_switch_to_next(self.evaluator)
    return ret

  def conv2d_serialize_y(self, helper, y):
    return helper.serialize_outputs(self.evaluator, y)

  def conv2d_deserialize_y(self, helper, s):
    return helper.deserialize_outputs(self.evaluator, s)

  def encrypt_cipher2d(self, x):
    return x.encrypt(self.encryptor)

  def add_plain_inplace(self, x, y):
    x.add_plain_inplace(self.evaluator, y)

  def add_inplace(self, x, y):
    x.add_inplace(self.evaluator, y)

  def add_plain(self, x, y):
    return x.add_plain(y)

  
  def relu_plain(self, x):
    return np.where(x > PLAIN_MODULUS//2, 0, x)

  def drelumul_plain(self, x, y):
    return np.where(x > PLAIN_MODULUS//2, 0, y)

  def truncate_plain(self, x, scale:int=SCALE):
    x = np.where(x > PLAIN_MODULUS//2, PLAIN_MODULUS - (PLAIN_MODULUS - x) // scale, x // scale)
    error = np.random.randint(0, 2, x.shape, dtype=np.uint64)
    # pm = np.random.randint(0, 2, x.shape, dtype=np.uint64) * 2 - 1
    x = self.field_mod(x - error)
    return x

  def divide_plain(self, x, divident):
    x = np.where(x > PLAIN_MODULUS//2, PLAIN_MODULUS - (PLAIN_MODULUS - x) // divident, x // divident)
    error = np.random.randint(0, 2, x.shape, dtype=np.uint64)
    # pm = np.random.randint(0, 2, x.shape, dtype=np.uint64) * 2 - 1
    x = self.field_mod(x - error)
    return x


class EncryptionUtils(EvaluationUtils):

  def __init__(self):
    super().__init__()

  def generate_keys(self):
    self.bfv_params = pytroy.EncryptionParameters(pytroy.SchemeType.bfv)
    self.bfv_params.set_poly_modulus_degree(BFV_POLY_DEGREE)
    self.bfv_params.set_plain_modulus(PLAIN_MODULUS)
    # print(f"CKKS max bit count = {pytroy.CoeffModulus.MaxBitCount(POLY_MODULUS_DEGREE)}")
    self.bfv_params.set_coeff_modulus(pytroy.CoeffModulus.create(BFV_POLY_DEGREE, BFV_Q_BITS))
    self.bfv_context = pytroy.SEALContext(self.bfv_params)
    self.keygen = pytroy.KeyGenerator(self.bfv_context)
    self.secret_key = self.keygen.secret_key()
    self.public_key = self.keygen.create_public_key()
    self.relin_key = self.keygen.create_relin_keys()
    self.encryptor = pytroy.Encryptor(self.bfv_context, self.public_key)
    self.decryptor = pytroy.Decryptor(self.bfv_context, self.secret_key)
    self.evaluator = pytroy.Evaluator(self.bfv_context)
    self.encoder = pytroy.BatchEncoder(self.bfv_context)
    self.slot_count = self.encoder.slot_count()

    self.encryptor.set_secret_key(self.keygen.secret_key())

    # x1 = self.encoder.encode_polynomial(np.array([1,2,3,4], dtype=np.uint64))
    # y1 = self.encryptor.encrypt(x1)
    # x2 = self.encoder.encode_polynomial(np.array([1,2,3,4], dtype=np.uint64))
    # y2 = self.encryptor.encrypt(x2)

    return (self.public_key.save(), self.relin_key.save())
      
  def matmul_decrypt_y(self, helper, y):
    return helper.decrypt_outputs(self.encoder, self.decryptor, y)

  def conv2d_decrypt_y(self, helper, y):
    return helper.decrypt_outputs(self.encoder, self.decryptor, y)

