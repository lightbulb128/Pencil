import socket
import config
import pickle
import os
import numpy as np
import time

from communication_common import *
from tools.cheetah_provider_alice import CheetahProvider
from tools.sci_provider_alice import SCIProvider

from crypto_gpu_cheetah import SCALE_BITS

class ServerCommunication:

  def __init__(self):
    pass

  def listen(self):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      self.socket.bind((config.HOST, config.PORT))
    except:
      self.socket.bind((config.HOST, config.PORT+1))
    self.socket.listen()
  
  def accept_connection(self):
    self.connection, self.client_address = self.socket.accept()
    self.cheetah = CheetahProvider(SCALE_BITS)
    self.cheetah.startComputation()
    self.sci = SCIProvider(SCALE_BITS)
    self.sci.startComputation()
    self.modulus = 1 << self.cheetah.dbits()

  def close_connection(self):
    self.cheetah.endComputation()
    self.sci.endComputation()
    self.connection.close()

  def send(self, obj):
    obj_bytes = pickle.dumps(obj)
    length = len(obj_bytes)
    if config.NETWORK_BANDWIDTH != 0:
      # print("Sleep:", length / config.NETWORK_BANDWIDTH)
      time.sleep(length / config.NETWORK_BANDWIDTH)
    send_msg(self.connection, obj_bytes)

  def recv(self):
    obj_bytes = recv_msg(self.connection)
    return pickle.loads(obj_bytes)

  def fmod(self, ans):
    ans = np.fmod(ans, self.modulus)
    ans = np.where(ans > self.modulus / 2, ans - self.modulus, ans)
    ans = np.where(ans < -self.modulus / 2, ans + self.modulus, ans)
    return ans

  def relu(self, x, truncate=True):
    shape = x.shape
    ans, d = self.cheetah.relu(x.flatten(), truncate)
    ans = ans.reshape(shape)
    d = d.reshape(shape)
    return ans, d

  def drelumul(self, x, d):
    shape = x.shape
    assert(shape == d.shape)
    ans = self.cheetah.drelumul(x.flatten(), d.flatten())
    ans = ans.reshape(shape)
    return ans

  def truncate(self, x):
    shape = x.shape
    result = self.cheetah.truncate(x.flatten())
    result = result.reshape(shape)
    return result

  def sqrt(self, x, scale_in, scale_out, inverse):
    shape = x.shape
    result = self.sci.sqrt(x.flatten(), scale_in, scale_out, inverse)
    result = result.reshape(shape)
    return result

  def divide(self, x, d):
    shape = x.shape
    result = self.cheetah.divide(x.flatten(), d)
    result = result.reshape(shape)
    return result

  def elementwise_multiply(self, x, y):
    shape = x.shape
    result = self.sci.elementwise_multiply(x.flatten(), y.flatten())
    result = result.reshape(shape)
    return result

  def softmax(self, x, d):
    shape = x.shape
    result = self.sci.softmax(x.flatten(), d)
    result = result.reshape(shape)
    return result

  def tanh(self, x):
    shape = x.shape
    result = self.sci.tanh(x.flatten())
    result = result.reshape(shape)
    return result