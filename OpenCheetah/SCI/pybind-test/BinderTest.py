import sys
sys.path.append("../../build/lib")

import argparse
import numpy as np

import socket
import pickle
import os
import numpy as np
import time
import struct

HOST = "localhost"
PORT = 30010

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

class ClientCommunication:
  
  def __init__(self):
    pass

  def connect(self):
    self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      self.connection.connect((HOST, PORT))
    except:
      self.connection.connect((HOST, PORT + 1))
  
  def close_connection(self):
    self.connection.close()

  def send(self, obj):
    obj_bytes = pickle.dumps(obj)
    length = len(obj_bytes)
    send_msg(self.connection, obj_bytes)

  def recv(self):
    obj_bytes = recv_msg(self.connection)
    return pickle.loads(obj_bytes)


class ServerCommunication:

  def __init__(self):
    pass

  def listen(self):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      self.socket.bind((HOST, PORT))
    except:
      self.socket.bind((HOST, PORT+1))
    self.socket.listen()
  
  def accept_connection(self):
    self.connection, self.client_address = self.socket.accept()

  def close_connection(self):
    self.connection.close()

  def send(self, obj):
    obj_bytes = pickle.dumps(obj)
    length = len(obj_bytes)
    send_msg(self.connection, obj_bytes)

  def recv(self):
    obj_bytes = recv_msg(self.connection)
    return pickle.loads(obj_bytes)


BFV_POLY_DEGREE = 8192
BFV_Q_BITS = (60, 60, 60)
PLAIN_MODULUS_BITS = 41
PLAIN_MODULUS = 1 << PLAIN_MODULUS_BITS
SCALE_BITS = 10
SCALE = 1<<SCALE_BITS

def to_field(a: np.ndarray, scale=SCALE):
    a = a.flatten() * scale
    a = np.where(a < 0, PLAIN_MODULUS + a, a).astype(np.uint64)
    return a


def to_decimal(a: np.ndarray, scale=SCALE, shape=None):
    a = a.astype(np.float64)
    a = np.where(a > PLAIN_MODULUS // 2, a - PLAIN_MODULUS, a) / scale
    if shape is not None:
        a = np.reshape(a, shape)
    return a

def field_random_mask(size):
    return np.random.randint(0, PLAIN_MODULUS, size, dtype=np.uint64)

def field_negate(x):
    return np.mod(PLAIN_MODULUS - x, PLAIN_MODULUS)

def field_mod(x):
    return np.mod(x, PLAIN_MODULUS)

def field_add(x, y):
    return np.mod(x + y, PLAIN_MODULUS)

def get_shares(x):
    x1 = field_random_mask(x.size)
    x2 = field_add(x, field_negate(x1))
    return x1, x2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int)
    args = parser.parse_args()
    party = args.p
    assert(party == 1 or party == 2)
    if party == 1:
        if PLAIN_MODULUS_BITS == 37:
          import cheetah_provider_alice_37 as sci_provider
        elif PLAIN_MODULUS_BITS == 41:
          import cheetah_provider_alice_41 as sci_provider
        else:
          import cheetah_provider_alice as sci_provider
        comm = ServerCommunication()
        comm.listen()
        comm.accept_connection()
    else:
        if PLAIN_MODULUS_BITS == 37:
          import cheetah_provider_bob_37 as sci_provider
        elif PLAIN_MODULUS_BITS == 41:
          import cheetah_provider_bob_41 as sci_provider
        else:
          import cheetah_provider_bob as sci_provider
        comm = ClientCommunication()
        comm.connect()

    np.random.seed(12345)

    def reconstruct(x):
        comm.send(x)
        return field_add(x, comm.recv())

    provider = sci_provider.CheetahProvider(SCALE_BITS)
    provider.startComputation()
    n = 5

    print("[Relu]")
    r = np.random.rand(n) * 2 - 1
    print("x    =", r)
    r_field = to_field(r)
    r_shares = get_shares(r_field)
    result, drelu = provider.relu(r_shares[party-1], False)
    result = reconstruct(result)
    result = to_decimal(result)
    print("relu =", result)


    print("[Drelumul]")
    r = np.random.rand(n) * 2 - 1
    print("d    =", r)
    r_field = to_field(r)
    r_shares = get_shares(r_field)
    result = provider.drelumul(r_shares[party-1], drelu)
    result = reconstruct(result)
    result = to_decimal(result)
    print("back =", result)


    print("[Relu]")
    r = np.random.rand(n) * 2 - 1
    print("x    =", r)
    r_field = to_field(r, SCALE*SCALE)
    r_shares = get_shares(r_field)
    result, drelu = provider.relu(r_shares[party-1], True)
    result = reconstruct(result)
    result = to_decimal(result)
    print("relu =", result)
    
    print("[Truncate]")
    r = np.random.rand(n) * 2 - 1
    print("x    =", r)
    r_field = to_field(r, SCALE*SCALE)
    r_shares = get_shares(r_field)
    result = provider.truncate(r_shares[party-1])
    result = reconstruct(result)
    result = to_decimal(result)
    print("trun =", result)

    
    print("[Divide]")
    r = np.random.rand(n) * 2 - 1
    print("x    =", r)
    r_field = to_field(r, SCALE)
    r_shares = get_shares(r_field)
    result = provider.divide(r_shares[party-1], 155)
    result = reconstruct(result)
    result = to_decimal(result)
    print("divd =", result)
    
    
    print("[Double scale test]")
    r = np.random.rand(n) * 2 - 1
    print("x    =", r)
    r_field = to_field(r, SCALE * SCALE)
    r_shares = get_shares(r_field)
    result = provider.divide(r_shares[party-1], SCALE)
    result = reconstruct(result)
    result = to_decimal(result)
    print("divd =", result)


    print("[Max]")
    n = 12
    r = np.random.rand(n) * 2 - 1
    print("x    =", r)
    r_field = to_field(r, SCALE)
    r_shares = get_shares(r_field)
    result = provider.max(r_shares[party-1], 4)
    result = reconstruct(result)
    result = to_decimal(result)
    print("divd =", result)
    

    provider.endComputation()
    comm.close_connection()