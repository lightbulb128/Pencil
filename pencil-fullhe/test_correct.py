from crypto_switch import cryptography as crypto
import crypto_plain
import numpy as np
import torch.nn
import torch
import time

def absmax(n):
    return np.max(np.abs(n))

def test_forward_fc():

    alice = crypto.EncryptionUtils()
    bob = crypto.EvaluationUtils()
    bob.receive_keys(alice.generate_keys())

    bound = 16
    batchsize = 64
    input_dims = 512
    output_dims = 10

    x = np.random.rand(batchsize, input_dims) * (bound*2) - bound
    w = np.random.rand(input_dims, output_dims) * (bound*2) - bound
    # x = np.random.rand(1, 1) * (r*2) - r
    # w = np.random.rand(1, 1) * (r*2) - r

    # x = np.array([
    #     [1,2,3,4]
    # ], dtype=np.float32)
    # w = np.array([
    #     [1,2,3,4],
    #     [5,6,7,8],
    #     [9,10,11,12],
    #     [13,14,15,16]
    # ], dtype=np.float32)

    alice_helper = alice.matmul_helper(batchsize, input_dims, output_dims)
    bob_helper = bob.matmul_helper(batchsize, input_dims, output_dims)

    x_field = alice.to_field(x)
    w_field = bob.to_field(w)

    x_encoded = alice.matmul_encode_x(alice_helper, x_field)
    x_enc = alice.encrypt_cipher2d(x_encoded)
    w_encoded = bob.matmul_encode_w(bob_helper, w_field)
    
    y_enc = bob.matmul(bob_helper, x_enc, w_encoded)
    y_dec = alice.matmul_decrypt_y(alice_helper, y_enc)
    y_dec = alice.to_decimal(y_dec, alice.default_scale()**2, (batchsize, output_dims))

    y = np.matmul(x, w)
    print(absmax(y_dec - y))

def test_forward():

    alice = crypto.EncryptionUtils()
    bob = crypto.EvaluationUtils()
    bob_plain = crypto_plain.EvaluationUtils()
    bob.receive_keys(alice.generate_keys())

    r = 512
    x = np.random.rand(1, 16, 22, 22) * (r*2) - r
    w = np.random.rand(1, 16, 5, 5) * (r*2) - r
    b = np.random.rand(1) * (r*2) - r
    
    # x = np.random.rand(16, 3, 7, 7) * 2 - 1
    # w = np.random.rand(5, 3, 3, 3) * 2 - 1
    # b = np.random.rand(5) * 2 - 1
    
    x_enc = alice.conv2d_encrypt(x, w.shape[0])
    x_enc = self.crypto.deserialize(self.crypto.serialize(x_enc))

    timed = time.time()
    y_enc = bob.conv2d_conv(x_enc, w, b)
    print("time =", time.time() - timed)

    y_dec = alice.conv2d_decrypt(y_enc)[:,:,:x.shape[2]-w.shape[2]+1,:x.shape[3]-w.shape[3]+1]
    y = bob_plain.conv2d_conv(x, w, b)[:,:,:y_dec.shape[2],:y_dec.shape[3]]
    print(absmax(y_dec - y))

    t = torch.nn.Conv2d(16, 1, 5, bias=True if b is not None else False)
    t.weight.data = torch.tensor(w)
    if b is not None:
        t.bias.data = torch.tensor(b)
    y_torch = t(torch.tensor(x)).detach().numpy()
    print(absmax(y_torch - y))

def test_forward_largek_cp():

    alice = crypto.EncryptionUtils()
    bob = alice
    bob_plain = crypto_plain.EvaluationUtils()
    bob.receive_keys(alice.generate_keys())

    batch_size, ic, oc, image_size, kernel_size = 1, 16, 1, 12, 5
    r = 512
    x = np.random.rand(batch_size, ic, image_size, image_size) * (r*2) - r
    w = np.random.rand(oc, ic, kernel_size, kernel_size) * (r*2) - r
    torchm = torch.nn.Conv2d(ic, oc, kernel_size, bias=False)
    torchm.weight.data = torch.tensor(w, dtype=torch.float32)
    x_enc = alice.conv2d_encrypt_natural(x)
    y_enc = bob.conv2d_largek_cp(x_enc, w)
    k = image_size - kernel_size + 1
    y_dec = alice.conv2d_decrypt_natural(y_enc)[:,:,:k,:k]
    y = bob_plain.conv2d_largek_cp(x, w)[:,:,:k,:k]
    y_torch = torchm(torch.tensor(x, dtype=torch.float32))
    # print(y_dec)
    # print(y)
    print(absmax(y_dec - y))
    print(absmax(y_torch.detach().numpy() - y))

def test_forward_largek_pc():

    alice = crypto.EncryptionUtils()
    bob = alice
    bob_plain = crypto_plain.EvaluationUtils()
    bob.receive_keys(alice.generate_keys())

    batch_size, ic, oc, image_size, kernel_size = 1, 16, 1, 12, 5
    r = 512
    x = np.random.rand(batch_size, ic, image_size, image_size) * (r*2) - r
    # x = np.array([[[
    #     [10,20,30,40,50],
    #     [10,20,30,40,50],
    #     [10,20,30,40,50],
    #     [10,20,30,40,50],
    #     [10,20,30,40,50],
    # ]]])
    w = np.random.rand(oc, ic, kernel_size, kernel_size) * (r*2) - r
    # w = np.array([[[
    #     [10,20,30],
    #     [10,20,30],
    #     [10,20,30],
    # ]]])
    torchm = torch.nn.Conv2d(ic, oc, kernel_size, bias=False)
    torchm.weight.data = torch.tensor(w, dtype=torch.float32)
    expanded_w = np.zeros((w.shape[0], w.shape[1], x.shape[2], x.shape[3]))
    expanded_w[:, :, :w.shape[2], :w.shape[3]] = w
    w_enc = alice.conv2d_encrypt_natural(expanded_w)
    y_enc = bob.conv2d_largek_pc(x, w_enc, w.shape[2], w.shape[3])
    k = image_size - kernel_size + 1
    y_dec = alice.conv2d_decrypt_natural(y_enc)[:,:,:k,:k]
    y = bob_plain.conv2d_largek_pc(x, expanded_w, w.shape[2], w.shape[3])[:,:,:k,:k]
    y_torch = torchm(torch.tensor(x, dtype=torch.float32))
    print(absmax(y_dec - y))
    print(absmax(y_torch.detach().numpy() - y))

if __name__ == "__main__":
    np.set_printoptions(3, suppress=True)
    test_forward_fc()
    # test_forward()
    # test_forward_largek_pc()