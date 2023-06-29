import torchvision
import numpy as np
import math
from matplotlib import pyplot
import time

modulus = 1<<59
scale = 1<<10
assert((1<<int(math.log2(modulus))) == modulus)

def load_data(count=100):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_data_raw = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    ret = []
    for i, item in enumerate(train_data_raw):
        if i==count: break
        ret.append(item[0])
    return ret

def to_field(a: np.ndarray, scale=scale, flatten=False):
    if flatten: a = a.flatten()
    a = (a * scale).astype(np.int64)
    a = np.where(a < 0, modulus + a, a).astype(np.uint64)
    return a

def field_random_mask(size = 1):
    return np.random.randint(0, modulus, size, dtype=np.uint64)

def field_negate(x):
    return np.mod(modulus - x, modulus)

def field_mod(x):
    return np.mod(x, modulus, dtype=np.uint64)

def field_add(x, y):
    return np.mod(x + y, modulus)

def to_decimal(a: np.ndarray, scale=scale):
    a = a.astype(np.float64)
    a = np.where(a > modulus // 2, a - modulus, a) / scale
    return a

def max_abs(a):
    a = a.astype(np.int64)
    a = np.where(a > modulus // 2, a - modulus, a)
    a = np.abs(a)
    return np.max(a)
    

# ax + by = gcd(a, b) = q
def exgcd(a, b):
    if b==0: return 1, 0, a
    else:
        x, y, q = exgcd(b, a%b)
        x, y = y, (x - a//b*y)
        return x, y, q

def inv(a):
    x, y, q = exgcd(a, modulus)
    if q != 1: return None
    return x


def min2(a):
    p=1
    while a%2==0: a, p = a//2, p*2
    return p

# ? * a = b (mod modulus)
def solve(a, b):
    a = (int(a) % modulus + modulus) % modulus
    b = (int(b) % modulus + modulus) % modulus
    x, y, q = exgcd(a, modulus)
    if b%q!=0: return 0, 0
    x = x * (b//q) % modulus
    increasement = modulus // min2(a)
    x %= increasement
    if (x > increasement//2): x-=increasement
    return x, increasement

def attack_two_combination(ki, tilde_v, value_range, realv=None):

    # attack sum(ki*vi) = tilde_v, for a scalar v
    def attack_single_pixel(ki, tilde_v, value_range):
        assert(len(ki) == 2)
        # k0 * v0 + k1 * v1 = tilde_v
        result = 0, 0, 0, 0
        for v0 in range(-value_range + 1, value_range):
            prod = int(tilde_v) - int(ki[0]) * int(v0)
            v1, v1inc = solve(ki[1], prod)
            if v1inc==0: continue
            v0inc = modulus // min2(int(ki[0]))
            if v1 > modulus//2: v1 -= modulus
            if v1 < -modulus//2: v1 += modulus
            if abs(v1) < value_range: 
                result = v0, v1, v0inc, v1inc
                break
        print("Attack", tilde_v, "=>", result)
        return result

    assert(len(ki) == 2)
    channels, h, w = tilde_v.shape
    v = np.zeros((channels, h, w, 4), dtype=np.uint64)
    for c in range(channels):
        for i in range(h):
            for j in range(w):
                print(f"{c},{i},{j} ", end="")
                p0, p1, p0inc, p1inc = attack_single_pixel(ki, tilde_v[c,i,j], value_range)
                if not (realv is None):
                    print(realv[0,c,i,j], realv[1,c,i,j])
                v[c,i,j] = [p0, p1, p0inc, p1inc]
    return v[:, :, :, 0], v[:, :, :, 1]



def attack_three_combination(ki, tilde_v, value_range, realv=None):

    # attack sum(ki*vi) = tilde_v, for a scalar v
    def attack_single_pixel(ki, tilde_v, value_range):
        assert(len(ki) == 3)
        # k0 * v0 + k1 * v1 + k2 * v2 = tilde_v
        result = 0, 0, 0
        for v0 in range(-value_range + 1, value_range):
            print(v0)
            for v1 in range(-value_range + 1, value_range):
                prod = int(tilde_v) - int(ki[0]) * int(v0) - int(ki[1]) * int(v1)
                v2, v2inc = solve(ki[2], prod)
                if v2inc==0: continue
                if v2 > modulus//2:  v2 -= modulus
                if v2 < -modulus//2: v2 += modulus
                if abs(v2) < value_range: 
                    print(v0, v1, v2)
    
    print(int(realv[0,0,0,0]) - modulus, realv[1,0,0,0], realv[2,0,0,0])
    attack_single_pixel(ki, tilde_v[0,0,0], value_range)


if __name__ == "__main__":
    data = load_data(10)
    data = [to_field(np.array(item)) for item in data]
    print("Max value:", max_abs(data[0]), max_abs(data[1]), max_abs(data[2]))

    def attack_2():
        np.random.seed(0)
        v = np.stack((data[0], data[1]))
        k = field_random_mask(2)
        tilde_v = field_mod(np.sum(v*k.reshape(2, 1, 1, 1), axis=0))
        attack_two_combination(k, tilde_v, 2400, realv=v)   
    
    def attack_3():
        np.random.seed(0)
        v = np.stack((data[0], data[1], data[2]))
        k = field_random_mask(3)
        tilde_v = field_mod(np.sum(v*k.reshape(3, 1, 1, 1), axis=0))
        attack_three_combination(k, tilde_v, 1200, realv=v)   
    
    timed = time.time()
    attack_3()
    print(time.time() - timed)
