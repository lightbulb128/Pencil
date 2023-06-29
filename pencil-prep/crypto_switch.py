from config import CRYPTO

assert(CRYPTO == "gpu_cheetah")
print("Using crypto: crypto_gpu_cheetah")
import crypto_gpu_cheetah as cryptography
cryptography.initialize_kernel()

# if CRYPTO == "ckks":
#     print("Using crypto: crypto_ckks")
#     import crypto_ckks as cryptography



# elif CRYPTO == "gpu_ezpc":
#     print("Using crypto: crypto_gpu_ezpc")
#     import crypto_gpu_ezpc as cryptography
#     cryptography.initialize_kernel()




# elif CRYPTO == "gpu_old":
#     print("Using crypto: crypto_gpu_old")
#     import crypto_gpu_old as cryptography
#     cryptography.initialize_kernel()



# elif CRYPTO == "gpu":
#     print("Using crypto: crypto_gpu")
#     import crypto_gpu as cryptography
#     cryptography.initialize_kernel()


# elif CRYPTO == "gpu_cheetah":
#     print("Using crypto: crypto_gpu_cheetah")
#     import crypto_gpu_cheetah as cryptography
#     cryptography.initialize_kernel()




# elif CRYPTO == "plain":
#     print("Using crypto: crypto_plain")
#     import crypto_plain as cryptography


    
# else:
#     raise Exception("Unknown cryptography setting: " + CRYPTO)



'''
ckks
    The original implementation, using SEAL.
    Does NOT treat matrix multiplication as multiple matrix-vector muls.
    Conv2d input is not expanded when encrypted.

gpu_old
    Much like "ckks", but use pytroy.

gpu_ezpc
    Change linear and conv2d operations to same as in SCI_HE

gpu
    Results show that gpu_ezpc's conv2d is not good as gpu_old, but linear is better.
    So "gpu" uses "gpu_ezpc"'s linear and "gpu_old"'s conv2d.

gpu_cheetah
    Changed both conv2d and linear implementation to Cheetah's polynomial solution.

plain
    Plaintext computation, for comparison on performance.


'''