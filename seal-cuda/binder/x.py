import numpy as np
import pytroy

def to_field(a: np.ndarray, scale, modulus, flatten=True):
    if flatten: a = a.flatten()
    a = a * scale
    a = np.where(a < 0, modulus + a, a).astype(np.uint64)
    return a


def to_decimal(a: np.ndarray, scale, modulus, shape=None) -> np.ndarray:
    a = a.astype(np.float64)
    a = np.where(a > modulus // 2, a - modulus, a) / scale
    if shape is not None:
        a = np.reshape(a, shape)
    return a

if __name__ == "__main__":

    pytroy.initialize_kernel()

    plain_modulus = 1 << 41
    scale = 1 << 12
    poly_modulus_degree = 8192
    coeff_modulus_bits = [60, 40, 40, 60]

    params = pytroy.EncryptionParameters(pytroy.SchemeType.bfv)

    params.set_poly_modulus_degree(poly_modulus_degree)
    params.set_plain_modulus(plain_modulus)
    params.set_coeff_modulus(pytroy.CoeffModulus.create(poly_modulus_degree, coeff_modulus_bits))

    context = pytroy.SEALContext(params)
    keygen = pytroy.KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.create_public_key()
    relin_key = keygen.create_relin_keys()

    encoder = pytroy.BatchEncoder(context)
    encryptor = pytroy.Encryptor(context, public_key)
    decryptor = pytroy.Decryptor(context, secret_key)
    evaluator = pytroy.Evaluator(context)
    encryptor.set_secret_key(secret_key)

    # matmul example

    batch_size = 5
    input_dim = 10
    output_dim = 20

    x = np.random.random((batch_size, input_dim))
    w = np.random.random((input_dim, output_dim))
    s = np.random.random((batch_size, output_dim))

    x_scaled = to_field(x, scale, plain_modulus, True)
    w_scaled = to_field(w, scale, plain_modulus, True)
    s_scaled = to_field(s, scale**2, plain_modulus, True)

    helper = pytroy.MatmulHelper(batch_size, input_dim, output_dim, poly_modulus_degree)

    x_encoded = helper.encode_inputs(encoder, x_scaled)
    w_encoded = helper.encode_weights(encoder, w_scaled)

    x_cipher = x_encoded.encrypt(encryptor)

    x_cipher_serialized = x_cipher.save()
    x_cipher_deserialized = pytroy.Cipher2d()
    x_cipher_deserialized.load(x_cipher_serialized, context)

    y_cipher = helper.matmul(evaluator, x_cipher_deserialized, w_encoded)

    s_encoded = helper.encode_outputs(encoder, s_scaled)

    y_cipher.add_plain_inplace(evaluator, s_encoded)

    y_cipher_serialized = helper.serialize_outputs(evaluator, y_cipher)
    y_cipher_deserialized = helper.deserialize_outputs(evaluator, y_cipher_serialized)

    y_scaled = helper.decrypt_outputs(encoder, decryptor, y_cipher_deserialized)

    y = to_decimal(y_scaled, scale**2, plain_modulus, (batch_size, output_dim)) - s

    y_plain = np.matmul(x, w)

    y_diff = np.abs(y - y_plain)
    y_diff_relative = np.max(y_diff) / np.max(np.abs(y_plain))
    print(np.max(y_diff), y_diff_relative)

    