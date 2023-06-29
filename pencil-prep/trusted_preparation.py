import model
import server_modules
import torch
import torch.nn
import argparse

from crypto_switch import cryptography as crypto

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, default="cifar10_lenet5") # preset = model_name
    args = parser.parse_args()
    model_name = args.p

    model_name, model_torch = model.get_model(model_name)
    dataset_name, input_shape = model.get_dataset(model_name)

    cry = crypto.EvaluationUtils()
    
    server_model = server_modules.server_model_from_pytorch(model_torch, cry, None)
    server_model.trusted_prepare(input_shape)