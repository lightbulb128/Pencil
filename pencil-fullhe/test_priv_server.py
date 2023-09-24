import os
import model
import server_modules
import communication_server
from crypto_switch import cryptography
import torch
import argparse

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, default="mnist_aby3") # preset = model_name
    args = parser.parse_args()
    model_name = args.p

    # establish connection
    comm = communication_server.ServerCommunication()
    comm.listen()
    comm.accept_connection()
    print("Server: Connection established.")

    # create crypto
    crypto = cryptography.EvaluationUtils()
    crypto.receive_keys(comm.recv())
    print("Server: Cryptography context created.")
    
    # load model
    model_name, model = model.get_model(args.p)
    model.load_state_dict(torch.load(os.path.join("./checkpoints", model_name)))
    model = server_modules.server_model_from_pytorch(model, crypto, comm)
    model_description = model.describe()
    comm.send(model_description)
    print("Server: Model created and loaded successfully.")

    # serve
    while True:
        direction = comm.recv()
        if direction == "finish":
            break
        if direction == "test":
            x = comm.recv()
            output = model.forward(x)
            comm.send(output)

    # close connection
    comm.close_connection()