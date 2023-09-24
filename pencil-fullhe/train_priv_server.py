import os
import model
import server_modules
import communication_server
from crypto_switch import cryptography
import torch
import optimizer
import numpy as np
import dataloader
from tqdm import tqdm
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, default="mnist_aby3") # preset = model_name
    parser.add_argument("-n", type=float, default=0) # dp noise levels, 0 to disable
    parser.add_argument("-C", type=float, default=8) # estimated gradient bound
    parser.add_argument("-o", type=str, default="SGD") # optimizer
    parser.add_argument("-om", type=float, default=0.8) # optimizer momentum, only for SGD
    parser.add_argument("-lr", type=float, default=1e-2) # learning rate
    args = parser.parse_args()
    model_name = args.p

    print(f"Model   = {args.p}")
    print(f"Noise   = {args.n}")
    print(f"Cbound  = {args.C}")
    print(f"All args = {args}")

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
    model_name, modelx = model.get_model(args.p)
    # modelx.load_state_dict(torch.load("./checkpoints/priv-sphinx_cifar10-epoch1-step779-split-join4.pt"))
    modelx = server_modules.server_model_from_pytorch(modelx, crypto, comm)
    model_description = modelx.describe()
    comm.send(model_description)
    print("Server: Model created and loaded successfully.")

    dataset_name, input_shape = model.get_dataset(model_name)
    comm.send((dataset_name, input_shape))
    batchsize = input_shape[0]
    model = modelx
    model.prepare(input_shape)
    print("Server: Model preprocessing finished.")

    _, test_data = dataloader.load_dataset(dataset_name)
    test_dataloader = dataloader.BatchLoader(test_data, batchsize, True)
    print("Server: Test dataset loaded")

    # set up optimizer
    optimizer_name = args.o
    optimizer_momentum = args.om
    learning_rate = args.lr
    if optimizer_name == "SGD":
        if optimizer_momentum == 0:
            optim = optimizer.SGD(model.parameters(), learning_rate)
        else:
            optim = optimizer.SGDMomentum(model.parameters(), learning_rate, optimizer_momentum)
    elif optimizer_name == "Adam":
        optim = optimizer.Adam(model.parameters(), learning_rate)
    optim.silence()
    optim.estimated_bound = args.C
    optim.noise_level = args.n
    optim.batchsize = batchsize

    # serve
    while True:
        direction = comm.recv()
        if direction == "finish":
            break

        if direction == "test":
            model.eval()
            torch_model = model.to_torch()
            correct = 0
            total = 0
            for _ in tqdm(range(len(test_dataloader))):
                inputs, labels = test_dataloader.get()
                x = inputs.numpy()
                output = torch_model(torch.tensor(x, dtype=torch.float32))
                output = output.detach().cpu().numpy()
                labels = labels.numpy()
                pred = np.argmax(output, axis=1)
                correct += np.sum(pred == labels)
                total += len(labels)
            print(f"correct/total = {correct}/{total} = {correct/total}")
            comm.send((correct, total))

        if direction == "train":
            model.train()
            x = np.zeros(np.product(input_shape), dtype=np.uint64)
            output = model.forward(x)
            comm.send(output)
            optim.zero_grad()
            model.backward(np.zeros_like(output))
            optim.step()
        if direction == "anneal":
            optim.learning_rate *= 1

        if direction == "save":
            name = comm.recv()
            torch_model = model.to_torch()
            torch.save(torch_model.state_dict(), f"checkpoints/priv-{model_name}-{name}.pt")

    # close connection
    comm.close_connection()