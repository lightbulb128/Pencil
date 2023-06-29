import server_modules
import communication_server
from crypto_switch import cryptography as crypto
import numpy as np
import time
import torch
import torch.nn
import torch.nn.functional
import torch.optim
import optimizer
import model
import dataloader
import config
import argparse

def create_shares(x):
    r = np.random.uniform(-1, 1, x.shape)
    return r, x-r

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, default="cifar10_lenet5") # preset = model_name
    args = parser.parse_args()
    model_name = args.p

    print(f"Model   = {args.p}")

    comm = communication_server.ServerCommunication()
    comm.listen()
    comm.accept_connection()
    cryp = crypto.EvaluationUtils()
    cryp.receive_keys(comm.recv())

    model_name, model_torch = model.get_model(model_name)
    dataset_name, input_shape = model.get_dataset(model_name)
    model_torch = model_torch.to(config.DEVICE)
    model_priv = server_modules.server_model_from_pytorch(model_torch, cryp, comm)
    comm.send(model_priv.describe())

    comm.send((dataset_name, input_shape))
    batchsize = input_shape[0]
    timed = time.time()
    model_priv.prepare(input_shape)
    print("Prepare used {0}s".format(time.time() - timed))
    print("Prepared")
    
    # optim_torch = torch.optim.SGD(model_torch.parameters(), 1e-3)
    # optim_priv = optimizer.SGD(model_priv.parameters(), 1e-3)
    
    optim_torch = torch.optim.SGD(model_torch.parameters(), 1e-2, momentum=0.8)
    optim_priv = optimizer.SGDMomentum(model_priv.parameters(), 1e-2, 0.8)

    # optim_torch = torch.optim.Adam(model_torch.parameters(), 1e-2, eps=1e-4)
    # optim_priv = optimizer.Adam(model_priv.parameters(), 1e-2, eps=1e-4)

    steps = comm.recv()

    for step in range(steps):
        print(f"Step = {step}")

        x = np.zeros(input_shape)
        x = cryp.to_field(x)
        timed = time.time()
        output = model_priv.forward(x)
        comm.send(output)

        partial = np.zeros(output.shape)
        partial = cryp.to_field(partial)
        optim_priv.zero_grad()
        x_grad = model_priv.backward(partial)
        optim_priv.step()
        print("Time =", time.time()-timed)

        inputs = torch.tensor(comm.recv(), device=config.DEVICE, requires_grad=True)
        labels = torch.tensor(comm.recv(), device=config.DEVICE)
        output_torch = model_torch(inputs)
        loss = torch.nn.functional.cross_entropy(output_torch, labels)
        optim_torch.zero_grad()
        loss.backward()
        optim_torch.step()

        def absmax(r): return np.max(np.abs(r))
        output = comm.recv()
        print("y diff = ", absmax(output - output_torch.detach().cpu().numpy()))
        if not (x_grad is None):
            x_grad = cryp.field_mod(x_grad + comm.recv())
            x_grad = cryp.to_decimal(x_grad, shape=input_shape)
            print("dx diff =", absmax(x_grad - inputs.grad.detach().cpu().numpy()))
        if False:
            for p1, p2 in zip(model_priv.parameters(), model_torch.parameters()):
                print("  grad diff = ", absmax(p1.grad - p2.grad.detach().cpu().numpy()));
                print("  val diff = ", absmax(p1.value - p2.data.detach().cpu().numpy()), p1.value.shape)

    
    def absmax(r): return np.max(np.abs(r))

    model_retorch = model_priv.to_torch().to(config.DEVICE)
    model_priv.eval()
    model_torch.eval()
    model_retorch.eval()
    print("test he")
    while True:
        direction = comm.recv()
        if direction == "finish":
            break
        if direction == "test":
            x = comm.recv()
            output = model_priv.forward_plain(x)
            output_torch = model_torch(torch.tensor(x, device=config.DEVICE)).detach().cpu().numpy()
            output_retorch = model_retorch(torch.tensor(x, device=config.DEVICE)).detach().cpu().numpy()

            # print("\npredict result")
            # print("retorch")
            # print(output_retorch)
            # print("torch")
            # print(output_torch)
            # print("output")
            # print(output)
            print("  logit diff (r-t) = ", absmax(output_retorch - output_torch))
            d = np.argmax(output_retorch, axis=1) - np.argmax(output_torch, axis=1)
            d = np.sum(d!=0)
            print("  pred diff (r-t) = ", d)

            print("  logit diff (r-p) = ", absmax(output_retorch - output))
            d = np.argmax(output_retorch, axis=1) - np.argmax(output, axis=1)
            d = np.sum(d!=0)
            print("  pred diff (r-p) = ", d)
            comm.send(output)

    


    comm.close_connection()