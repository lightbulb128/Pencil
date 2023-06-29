import client_modules
import communication_client
from crypto_switch import cryptography as crypto
import numpy as np
import dataloader

import torch.nn.functional

def one_hot(x):
    return torch.nn.functional.one_hot(torch.tensor(x), num_classes=10).numpy()

if __name__ == "__main__":
    comm = communication_client.ClientCommunication()
    comm.connect()
    cryp = crypto.EncryptionUtils()
    comm.send(cryp.generate_keys())

    model_description = comm.recv()
    model = client_modules.client_model_from_description(model_description, cryp, comm)

    dataset_name, input_shape = comm.recv()
    batchsize = input_shape[0]
    model.prepare(input_shape)

    train_data, test_data = dataloader.load_dataset(dataset_name)
    test_dataloader = dataloader.BatchLoader(test_data, batchsize, True)
    train_dataloader = dataloader.BatchLoader(train_data, batchsize, True)
    print("Data loaded")

    steps = 1
    comm.send(steps)
    
    for step in range(steps):

        comm.clear_accumulation()
        comm.counter += 1

        print(f"Step = {step}")

        inputs, labels = train_dataloader.get()
        inputs = inputs.numpy()
        labels = labels.numpy()
        x = inputs
        x = cryp.to_field(x)
        output_my_share = model.forward(x)
        output_another_share = comm.recv()
        output = cryp.field_mod(output_my_share + output_another_share)
        output = cryp.to_decimal(output, cryp.default_scale() ** 2, (batchsize, -1))
        # output = np.where(output > 50, 50, output)
        # output = np.where(output < -50, -50, output)
        
        pred = np.argmax(output, axis=1)
        correct = np.sum(pred == labels)
        total = len(labels)

        loss = torch.nn.functional.cross_entropy(torch.tensor(output), torch.tensor(labels))
        print(f"Loss = {loss:.6f}", "logit max =", np.max(np.abs(output)), f"correct={correct}/{total}")
        output_softmax = np.exp(output) / np.reshape(np.sum(np.exp(output), axis=1), (-1, 1))
        output_grad = (output_softmax - one_hot(labels)) / len(labels)
        output_grad_field = cryp.to_field(output_grad)
        x_grad = model.backward(output_grad_field)

        print("Transmission (MB):", comm.get_average_transmission())

        comm.send(inputs)
        comm.send(labels)

        comm.send(output)
        if not (x_grad is None):
            comm.send(x_grad)

    
    print("test he")
    correct = 0
    total = 0
    # for _ in range(len(test_dataloader)):
    for _ in range(0):
        comm.send("test")
        inputs, labels = test_dataloader.get()
        inputs = inputs.numpy()
        labels = labels.numpy()
        comm.send(inputs)
        output = comm.recv()
        pred = np.argmax(output, axis=1)
        correct += np.sum(pred == labels)
        total += len(labels)
    if total != 0:
        print(f"correct/total = {correct}/{total} = {correct/total}")
    comm.send("finish")

    comm.close_connection()