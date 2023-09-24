import model
import client_modules
import communication_client
from crypto_switch import cryptography
import dataloader
from tqdm import tqdm
import numpy as np
import torch.nn.functional
import time
import argparse

def one_hot(x, dataset_name):
    if "cifar100" in dataset_name:
        classes = 100
    elif "agnews" in dataset_name:
        classes = 4
    else:
        classes = 10
    return torch.nn.functional.one_hot(torch.tensor(x), num_classes=classes).numpy()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str, default="None") # split, "None" to disable, "p:q" to split into p parts, q ratio of total data are sorted
    parser.add_argument("-e", type=int, default=10) # epochs
    args = parser.parse_args()

    print(f"Split   = {args.s}")
    print(f"Epoch   = {args.e}")

    SPLIT = args.s != "None"
    if SPLIT:
        x = args.s.split(":")
        SPLIT_JOIN_COUNT = int(x[0])
        SPLIT_SORTED_RATIO = float(x[1])

    # establish connection
    comm = communication_client.ClientCommunication()
    comm.connect()
    print("Client: Connection established.")

    # create crypto
    crypto = cryptography.EncryptionUtils()
    comm.send(crypto.generate_keys())
    print("Client: Cryptography context created.")
    
    # load model
    model_description = comm.recv()
    model = client_modules.client_model_from_description(model_description, crypto, comm)
    print("Client: Model created from description successfully.")

    # prepare
    dataset_name, input_shape = comm.recv()
    batchsize = input_shape[0]
    model.prepare(input_shape)
    print("Client: Model preprocessing finished.")
    print("dataset name = ", dataset_name);

    # load data
    train_data, test_data = dataloader.load_dataset(dataset_name)
    # train_data, test_data = dataloader.load_dataset("cifar10-32")
    test_dataloader = dataloader.BatchLoader(test_data, batchsize, True)
    if SPLIT:
        # take sorted and unsorted
        split_index = int(len(train_data) * SPLIT_SORTED_RATIO)
        sorted_data = train_data[:split_index]
        unsorted_data = train_data[split_index:]
        sorted_data.sort(key=lambda x: x[1])
        # split into SPLIT_JOIN_COUNT parts
        train_dataloaders = []
        for i in range(SPLIT_JOIN_COUNT):
            lower = int(i * len(sorted_data) / SPLIT_JOIN_COUNT)
            upper = int((i + 1) * len(sorted_data) / SPLIT_JOIN_COUNT)
            current_data = sorted_data[lower:upper] + unsorted_data[i::SPLIT_JOIN_COUNT]
            train_dataloaders.append(dataloader.BatchLoader(current_data, batchsize, True))
        train_dataloader = dataloader.BatchLoader(train_data, batchsize, True)
    else:
        train_dataloader = dataloader.BatchLoader(train_data, batchsize, True)
    print("Client: Train dataset loaded")

    # train
    for epoch in range(args.e):
        epoch_time = 0
        print(f"Epoch = {epoch}")
        for step in tqdm(range(len(train_dataloader))):
            # if step > 50: break
            timer = time.time()
            skip_train_step = False
            if SPLIT:
                inputs, labels = train_dataloaders[step % SPLIT_JOIN_COUNT].get()
                print("Training from split {0}, with labels {1}".format(step % SPLIT_JOIN_COUNT, labels))
            else:
                inputs, labels = train_dataloader.get()
            if not skip_train_step:
                comm.send("train")
                inputs = inputs.numpy()
                labels = labels.numpy()
                x = crypto.to_field(inputs)
                output = model.forward(x)
                output_another_share = comm.recv()
                output = crypto.field_mod(output + output_another_share)
                output = crypto.to_decimal(output, crypto.default_scale() ** 2, (batchsize, -1))
                loss = torch.nn.functional.cross_entropy(torch.tensor(output), torch.tensor(labels))
                
                pred = np.argmax(output, axis=1)
                correct = np.sum(pred == labels)
                total = len(labels)

                print(f"Loss = {loss:.6f}", "logit max =", np.max(np.abs(output)), f"correct={correct}/{total}")
                output_softmax = np.exp(output) / np.reshape(np.sum(np.exp(output), axis=1), (-1, 1))
                output_grad = (output_softmax - one_hot(labels, dataset_name)) / len(labels)
                output_grad = crypto.to_field(output_grad)
                model.backward(output_grad)
                epoch_time += time.time() - timer
            if (step + 1) % (len(train_dataloader) // 5) == 0:
                comm.send("anneal")
                print("testing")
                comm.send("test")
                correct, total = comm.recv()
                print(f"correct/total = {correct}/{total} = {correct/total}")
                comm.send("save")
                name = f"epoch{epoch}-step{step}"
                if SPLIT: name += f"-split-join{SPLIT_JOIN_COUNT}"
                comm.send(name)
        print("Epoch time =", epoch_time)
    comm.send("finish")

    # close connection
    comm.close_connection()
