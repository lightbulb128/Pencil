import model
import server_modules
from crypto_switch import cryptography
import dataloader
from tqdm import tqdm
import numpy as np
import torch.nn.functional
import time
import optimizer
import config

def one_hot(x):
    return torch.nn.functional.one_hot(torch.tensor(x), num_classes=10).numpy()

if __name__ == "__main__":

    SPLIT = False
    SPLIT_JOIN_COUNT = 4

    print(f"SPLIT = {SPLIT}, SPLIT_JOIN_COUNT = {SPLIT_JOIN_COUNT}")

    # create crypto
    crypto = cryptography.EvaluationUtils()
    
    # load model
    model_name, modelx = model.get_model()
    # modelx.load_state_dict(torch.load("./checkpoints/priv-sphinx_cifar10-epoch1-step779-split-join4.pt"))
    modelx = server_modules.server_model_from_pytorch(modelx, crypto, None)
    print("Client: Model created from description successfully.")

    # prepare
    dataset_name, input_shape = model.get_dataset(model_name)
    batchsize = input_shape[0]
    model = modelx
    model.prepare(input_shape)
    print("Client: Model preprocessing finished.")

    # load data
    train_data, test_data = dataloader.load_dataset(dataset_name)
    # train_data, test_data = dataloader.load_dataset("cifar10-32")
    test_dataloader = dataloader.BatchLoader(test_data, batchsize, True)
    if SPLIT:
        train_dataloaders = [
            dataloader.BatchLoader(list(filter(lambda sample: sample[1] >= i*2 and sample[1] <= (i*2+1), train_data)), batchsize, True)
            for i in range(5)
        ]
        train_dataloader = dataloader.BatchLoader(train_data, batchsize, True)
    else:
        train_dataloader = dataloader.BatchLoader(train_data, batchsize, True)

    optim = optimizer.SGDMomentum(model.parameters(), 5e-2, 0.8)
    optim.silence()

    # train
    for epoch in range(0, 10):
        epoch_time = 0
        print(f"Epoch = {epoch}")
        for step in tqdm(range(len(train_dataloader))):
            # if step > 50: break
            timer = time.time()
            skip_train_step = False
            if SPLIT:
                if step % 5 >= SPLIT_JOIN_COUNT: skip_train_step = True
                inputs, labels = train_dataloaders[step % 5].get()
            else:
                inputs, labels = train_dataloader.get()
            if not skip_train_step:
                inputs = inputs.numpy()
                labels = labels.numpy()
                x = crypto.to_field(inputs)
                output = crypto.field_mod(model.forward_plain(x))
                output = crypto.to_decimal(output, crypto.default_scale() ** 2, (batchsize, -1))
                loss = torch.nn.functional.cross_entropy(torch.tensor(output), torch.tensor(labels))
                
                if step % 100 == 0:
                    pred = np.argmax(output, axis=1)
                    correct = np.sum(pred == labels)
                    total = len(labels)
                    print(f"Loss = {loss:.6f}", "logit max =", np.max(np.abs(output)), f"correct={correct}/{total}")
                
                output_softmax = np.exp(output) / np.reshape(np.sum(np.exp(output), axis=1), (-1, 1))
                output_grad = (output_softmax - one_hot(labels)) / len(labels)
                output_grad = crypto.to_field(output_grad)
                optim.zero_grad()
                model.backward_plain(output_grad)
                optim.step()
                epoch_time += time.time() - timer

            if (step + 1) % (len(train_dataloader) // 5) == 0:

                model.eval()
                torch_model = model.to_torch().to(device=config.DEVICE)
                correct = 0
                total = 0
                for _ in tqdm(range(len(test_dataloader))):
                    inputs, labels = test_dataloader.get()
                    x = inputs.numpy()
                    output = torch_model(torch.tensor(x, dtype=torch.float32, device=config.DEVICE))
                    output = output.detach().cpu().numpy()
                    labels = labels.numpy()
                    pred = np.argmax(output, axis=1)
                    correct += np.sum(pred == labels)
                    total += len(labels)
                print(f"correct/total = {correct}/{total} = {correct/total}")
                
                name = f"epoch{epoch}-step{step}"
                if SPLIT: name += f"-split-join{SPLIT_JOIN_COUNT}"
                torch.save(torch_model.state_dict(), f"checkpoints/priv-{model_name}-{name}.pt")

        print("Epoch time =", epoch_time)

