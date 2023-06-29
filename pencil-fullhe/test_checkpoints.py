import torch
import dataloader
import model
from tqdm import tqdm
import numpy as np

def print_accs(x):
    print("[")
    for i, each in enumerate(x):
        if i!=0: print(", ", end="")
        if (i%5==0 and i!=0): print()
        print(f"{each:.4f}", end="")
    print("\n]")

if __name__ == "__main__":

    device = "cuda"
    model_name = "alexnet_classifier"
    torch_model = model.get_model_alexnet_classifier()
    dataset_name, input_shape = model.get_dataset(model_name)
    batchsize = input_shape[0]
    epochs = 10
    tests_per_epoch = 5
    
    train_data, test_data = dataloader.load_dataset(dataset_name)
    # train_data, test_data = dataloader.load_dataset("cifar10-32")
    test_dataloader = dataloader.BatchLoader(test_data, batchsize, True)
    train_dataloader = dataloader.BatchLoader(train_data, batchsize, True)

    torch_model.eval()
    torch_model.to(device)

    accs = []

    for e in range(epochs):
        for j in range(tests_per_epoch):
            step = len(train_dataloader) // tests_per_epoch * (j+1) - 1
            checkpoint_name = f"./checkpoints/priv-{model_name}-epoch{e}-step{step}.pt"
            state_dict = torch.load(checkpoint_name)
            rkey = []
            for key in state_dict.keys():
                if "0.fe" in key:
                    rkey.append(key)
            for key in rkey:
                key_str: str = key
                key_str = key_str.replace("0.fe", "0.model.fe")
                state_dict[key_str] = state_dict[key]
                state_dict.pop(key)
            torch_model.load_state_dict(state_dict)
            
            print(f"Epoch {e} step {step}")
            correct = 0
            total = 0
            for _ in tqdm(range(len(test_dataloader))):
                inputs, labels = test_dataloader.get()
                x = inputs.numpy()
                output = torch_model(torch.tensor(x, dtype=torch.float32, device=device))
                output = output.detach().cpu().numpy()
                labels = labels.numpy()
                pred = np.argmax(output, axis=1)
                correct += np.sum(pred == labels)
                total += len(labels)
            print(f"correct/total = {correct}/{total} = {correct/total}")
            accs.append(correct/total)

    print_accs(accs)