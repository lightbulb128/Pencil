import os
import torchvision
import model
from model import DEVICE
import dataloader
import torch
import torch.utils.data
import torch.optim
import torch.nn.functional
from tqdm import tqdm
import math
import argparse

def l2_norm(params):
    sums = 0
    for each in params:
        grad = each.grad
        sums += torch.sum(grad*grad)
    return torch.sqrt(sums).item()

if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, default="cifar10_lenet5") # preset = model_name
    parser.add_argument("-s", type=int, default=0) # split count, 0 to disable
    parser.add_argument("-n", type=float, default=0) # dp noise levels, 0 to disable
    parser.add_argument("-C", type=float, default=8) # estimated gradient bound
    parser.add_argument("-e", type=int, default=10) # epochs
    args = parser.parse_args()
    model_name = args.p

    print(f"Model   = {args.p}")
    print(f"Split   = {args.s}")
    print(f"Epoch   = {args.e}")
    print(f"Noise   = {args.n}")
    print(f"Cbound  = {args.C}")

    SPLIT = args.s != 0
    SPLIT_JOIN_COUNT = args.s

    # load model
    load_checkpoint = ""
    model_name, modelx = model.get_model(args.p)
    dataset_name, input_shape = model.get_dataset(model_name)
    model = modelx
    if load_checkpoint != "":
        state_dict = torch.load(load_checkpoint)
        model.load_state_dict(state_dict)
    model = model.to(DEVICE)

    batchsize = input_shape[0]
    lr = 1e-2
    estimated_bound = args.C
    noise_level = args.n

    print([r.requires_grad for r in model.parameters()])
    optimizer = torch.optim.SGD(filter(lambda r:r.requires_grad, model.parameters()), lr, momentum=0.8)

    # load data
    train_data, test_data = dataloader.load_dataset(dataset_name)
    # train_dataloader = torch.utils.data.DataLoader(train_data, batchsize, True)
    
    if SPLIT:
        train_dataloaders = [
            dataloader.BatchLoader(list(filter(lambda sample: sample[1] >= i*2 and sample[1] <= (i*2+1), train_data)), batchsize, True)
            for i in range(5)
        ]
    train_dataloader = dataloader.BatchLoader(train_data, batchsize, True)

    test_dataloader = torch.utils.data.DataLoader(test_data, 32, False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.6)

    # train
    # model.train()
    best_accuracy = 0
    max_l2 = estimated_bound
    max_actual_l2 = 0
    for epoch in range(args.e):
        print(f"Epoch {epoch}")
        # print([r.requires_grad for r in model.parameters()])
        for step in tqdm(range(len(train_dataloader))):
            # if step > 50: break
            optimizer.zero_grad()
            skip_train_step = False
            if SPLIT:
                if step % 5 >= SPLIT_JOIN_COUNT: skip_train_step = True
                inputs, labels = train_dataloaders[step % 5].get()
            else:
                inputs, labels = train_dataloader.get()
            
            if not skip_train_step:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(inputs)
                # print("output", float(torch.max(logits)))
                loss = torch.nn.functional.cross_entropy(logits, labels)
                # print("loss =", float(loss))
                loss.backward()

                if noise_level > 0:
                    params = list(filter(lambda r:r.requires_grad, model.parameters()))
                    param_l2 = l2_norm(params)
                    if param_l2 > max_l2:
                        max_l2 = param_l2 * 2
                        print("l2 =", max_l2)
                    if param_l2 > max_actual_l2:
                        max_actual_l2 = param_l2
                    for param in params:
                        param.grad += torch.randn(param.shape, device=param.grad.device) * (noise_level * estimated_bound / math.sqrt(batchsize)) 

                optimizer.step()

            if (step + 1) % (len(train_dataloader) // 5) == 0:
                # model.eval()
                correct = 0
                total = 0
                for inputs, labels in test_dataloader:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    logits = model(inputs)
                    _, pred = torch.max(logits, dim=1)
                    total += len(labels)
                    correct += torch.sum(pred==labels)
                print(f"Acc = {correct/total:.4f}")

                accuracy = correct / total
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    # save model
                    if not os.path.exists("./checkpoints"):
                        os.mkdir("./checkpoints")
                    torch.save(model.state_dict(), os.path.join("./checkpoints", model_name))
                    
                # model.train()
    print("Best accuracy =", best_accuracy)
    print("Max L2 =", max_actual_l2)
        