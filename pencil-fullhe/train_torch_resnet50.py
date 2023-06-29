import os
import torchvision
import dataloader
import torch
import torch.utils.data
import torch.optim
import torch.nn.functional
import torch.nn as nn
from tqdm import tqdm
import math
import argparse

def l2_norm(params):
    sums = 0
    for each in params:
        grad = each.grad
        sums += torch.sum(grad*grad)
    return torch.sqrt(sums).item()


class AlexNetFeatureExtractor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fe = torchvision.models.alexnet(True).features
        self.pool = torch.nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        y = torch.flatten(self.pool(self.fe(x)), 1)
        return y

class ResNet50FeatureExtractor(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(True)
        self.fe = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )

    def forward(self, x):
        y = torch.flatten(self.fe(x), 1)
        return y

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, default="resnet50") # preset = model_name
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

    DEVICE = "cuda"

    # load model
    if args.p == "resnet50":
        feature_extractor = ResNet50FeatureExtractor().to(DEVICE)
        feature_dims = 2048
    else:
        feature_extractor = AlexNetFeatureExtractor().to(DEVICE)
        feature_dims = 256*6*6
    feature_extractor.eval()
    model = torch.nn.Sequential(
        nn.Linear(feature_dims, 512),
        nn.ReLU(inplace=True),
        # nn.Dropout(p=dropout),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 10),
    ).to(DEVICE)

    input_shape = (64, 3, 224, 224)
    dataset_name = "cifar10-224"

    batchsize = input_shape[0]
    lr = 1e-2
    estimated_bound = args.C
    noise_level = args.n

    print([r.requires_grad for r in model.parameters()])
    optimizer = torch.optim.SGD(filter(lambda r:r.requires_grad, model.parameters()), lr, momentum=0.8)

    # load data
    train_data, test_data = dataloader.load_dataset(dataset_name)
    def extract_features(dataset):
        print("Extract features")
        ret = []
        i = 0
        while i < len(dataset):
            print(f"\r{i}/{len(dataset)}", end="")
            j = i + 32
            if (j>len(dataset)): j = len(dataset)
            inputs = torch.stack([dataset[id][0] for id in range(i,j)]).to(DEVICE)
            features = feature_extractor(inputs)
            for id in range(i, j):
                ret.append((features[id-i].detach().clone().cpu(), dataset[id][1]))
            i = j
        print("")
        return ret
    train_data = extract_features(train_data)
    test_data = extract_features(test_data)
    
    if SPLIT:
        train_dataloaders = [
            dataloader.BatchLoader(list(filter(lambda sample: sample[1] >= i*2 and sample[1] <= (i*2+1), train_data)), batchsize, True)
            for i in range(5)
        ]
    train_dataloader = dataloader.BatchLoader(train_data, batchsize, True)

    test_dataloader = torch.utils.data.DataLoader(test_data, 32, False)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.6)

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
                if correct/total > best_accuracy: best_accuracy = correct/total
                    
                # model.train()
    print("Best accuracy =", best_accuracy)
    print("Max L2 =", max_actual_l2)
        
        