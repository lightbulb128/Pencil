import model
import dataloader
import torch.utils.data
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math

def clip_image(image):
    minimum = np.min(image)
    maximum = np.max(image)
    return (image-minimum)/(maximum-minimum)

def l2_norm(grads):
    sums = 0
    for each in grads:
        sums += torch.sum(each*each)
    return torch.sqrt(sums).item()

if __name__ == "__main__":

    DEVICE = model.DEVICE

    model_name, model = model.get_model()
    train_data, _ = dataloader.load_dataset("cifar10-32", 100)
    model_checkpoint = False
    batchsize = 1
    estimated_bound = 16
    optimizer_selection = "LBFGS" # Adam, LBFGS
    history_step = 100 if optimizer_selection == "Adam" else 2
    iteration_count = 10000 if optimizer_selection == "Adam" else 100

    if model_checkpoint:
        state_dict = torch.load("./checkpoints/" + model_name)
        model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    train_dataloader = torch.utils.data.DataLoader(train_data, batchsize, False)

    def parameters():
        return list(filter(lambda r:r.requires_grad, model.parameters()))

    def attack(inputs, labels, noise_level=0, save_history=False):

        print()
        
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        gradients = torch.autograd.grad(loss, parameters())
        gradients = list([_.detach().clone() for _ in gradients])

        gradients_l2 = l2_norm(gradients)
        print("Gradient l2 norm =", gradients_l2)
        # assert(gradients_l2 < estimated_bound)

        if noise_level > 0:
            for gradient in gradients:
                gradient += torch.randn(gradient.shape).to(DEVICE) * (noise_level * estimated_bound / math.sqrt(batchsize)) 

        dummy_inputs = torch.randn(inputs.shape).to(DEVICE).requires_grad_(True)
        if optimizer_selection == "Adam":
            optimizer = torch.optim.Adam([dummy_inputs,])
        else:
            optimizer = torch.optim.LBFGS([dummy_inputs,])

        history = []

        for iteration_step in range(iteration_count):

            def closure():
                optimizer.zero_grad()
                dummy_logits = model(dummy_inputs)
                dummy_loss = torch.nn.functional.cross_entropy(dummy_logits, labels)
                dummy_gradients = torch.autograd.grad(dummy_loss, parameters(), create_graph=True)

                grad_diff = 0
                for ograd, dgrad in zip(gradients, dummy_gradients):
                    grad_diff += ((ograd - dgrad) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)
            
            if (iteration_step + 1) % history_step == 0:  
                history.append(dummy_inputs.detach().cpu().numpy())
            
            mse = torch.mean((dummy_inputs - inputs) ** 2).item()
            print(f"\rStep = {iteration_step}, MSE = {mse}", end="")

            if (iteration_step + 1) % (history_step*10) == 0 and save_history:
                plt.figure(figsize=(12,6))
                for i in range(batchsize):
                    plt.subplot(batchsize, 11, 1 + 11 * i)
                    im = np.transpose(inputs[i].detach().cpu().numpy(), (1, 2, 0))
                    plt.imshow(clip_image(im))
                    plt.axis('off')
                    for j in range(10):
                        plt.subplot(batchsize, 11, j + 2 + 11 * i)
                        im = np.transpose(history[j][i], (1, 2, 0))
                        plt.imshow(clip_image(im))
                        plt.axis('off')

                if not os.path.exists("./gradient-matching/"):
                    os.mkdir("./gradient-matching/")
                plt.savefig(f"./gradient-matching/{model_name}-{noise_level}-{optimizer_selection}-step{(iteration_step+1):04d}.png")
                plt.close()
                history = []

        return dummy_inputs.detach().cpu().numpy()

    attack_count = 4
    train_iterator = iter(train_dataloader)
    noise_levels = [0, 1e-4, 5e-4, 1e-3, 1e-2]
    noise_levels_name = ["0", "1e-4", "5e-4", "1e-3", "1e-2"]
    original_data = [next(train_iterator) for i in range(attack_count)]
    original_labels = [a[1] for a in original_data]
    original_images = [a[0] for a in original_data]
    
    def imshow(x):
        if len(x.shape) == 4:
            assert(x.shape[0] == 1)
            x = x[0]
        im = np.transpose(x, (1, 2, 0))
        plt.imshow(clip_image(im))
        plt.xticks([])
        plt.yticks([])
        # plt.axis('off')

    plt.figure(figsize=(12,8))
    for i in range(attack_count):
        plt.subplot(attack_count, 1+len(noise_levels), (1+len(noise_levels)) * i + 1)
        imshow(original_images[i].detach().cpu().numpy())
        if i == attack_count - 1: plt.xlabel("Original", fontsize="xx-large")
    for j, noise_level in enumerate(noise_levels):
        for i in range(attack_count):
            recon = attack(original_images[i], original_labels[i], noise_level)
            plt.subplot(attack_count, 1+len(noise_levels), (1+len(noise_levels)) * i + 2 + j)
            imshow(recon)
            if i == attack_count - 1: plt.xlabel(f"$\sigma$={noise_levels_name[j]}", fontsize="xx-large")
    if not os.path.exists("./gradient-matching/"):
        os.mkdir("./gradient-matching/")
    plt.savefig(f"./gradient-matching/idlg-full.png")
    plt.savefig(f"./gradient-matching/idlg-full.pdf")
    plt.close()
    

            
        





