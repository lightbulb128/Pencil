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
import torch_models

def create_shares(x):
    m = np.max(x)
    r = np.random.uniform(-m, m, x.shape)
    return r, x-r

def absmax(x):
    return np.max(np.abs(x))

def absmean(x):
    return np.mean(np.abs(x))

if __name__ == "__main__":

    comm = communication_server.ServerCommunication()
    comm.listen()
    comm.accept_connection()
    cryp = crypto.EvaluationUtils()
    cryp.receive_keys(comm.recv())

    start_scale = cryp.default_scale()
    forward_only = False

    # input_shape = (64, 2048)
    # model_name, model_torch = "", torch.nn.Linear(2048, 1001)

    # input_shape = (64, 3, 32, 32)
    # model_name, model_torch = "", torch.nn.Sequential( # (3, 32, 32)
    #     torch.nn.Conv2d(3, 64, 5, padding=2), # (64, 32, 32)
    #     torch.nn.ReLU(),
    #     torch.nn.AvgPool2d(2, 2), # (64, 16, 16)
    #     torch.nn.Conv2d(64, 64, 5, padding=2), # (64, 16, 16)
    #     torch.nn.ReLU(),
    #     torch.nn.AvgPool2d(2, 2), # (64, 8, 8)
    #     torch.nn.Conv2d(64, 64, 3, padding=1), # (64, 8, 8)
    #     torch.nn.ReLU(),
    #     torch.nn.Conv2d(64, 64, 1), # (64, 8, 8)
    #     torch.nn.ReLU(),
    #     torch.nn.Conv2d(64, 16, 1), # (16, 8, 8)
    #     torch.nn.ReLU(),
    #     torch.nn.Flatten(),
    #     torch.nn.Linear(1024, 10)
    # )
    
    # input_shape = (64, 512)
    # model_name, model_torch = "", torch.nn.ReLU()

    # input_shape = (1,)
    # model_name, model_torch = "", torch.nn.GELU()
    # start_scale = cryp.default_scale() ** 2

    # input_shape = (2, 2)
    # model_name, model_torch = "", torch.nn.Linear(2, 2)

    # input_shape = (64, 64, 56, 56)
    # model_name, model_torch = "", torch.nn.Conv2d(64, 3, 3)

    # input_shape = (64, 64, 56, 56)
    # model_name, model_torch = "", torch.nn.Conv2d(64, 3, 3)
    
    # input_shape = (4, 64, 56, 56)
    # model_name, model_torch = "", torch.nn.Conv2d(64, 3, 3)

    # input_shape = (4, 8, 56)
    # model_name, model_torch = "", torch.nn.Conv1d(8, 16, 3)

    # input_shape = (3, 10)
    # model_name, model_torch = "", torch.nn.ReLU()
    # start_scale = cryp.default_scale() ** 2

    # input_shape = (128, 1, 128)
    # model_name, model_torch = "", torch.nn.LayerNorm(128)
    # model_torch.weight.data = torch.randn_like(model_torch.weight.data)
    # model_torch.bias.data = torch.randn_like(model_torch.bias.data)
    # start_scale = cryp.default_scale()

    # input_shape = (128, 1, 128) # [N, B, E]
    # model_name, model_torch = "", torch_models.MultiheadAttention(128, 2, dropout=0)
    # model_torch = torch_models.Residual(model_torch) # Note: input to residual must be same shape as output, and have f precision

    # input_shape = (128, 1, 128) # [N, B, E]
    # model_name, model_torch = "", torch.nn.TransformerEncoderLayer(128, 2, 512, 0, "gelu")


    # input_shape = (3, 4, 10)
    # model_name, model_torch = "", torch.nn.Softmax(dim=-1)

    # input_shape = (1, 1, 3)
    # model_name, model_torch = "", torch_models.ScaledDotProductAttention(dropout=0)

    # input_shape = (128, 1, 128) # [N, B, E]
    # model_name, model_torch = "", torch_models.MultiheadAttention(128, 2, dropout=0)

    # input_shape = (1, 16, 22, 22)
    # model_name, model_torch = "", torch.nn.Conv2d(16, 1, 5)
    # input_shape = (1, 1, 5, 5)
    # model_name, model_torch = "", torch.nn.Conv2d(1, 1, 3)

    # input_shape = (4, 3, 7, 7)
    # model_name, model_torch = "", torch.nn.Conv2d(3, 5, 3, stride=2, padding=3)

    # input_shape = (16, 3, 224, 224)
    # model_name, model_torch = "", torch.nn.Conv2d(3, 3, 7, stride=2, padding=3)

    # input_shape = (16, 64, 56, 56)
    # model_name, model_torch = "", torch.nn.Conv2d(64, 256, 1)

    # input_shape = (8, 16, 64, 64)
    # model_name, model_torch = "", torch.nn.BatchNorm2d(16)

    # input_shape = (4, 8, 6, 9)
    # model_name, model_torch = "", torch.nn.AvgPool2d(3, 2, 1)

    input_shape = (4, 8, 6)
    model_name, model_torch = "", torch.nn.AvgPool1d(3, 2, 1)
    
    # input_shape = (4, 4)
    # model_name, model_torch = "", torch.nn.Dropout(0.5)

    # input_shape = (1, 3, 224, 224)
    # model_name, model_torch = "", torch_models.TorchNative("alexnet-fe")

    if not isinstance(model_torch, torch.nn.Sequential):
        model_torch = torch.nn.Sequential(model_torch,)

    comm.send(input_shape)
    
    model = server_modules.server_model_from_pytorch(model_torch, cryp, comm)
    comm.send(model.describe())

    optimize = len(model.parameters()) != 0
    
    if optimize:
        optim_torch = torch.optim.SGD(model_torch.parameters(), 1e-3, momentum=0.8)
        optim_priv = optimizer.SGDMomentum(model.parameters(), 1e-3, 0.8)

    
    model.prepare(input_shape)
    print("Prepared")

    steps = 1
    if forward_only: steps = 1
    comm.send(steps)
    comm.send(forward_only)

    SCALE = cryp.default_scale()
    DOUBLE_SCALE = SCALE * SCALE

    for step in range(steps):
        inputs = (np.random.rand(*input_shape) - 0.5)

        inputs_field = cryp.to_field(inputs, start_scale)
        inputs_a = cryp.field_random_mask(inputs_field.size)
        inputs_b = cryp.field_add(inputs_field, cryp.field_negate(inputs_a))
        comm.send(inputs_b)

        timed = time.time()
        outputs_a = inputs_a
        outputs_list = [outputs_a,]
        for i, layer in enumerate(model.items):
            print(f"Forward layer {i}")
            outputs_a = layer.forward(outputs_a)
            outputs_reconstructed = cryp.field_mod(outputs_a + comm.recv())
            if isinstance(layer, (
                server_modules.ReLU, server_modules.AvgPool2d, server_modules.AvgPool1d, 
                server_modules.Flatten, server_modules.Softmax,
                server_modules.MultiheadAttention, server_modules.Residual,
                server_modules.TransformerEncoderLayer,
                server_modules.GELU
            )):
                outputs_list.append(cryp.to_decimal(outputs_reconstructed, SCALE))
            else:
                outputs_list.append(cryp.to_decimal(outputs_reconstructed, DOUBLE_SCALE))
        print("forward time =", time.time() - timed)

        input_tensor = torch.tensor(inputs, requires_grad=True, dtype=torch.float32)
        output_tensor = input_tensor
        output_tensor_list = [output_tensor, ]
        for layer in model_torch.children():
            output_tensor = layer(output_tensor)
            output_tensor.requires_grad_(True)
            output_tensor.retain_grad()
            output_tensor_list.append(output_tensor)

        for i in range(1, len(outputs_list)):
            oreshaped = np.reshape(outputs_list[i], output_tensor_list[i].shape)
            d = oreshaped - output_tensor_list[i].detach().numpy()
            print(f"i={i}", "D(y) ", absmax(d), "[ max", absmax(output_tensor_list[i].detach().numpy()), ']', "mean", absmean(d))
            # print(oreshaped)
            # print(output_tensor_list[i].detach().numpy())

        if forward_only: break

        partial_output = np.random.rand(*output_tensor.shape) - 0.5

        partial_output_field = cryp.to_field(partial_output, SCALE)
        partial_output_a = cryp.field_random_mask(partial_output_field.size)
        partial_output_b = cryp.field_add(partial_output_field, cryp.field_negate(partial_output_a))
        comm.send(partial_output_b)


        if optimize:
            optim_priv.zero_grad()

        backward_time = time.time()
        partials_list = []
        for i, layer in enumerate(reversed(model.items)):
            print(f"Backward layer {i}")
            partial_output_a = layer.backward(partial_output_a)
            partial_reconstructed = cryp.field_mod(partial_output_a + comm.recv())
            partials_list.append(cryp.to_decimal(partial_reconstructed, SCALE))
        
        print("backward time =", time.time() - backward_time)

        if optimize:
            optim_torch.zero_grad()
        output_tensor.backward(torch.tensor(partial_output, dtype=torch.float32))

        for i in range(len(outputs_list) - 1):
            partial_torch = output_tensor_list[len(output_tensor_list) - i - 2].grad
            partial_reshaped = np.reshape(partials_list[i], partial_torch.shape)
            d = partial_reshaped - partial_torch.detach().numpy()
            print(f"i={i}", "D(dx)", absmax(d), "[ max", absmax(partial_torch.detach().numpy()), ']', "mean", absmean(d))

        
        print("total time =", time.time() - timed)

        def compare(model, model_torch: torch.nn.Module):

            if isinstance(model, (server_modules.Sequential)):

                model_torch_childen = list(model_torch.children())
                for i, model_item in enumerate(model.items):
                    compare(model_item, model_torch_childen[i])

            else:

                if isinstance(model, (
                    server_modules.ReLU, server_modules.AvgPool2d, server_modules.AvgPool1d, 
                    server_modules.Flatten, server_modules.Softmax,
                    server_modules.ScaledDotProductAttention,
                )):
                    return
                
                if isinstance(model, server_modules.Residual):
                    compare(model.branch_module, model_torch.module)
                    return
                
                if isinstance(model, server_modules.TransformerEncoderLayer):
                    compare(model.res1.branch_module.items[0], model_torch.self_attn)
                    compare(model.norm1, model_torch.norm1)
                    compare(model.res2.branch_module.items[0], model_torch.linear1)
                    compare(model.res2.branch_module.items[2], model_torch.linear2)
                    compare(model.norm2, model_torch.norm2)
                    return
                
                if isinstance(model, server_modules.MultiheadAttention):
                    mWin = model.in_linear.weight.grad
                    mbin = model.in_linear.bias.grad
                    mWout = model.out_linear.weight.grad
                    mbout = model.out_linear.bias.grad
                    if isinstance(model_torch, torch_models.MultiheadAttention):
                        model_torch = model_torch.inner
                    assert(isinstance(model_torch, torch.nn.MultiheadAttention))
                    dWin = mWin - model_torch.in_proj_weight.grad.data.detach().numpy()
                    dbin = mbin - model_torch.in_proj_bias.grad.data.detach().numpy()
                    dWout = mWout - model_torch.out_proj.weight.grad.data.detach().numpy()
                    dbout = mbout - model_torch.out_proj.bias.grad.data.detach().numpy()
                    print("D(dWin)", absmax(dWin), "[ max", absmax(mWin), "]", "mean", absmean(dWin))
                    print("D(dbin)", absmax(dbin), "[ max", absmax(mbin), "]", "mean", absmean(dbin))
                    print("D(dWout)", absmax(dWout), "[ max", absmax(mWout), "]", "mean", absmean(dWout))
                    print("D(dbout)", absmax(dbout), "[ max", absmax(mbout), "]", "mean", absmean(dbout))
                    return
                
                if isinstance(model, server_modules.Conv1d):
                    mW = model.inner.weight.grad
                    d0, d1, d2, d3 = mW.shape
                    assert(d3 == 1)
                    mW = np.reshape(mW, (d0, d1, d2))
                    mb = model.inner.bias.grad
                elif isinstance(model, (server_modules.Conv2dStrided, server_modules.Conv2dPadded)):
                    mW = model.merged_weight_grad()
                    mb = model.merged_bias_grad()
                else:
                    mW = model.weight.grad
                    mb = model.bias.grad
                dW = mW - model_torch.weight.grad.data.detach().numpy()
                # print("torch W =", model_torch.weight.grad.data.detach().numpy())
                print("D(dW)", absmax(dW), "[ dW max", absmax(mW), "]", "mean", absmean(dW))
                db = mb - model_torch.bias.grad.data.detach().numpy()
                print("D(db)", absmax(db), "[ db max", absmax(mb), "]", "mean", absmean(db))

        compare(model, model_torch)
    
        if optimize:
            optim_priv.step()
            optim_torch.step()

    comm.close_connection()