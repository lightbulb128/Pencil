import model
from server_modules_switch import server_modules
import torch
import torch.nn

if __name__ == "__main__":

    # input_shape = (64, 3, 224, 224)
    # model_name, model_torch = model.get_model()
    # input_shape = (32, 1, 28, 28)
    # model_name, model_torch = model.get_model()
    input_shape = (64, 3, 32, 32)
    model_name, model_torch = model.get_model()
    
    
    server_model = server_modules.server_model_from_pytorch(model_torch, None, None)
    server_model.trusted_prepare(input_shape, True)