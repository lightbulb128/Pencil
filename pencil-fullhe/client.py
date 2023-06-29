import client_modules
import communication_client
from crypto_switch import cryptography as crypto
import numpy as np
import model
import time

if __name__ == "__main__":

    comm = communication_client.ClientCommunication()
    comm.connect()
    cryp = crypto.EncryptionUtils()
    comm.send(cryp.generate_keys())

    input_shape = comm.recv()

    description = comm.recv()
    model = client_modules.client_model_from_description(description, cryp, comm)
    
    model.prepare(input_shape)
    print("Prepared")

    steps = comm.recv()
    forward_only = comm.recv()

    for step in range(steps):
        inputs_b = comm.recv()
        comm.clear_accumulation()
        comm.counter = 1

        transmission = 0
        transmission_tick = 0
        
        timed = time.time()
        outputs_b = inputs_b
        
        for layer in model.items:
            transmission_tick = comm.get_transmission()
            outputs_b = layer.forward(outputs_b)
            transmission += comm.get_transmission() - transmission_tick
            comm.send(outputs_b)

        print("forward time =", time.time() - timed)
        print("forward trans = ", transmission)

        if forward_only:
            break

        partial_output_b = comm.recv()
        for layer in reversed(model.items):
            transmission_tick = comm.get_transmission()
            partial_output_b = layer.backward(partial_output_b)
            transmission += comm.get_transmission() - transmission_tick
            comm.send(partial_output_b)

        print("Transmission:", transmission)
    

    comm.close_connection()