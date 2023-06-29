import model
from client_modules_switch import client_modules
import communication_client
from crypto_switch import cryptography
import dataloader
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":

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

    # load data
    _, test_data = dataloader.load_dataset("mnist")
    test_dataloader = dataloader.BatchLoader(test_data, 1, False)

    # test
    correct = 0
    total = 0
    for _ in tqdm(range(len(test_dataloader))):
        comm.send("test")
        inputs, labels = test_dataloader.get()
        inputs = inputs.numpy()
        labels = labels.numpy()
        x, x_another_share = client_modules.create_shares(inputs)
        comm.send(x_another_share)
        output = model.forward(x)
        output_another_share = comm.recv()
        output += output_another_share
        pred = np.argmax(output, axis=1)
        print(labels)
        print(pred)
    comm.send("finish")

    # close connection
    comm.close_connection()
