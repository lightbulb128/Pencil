import struct
import os
import subprocess


MPC_SERVER_INPUT_FILENAME = "./tools/Temp/Input-P0-0"
MPC_CLIENT_INPUT_FILENAME = "./tools/Temp/Input-P1-0"
MPC_SERVER_OUTPUT_FILENAME = "./tools/Temp/Output-P0-0"
MPC_CLIENT_OUTPUT_FILENAME = "./tools/Temp/Output-P1-0"

def run_mpc_command(name, party):
    wd = os.getcwd()
    os.chdir("./tools/")
    com = [
        "./semi2k-party.x",
        str(party),
        name,
        "-pn", "12147",
        "-h", "localhost",
        "-N", "2",
        "-IF", "./Temp/Input",
        "-OF", "./Temp/Output"
    ]
    subprocess.run(com, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    os.chdir(wd)

def read_list(filename):
    fd = open(filename, "r")
    lines = fd.readlines()
    fd.close()
    return [float(l) for l in lines]

def write_list(filename, x, write_length_first=False):
    with open(filename, "w") as fd:
        if write_length_first: fd.write(str(len(x)) + "\n")
        for i in x:
            fd.write(str(i) + " ")
        fd.write("\n")

def write_list_zip(filename, x, y, write_length_first=False):
    with open(filename, "w") as fd:
        if write_length_first: fd.write(str(len(x)) + "\n")
        for i, j in zip(x, y):
            fd.write(str(i) + " " + str(j) + " ")
        fd.write("\n")

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data