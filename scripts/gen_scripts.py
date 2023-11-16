python = "python3"

def end_to_end(
    name, preset, 
    noise=0, gradient_bound=8, 
    optimizer="SGD", momentum=0.8, learning_rate=1e-2,
    split="None", epochs=10,
):
    s_args = f"-p {preset} -n {noise} -C {gradient_bound} -o {optimizer} -om {momentum} -lr {learning_rate}"
    c_args = f"-s {split} -e {epochs}"
    s_cmd = f"{python} train_priv_server.py {s_args} > /dev/null &"
    c_cmd = f"{python} train_priv_client.py {c_args} > logs/{name}.log &"
    return [s_cmd, "sleep 5", c_cmd]

def costs(name, preset):
    s_args = f"-p {preset}"
    s_cmd = f"{python} server_costs.py {s_args} > logs/{name}.log &"
    c_cmd = f"{python} client_costs.py > /dev/null &"
    return [s_cmd, "sleep 5", c_cmd]

def change_directory(directory, inner_commands):
    ret = [f'cd {directory}']
    ret.extend(inner_commands)
    ret.append('cd ..')
    return ret

def write_list_to_file(l, filename):
    with open(filename, "w") as f:
        for line in l:
            f.write(line + "\n")

def pencil(name, inner_commands):
    l_fullhe = change_directory("pencil-fullhe", inner_commands)
    l_prep = change_directory("pencil-prep", inner_commands)
    write_list_to_file(l_fullhe, f"fullhe_{name}.sh")
    write_list_to_file(l_prep, f"prep_{name}.sh")

if __name__ == "__main__":
    pencil("trivial", end_to_end("trivial", "mnist_aby3", epochs=1))
    pencil("train_nn1", end_to_end("train_nn1", "mnist_aby3"))
    pencil("train_nn2", end_to_end("train_nn2", "mnist_chameleon"))
    pencil("train_nn3", end_to_end("train_nn3", "agnews_cnn"))
    pencil("train_nn4", end_to_end("train_nn4", "cifar10_sphinx"))
    pencil("train_nn5", end_to_end("train_nn5", "agnews_gpt2"))
    pencil("train_nn6", end_to_end("train_nn6", "resnet50_classifier"))
    pencil("costs_nn1", costs("costs_nn1", "mnist_aby3"))
    pencil("costs_nn2", costs("costs_nn2", "mnist_chameleon"))
    pencil("costs_nn3", costs("costs_nn3", "agnews_cnn"))
    pencil("costs_nn4", costs("costs_nn4", "cifar10_sphinx"))
    pencil("costs_nn5", costs("costs_nn5", "agnews_gpt2"))
    pencil("costs_nn6", costs("costs_nn6", "resnet50_classifier"))
    for dp in [0.005, 0.01, 0.02, 0.05]:
        pencil(f"dp{dp:.3f}_nn4", end_to_end(f"dp{dp:.3f}_nn4", "cifar10_sphinx", noise=dp))
        pencil(f"dp{dp:.3f}_nn6", end_to_end(f"dp{dp:.3f}_nn6", "resnet50_classifier", noise=dp))
    pencil("hetero_nn4", end_to_end("hetero_nn4", "cifar10_sphinx", split="5:5"))
    pencil("hetero_nn6", end_to_end("hetero_nn6", "resnet50_classifier", split="5:5"))

