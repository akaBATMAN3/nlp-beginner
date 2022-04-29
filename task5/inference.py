import torch


def main(config):
    torch.manual_seed(config["seed"])
    if config["gpu"] and torch.cuda.is_available():
        print("[ Using CUDA ]")
        torch.cuda.manual_seed(config["seed"])
        device = torch.device("cuda")
    else:
        print("[ Using CPU ]")
        device = torch.device("cpu")

    model = torch.load(config["model_file"]).to(device)
    poems = model.generate(device, config["heads"], config["poems_size"], config["max_poem_len"])
    output = ""
    for poem in poems:
        output += ''.join(poem) + '\n'
    print(output)

if __name__ == "__main__":
    config = {
        "seed": 2022,
        "gpu": True,

        "model_file": "task5/out/20epochs_gru_bestmodel.pt",
        "heads": "春夏秋冬",
        "poems_size": 4,
        "max_poem_len": 30
    }
    main(config)