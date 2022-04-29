import numpy as np
import torch
import wandb
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append(".")
from utils import LogWriter, EarlyStopping, to_device
from dataset import PoemDataset
from model import SumModel


class ModelHandler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.logwriter = LogWriter(config["out_dir"])

        self._build_dataloader()
        self._build_model()
        self._build_optimizer()

    def _build_dataloader(self):
        self.data = PoemDataset(self.config["data_file"], self.config["data_size"])
        self.train_dataloader = DataLoader(
            self.data.train,
            batch_size = self.config["batch_size"],
            shuffle=True, # shuffle if batch_size=1
            num_workers=self.config["num_workers"],
            collate_fn=self.data.collate_fn,
            drop_last=True
        )
        log = f"Train size: {len(self.data.train)}"
        print(log)
        self.logwriter.write(log)

    def _build_model(self):
        self.model = SumModel(
            word2index_dict=self.data.word2index_dict,
            index2word_dict=self.data.index2word_dict,
            word_emb_type=self.config["word_emb_type"],
            pretrained_emb_file=self.config["pretrained_emb_file"],
            word_emb_size=self.config["word_emb_size"],
            rnn_type=self.config["rnn_type"],
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"]
        ).to(self.device)

    def _build_optimizer(self):
        self.loss_fc = F.cross_entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min", # monitor perplexity
            factor=self.config["lr_reduce_factor"],
            patience=self.config["lr_patience"],
            verbose=True
        )
        self.save_model_path = os.path.join(
            self.config["out_dir"], str(self.config["epochs"]) + "epochs_" + self.config["rnn_type"] + "_bestmodel.pt"
        )
        self.stopper = EarlyStopping(self.save_model_path, self.config["patience"])

    def train(self):
        train_loss = list()
        train_ppl = list()
        for epoch in range(self.config["epochs"]):
            batch_loss = list()
            batch_ppl = list()
            self.model.train()
            t0 = time.time()
            for _, x in enumerate(self.train_dataloader):
                x = to_device(x, self.device)
                x, y = x[:, :-1], x[:, 1:]
                prob = self.model(x).transpose(1, 2)
                loss = self.loss_fc(prob, y)

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.get("grad_clipping"):
                    parameters = [p for p in self.model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(parameters, self.config["grad_clipping"])
                self.optimizer.step()

                batch_loss.append(loss.item())
                batch_ppl.append(np.exp(loss.item() / x.shape[1] + 1))

            self.scheduler.step(np.mean(batch_ppl)) # monitor train's ppl due to the lack of val

            train_loss.append(np.mean(batch_loss))
            train_ppl.append(np.mean(batch_ppl))
            log = "Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.3f} | Ppl: {:.3f}".format(
                epoch + 1, self.config["epochs"], time.time() - t0, train_loss[-1], train_ppl[-1]
            )
            print(log)
            self.logwriter.write(log)

            if self.stopper.step(-np.mean(batch_ppl), self.model):
                break

        return train_loss, train_ppl

def draw(loss, ppl, epochs, rnn_type, out_dir):
    x = list(range(1, len(loss) + 1))
    plt.subplot()
    plt.title(str(epochs) + "epochs_loss&ppl")
    plt.xlabel("Actual epochs")
    plt.ylabel("Loss/Ppl")
    plt.plot(x, loss, 'r--', label='loss')
    plt.plot(x, ppl, 'g--', label='ppl')
    plt.legend()
    plt.savefig(os.path.join(out_dir, str(epochs) + "epochs_" + rnn_type + ".png"))

def main(config):
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["gpu"] and torch.cuda.is_available():
        print("[ Using CUDA ]")
        torch.cuda.manual_seed(config["seed"])
        device = torch.device("cuda")
    else:
        print("[ Using CPU ]")
        device = torch.device("cpu")
    #wandb.init(project=config["proj_name"]) # colab

    runner = ModelHandler(config, device)
    t0 = time.time()
    train_loss, train_ppl = runner.train()
    draw(train_loss, train_ppl, config["epochs"], config["rnn_type"], config["out_dir"])
    log = "Total run time: {:.2f}s\n".format(time.time() - t0)
    print(log)
    runner.logwriter.write(log)
    runner.logwriter.close()

if __name__ == "__main__":
    config = {
        "seed": 2022,
        "gpu": True,
        "proj_name": "nlp-beginner-task5",
        "out_dir": "task5/out",

        "data_file": {
            "train": "task5/data/poetryFromTang.txt",
            "val": None,
            "test": None
        },
        "data_size": {"train": None, "val": None, "test": None},
        "batch_size": 1, # due to the small amount of data
        "num_workers": 0, # colab

        "word_emb_type": "random",
        "pretrained_emb_file": None,
        "word_emb_size": 50,
        "dropout": 0.5,
        "rnn_type": "gru",
        "input_size": 50,
        "hidden_size": 50,
        "num_layers": 1,

        "lr": 0.01,
        "lr_reduce_factor": 0.5,
        "lr_patience": 2,
        "epochs": 20, # colab
        "patience": 10,
        "grad_clipping": 5
    }
    main(config)