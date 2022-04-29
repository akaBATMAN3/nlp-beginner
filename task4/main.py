import numpy as np
import torch
import wandb
import os
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append(".")
from utils import LogWriter, EarlyStopping, to_device, f1_calc
from dataset import CONLLDataset
from model import LSTM_CRF


class ModelHandler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.logwriter = LogWriter(config["out_dir"])

        self._build_dataloader()
        self._build_model()
        self._build_optimizer()

    def _build_dataloader(self):
        self.data = CONLLDataset(self.config["data_file"], self.config["data_size"], self.config["tag_dict"])
        self.train_dataloader = DataLoader(
            self.data.train,
            batch_size=self.config["batch_size"],
            shuffle=False, # keep the sorted order
            num_workers=self.config["num_workers"],
            collate_fn=self.data.collate_fn,
            drop_last=True
        )
        self.val_dataloader = DataLoader(
            self.data.val,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=self.data.collate_fn,
            drop_last=True
        )
        self.test_dataloader = DataLoader(
            self.data.test,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=self.data.collate_fn,
            drop_last=True
        )
        log = f"Train size: {len(self.data.train)}, Val size: {len(self.data.val)}, Test size: {len(self.data.test)}"
        print(log)
        self.logwriter.write(log)

    def _build_model(self):
        self.model = LSTM_CRF(
            word_emb_type=self.config["word_emb_type"],
            pretrained_emb_file=self.config["pretrained_emb_file"],
            vocab=self.data.vocab,
            word_emb_size=self.config["word_emb_size"],
            dropout=self.config["dropout"],
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            tag_dict=self.data.tag_dict
        ).to(self.device)

    def _build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max", # monitor F1-score
            factor=self.config["lr_reduce_factor"],
            patience=self.config["lr_patience"],
            verbose=True
        )
        self.save_model_path = os.path.join(
            self.config["out_dir"], str(self.config["epochs"]) + "epochs_bestmodel.pt"
        )
        self.stopper = EarlyStopping(self.save_model_path, self.config["patience"])

    def train(self):
        train_loss = list()
        train_acc = list()
        train_f1 = list()
        for epoch in range(self.config["epochs"]):
            batch_loss = list()
            batch_acc = list()
            batch_f1 = list()
            self.model.train()
            t0 = time.time()
            for _, (x, y) in enumerate(self.train_dataloader):
                mask = (y != 0).int()
                x, y, mask = to_device([x, y, mask], self.device)
                loss, pred = self.model(x, y, mask)

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.get("grad_clipping"):
                    parameters = [p for p in self.model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(parameters, self.config["grad_clipping"])
                self.optimizer.step()

                batch_loss.append(loss.item())
                batch_acc.append(torch.mean((pred == y).float(), dtype=torch.float).item())
                batch_f1.append(f1_calc(y, pred))

            val_f1 = self.evaluate()
            self.scheduler.step(val_f1)

            train_loss.append(np.mean(batch_loss))
            train_acc.append(np.mean(batch_acc))
            train_f1.append(np.mean(batch_f1))
            log = "Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.3f} | Acc: {:.3f} | F1: {:.3f}".format(
                epoch + 1, self.config["epochs"], time.time() - t0, train_loss[-1], train_acc[-1], train_f1[-1]
            )
            print(log)
            self.logwriter.write(log)

            if self.stopper.step(val_f1, self.model):
                break

        return train_loss, train_acc, train_f1

    def evaluate(self):
        batch_f1 = list()
        self.model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(self.val_dataloader):
                mask = (y != 0).int()
                x, y, mask = to_device([x, y, mask], self.device)
                _, pred = self.model(x, y, mask)

                batch_f1.append(f1_calc(y, pred))
        val_f1 = np.mean(batch_f1)

        return val_f1

    def test(self):
        self.model = torch.load(self.save_model_path)
        batch_loss = list()
        batch_acc = list()
        batch_f1 = list()
        self.model.eval()
        t0 = time.time()
        with torch.no_grad():
            for _, (x, y) in enumerate(self.test_dataloader):
                mask = (y != 0).int()
                x, y, mask = to_device([x, y, mask], self.device)
                loss, pred = self.model(x, y, mask)

                batch_loss.append(loss.item())
                batch_acc.append(torch.mean((pred == y).float(), dtype=torch.float).item())
                batch_f1.append(f1_calc(y, pred))
        test_loss = np.mean(batch_loss)
        test_acc = np.mean(batch_acc)
        test_f1 = np.mean(batch_f1)
        log = "Test time: {:.2f}s | Loss: {:.3f} | Acc: {:.3f} | F1: {:.3f}".format(
            time.time() - t0, test_loss, test_acc, test_f1
        )
        print(log)
        self.logwriter.write(log)

def draw(loss, acc, f1, epochs, out_dir):
    x = list(range(1, len(loss) + 1))
    plt.subplot(1, 2, 1)
    plt.title(str(len(loss)) + "epochs_loss")
    plt.xlabel("Actual epochs")
    plt.ylabel("Loss")
    plt.plot(x, loss, 'r--', label='loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(str(len(loss)) + "epochs_acc&f1")
    plt.xlabel("Actual epochs")
    plt.ylabel("Acc/F1")
    plt.plot(x, acc, 'g--', label='acc')
    plt.plot(x, f1, 'b--', label='f1')
    plt.legend()
    plt.savefig(os.path.join(out_dir, str(epochs) + "epochs.png"))

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
    train_loss, train_acc, train_f1 = runner.train()
    draw(train_loss, train_acc, train_f1, config["epochs"], config["out_dir"])
    runner.test()
    log = "Total run time: {:.2f}s\n".format(time.time() - t0)
    print(log)
    runner.logwriter.write(log)
    runner.logwriter.close()

if __name__ == "__main__":
    config = {
        "seed": 2022,
        "gpu": True,
        "proj_name": "nlp-beginner-task4",
        "out_dir": "task4/out",

        "data_file": {
            "train": "task4/data/CoNLL-2003/train.txt",
            "val": "task4/data/CoNLL-2003/dev.txt",
            "test": "task4/data/CoNLL-2003/test.txt"
        },
        "data_size": {"train": None, "val": None, "test": None}, # colab
        "tag_dict": {"<pad>": 0, "<start>": 1, "<end>": 2},
        "batch_size": 128, # colab
        "num_workers": 0, # colab

        "word_emb_type": "glove",
        "pretrained_emb_file": "task2/pretrained_emb/glove.6B.50d.txt",
        "word_emb_size": 50,
        "dropout": 0.5,
        "input_size": 50,
        "hidden_size": 50,
        "num_layers": 1,

        "lr": 0.01,
        "lr_reduce_factor": 0.5,
        "lr_patience": 2,
        "epochs": 100, # colab
        "patience": 10,
        "grad_clipping": 5
    }
    main(config)