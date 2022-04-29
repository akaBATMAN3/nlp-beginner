import numpy as np
import torch
import wandb
import time
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append(".")
from utils import to_device, LogWriter, EarlyStopping
from dataset import SNLIDataset
from model import ESIM


class ModelHandler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.logwriter = LogWriter(config["out_dir"])

        self._build_dataloader()
        self._build_model()
        self._build_optimizer()

    def _build_dataloader(self):
        self.data = SNLIDataset(self.config["data_file"], self.config["data_size"], self.config["type_dict"])
        self.train_dataloader = DataLoader(
            self.data.train,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=self.data.collate_fn
        )
        self.val_dataloader = DataLoader(
            self.data.val,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=self.data.collate_fn
        )
        self.test_dataloader = DataLoader(
            self.data.test,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=self.data.collate_fn
        )
        log = f"Train size: {len(self.data.train)}, Val size: {len(self.data.val)}, Test size: {len(self.data.test)}"
        print(log)
        self.logwriter.write(log)

    def _build_model(self):
        self.model = ESIM(
            word_emb_type=self.config["word_emb_type"],
            pretrained_emb_file=self.config["pretrained_emb_file"],
            vocab=self.data.vocab,
            word_emb_size=self.config["word_emb_size"],
            dropout=self.config["dropout"],
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            type_num=len(self.config["type_dict"])
        ).to(self.device)

    def _build_optimizer(self):
        self.loss_fc = F.cross_entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
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
        for epoch in range(self.config["epochs"]):
            batch_loss = list()
            batch_acc = list()
            self.model.train()
            t0 = time.time()
            for _, (x1, x2, y) in enumerate(self.train_dataloader):
                x1, x2, y = to_device([x1, x2, y], self.device) # x-[batch size, batched sent len], y-[batch size]
                prob = self.model(x1, x2)
                loss = self.loss_fc(prob, y)

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.get("grad_clipping"):
                    parameters = [p for p in self.model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(parameters, self.config["grad_clipping"])
                self.optimizer.step()

                batch_loss.append(loss.item())
                pred = prob.argmax(-1)
                batch_acc.append(torch.mean((pred == y).float(), dtype=torch.float).item())

            val_acc = self.evaluate()
            self.scheduler.step(val_acc)

            train_loss.append(np.mean(batch_loss))
            train_acc.append(np.mean(batch_acc))
            log = "Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.3f} | Acc: {:.3f}".format(
                epoch + 1, self.config["epochs"], time.time() - t0, train_loss[-1], train_acc[-1]
            )
            print(log)
            self.logwriter.write(log)

            if self.stopper.step(val_acc, self.model):
                break

        return train_loss, train_acc

    def evaluate(self):
        batch_acc = list()
        self.model.eval()
        with torch.no_grad():
            for _, (x1, x2, y) in enumerate(self.val_dataloader):
                x1, x2, y = to_device([x1, x2, y], self.device)
                prob = self.model(x1, x2)

                pred = prob.argmax(-1)
                batch_acc.append(torch.mean((pred == y).float(), dtype=torch.float).item())
        val_acc = np.mean(batch_acc)

        return val_acc

    def test(self):
        self.model = torch.load(self.save_model_path) # load the best model evaluated on val
        batch_loss = list()
        batch_acc = list()
        self.model.eval()
        t0 = time.time()
        with torch.no_grad():
            for _, (x1, x2, y) in enumerate(self.test_dataloader):
                x1, x2, y = to_device([x1, x2, y], self.device)
                prob = self.model(x1, x2)
                loss = self.loss_fc(prob, y)

                batch_loss.append(loss.item())
                pred = prob.argmax(-1)
                batch_acc.append(torch.mean((pred == y).float(), dtype=torch.float).item())
        test_loss = np.mean(batch_loss)
        test_acc = np.mean(batch_acc)
        log = "Test time: {:.2f}s | Loss: {:.3f} | Acc: {:.3f}".format(
            time.time() - t0, test_loss, test_acc
        )
        print(log)
        self.logwriter.write(log)

def draw(loss, acc, epochs, out_dir):
    x = list(range(1, len(loss) + 1))
    plt.subplot()
    plt.title(str(epochs) + "epochs_loss&acc")
    plt.xlabel("Actual epochs")
    plt.ylabel("Loss/Acc")
    plt.plot(x, loss, 'r--', label='loss')
    plt.plot(x, acc, 'g--', label='acc')
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
    #wandb.init(project=config["proj_name"]) # only used to monitor hardware status of colab for now

    runner = ModelHandler(config, device)
    t0 = time.time() # total run time
    train_loss, train_acc = runner.train()
    draw(train_loss, train_acc, config["epochs"], config["out_dir"])
    runner.test()
    log = "Total run time: {:.2f}s\n".format(time.time() - t0)
    print(log)
    runner.logwriter.write(log)
    runner.logwriter.close()

if __name__ == "__main__":
    template = {
        "seed": 2022,
        "gpu": True,
        "proj_name": "nlp-beginner-task3",
        "out_dir": "task3/out",

        "data_file": {
            "train": "task3/data/snli_1.0/snli_1.0_train.jsonl",
            "val": "task3/data/snli_1.0/snli_1.0_dev.jsonl",
            "test": "task3/data/snli_1.0/snli_1.0_test.jsonl"
        },
        "data_size": {"train": None, "val": None, "test": None}, # None means take all
        "type_dict": {"contradiction": 0, "entailment": 1, "neutral": 2, "-": 3},
        "batch_size": 256,

        "word_emb_type": "glove",
        "pretrained_emb_file": "task2/pretrained_emb/glove.6B.50d.txt",
        "word_emb_size": 50,
        "input_size": 50, # equal to word_emb_size
        "hidden_size": 50, # equal to word_emb_size
        "num_layers": 1,
        "dropout": 0.2,

        "lr": 0.001,
        "epochs": 100,
        "lr_reduce_factor": 0.5,
        "lr_patience": 3,
        "patience": 10, # early stop
        "grad_clipping": 10
    }
    main(template)