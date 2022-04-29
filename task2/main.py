import numpy as np
import torch
import wandb
import time
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim

import sys
sys.path.append(".")
from utils import EarlyStopping, LogWriter, to_device
from dataset import MovieDataset
from model import Text_RNN, Text_CNN


class ModelHandler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.logwriter = LogWriter(config["out_dir"])

        self._build_dataloader()
        self._build_model()
        self._build_optimizer()

    def _build_dataloader(self):
        self.data = MovieDataset(self.config["data_file"], self.config["data_size"], self.config["test_rate"])
        self.train_dataloader = DataLoader(
            self.data.train,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=self.data.collate_fn,
            drop_last=True # drop the last incomplete batch
        )
        self.test_dataloader = DataLoader(
            self.data.test,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=self.data.collate_fn,
            drop_last=True
        )
        log = f"Train size: {len(self.data.train)}, Test size: {len(self.data.test)}"
        print(log)
        self.logwriter.write(log)

    def _build_model(self):
        if self.config["model_type"] == "rnn":
            self.model = Text_RNN(
                word_emb_type=self.config["word_emb_type"],
                pretrained_emb_file=self.config["pretrained_emb_file"],
                vocab=self.data.vocab,
                word_emb_size=self.config["word_emb_size"],
                dropout=self.config["dropout"],
                input_size=self.config["input_size"],
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                category_size=len(self.data.category)
            ).to(self.device)
        elif self.config["model_type"] == "cnn":
            self.model = Text_CNN(
                word_emb_type=self.config["word_emb_type"],
                pretrained_emb_file=self.config["pretrained_emb_file"],
                vocab=self.data.vocab,
                word_emb_size=self.config["word_emb_size"],
                dropout=self.config["dropout"],
                out_channels=self.data.max_sent_len,
                category_size=len(self.data.category)
            ).to(self.device)
        else:
            raise Exception("Unknow model type!")

    def _build_optimizer(self):
        self.loss_fc = F.cross_entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.save_model_path = os.path.join(
            self.config["out_dir"], str(self.config["epochs"]) + "epochs_" + 
            self.config["word_emb_type"] + "_" +self.config["model_type"] + "_bestmodel.pt"
        )
        self.stopper = EarlyStopping(self.save_model_path, self.config["patience"])

    def train(self):
        train_loss = list()
        train_acc = list()
        for epoch in range(self.config["epochs"]):
            batch_loss = list()
            batch_acc = list()
            self.model.train()
            t0 = time.time() # training time per epoch
            for _, (x, y) in enumerate(self.train_dataloader): # every 32 samples
                x, y = to_device([x, y], self.device)
                prob = self.model(x)
                loss = self.loss_fc(prob, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                pred = prob.argmax(-1) # get the category of max value in categories as result category
                batch_acc.append(torch.mean((pred == y).float(), dtype=torch.float).item())
            train_loss.append(np.mean(batch_loss))
            train_acc.append(np.mean(batch_acc))
            log = "Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.3f} | Acc: {:.3f}".format(
                epoch + 1, self.config["epochs"], time.time() - t0, train_loss[-1], train_acc[-1]
            )
            print(log)
            self.logwriter.write(log)

            if self.stopper.step(train_acc[-1], self.model):
                pass # break

        return train_loss, train_acc

    def test(self):
        self.model = torch.load(self.save_model_path) # load the best saved model
        batch_loss = list()
        batch_acc = list()
        batch_long_loss = list()
        batch_long_acc = list()
        self.model.eval()
        t0 = time.time() # total test time
        with torch.no_grad():
            for _, (x, y) in enumerate(self.test_dataloader):
                x, y = to_device([x, y], self.device)
                prob = self.model(x)
                loss = self.loss_fc(prob, y)

                batch_loss.append(loss.item())
                pred = prob.argmax(-1)
                batch_acc.append(torch.mean((pred == y).float(), dtype=torch.float).item())
                if x.size(1) > 25: # long sentence
                    batch_long_loss.append(batch_loss[-1])
                    batch_long_acc.append(batch_acc[-1])
        test_loss = np.mean(batch_loss)
        test_acc = np.mean(batch_acc)
        test_long_loss = np.mean(batch_long_loss)
        test_long_acc = np.mean(batch_long_acc)
        log = "Test time: {:.2f} | Loss: {:.3f} | Long sent loss: {:.3f} | Acc: {:.3f} | Long sent acc: {:.3f}".format(
            time.time() - t0, test_loss, test_long_loss, test_acc, test_long_acc
        )
        print(log)
        self.logwriter.write(log)

def draw_single(loss, acc, epochs, word_emb_type, model_type, out_dir):
    x = list(range(1, len(loss) + 1))
    plt.subplot()
    plt.title(str(epochs) + "epochs_" + word_emb_type + "_" + model_type + "_loss&acc")
    plt.xlabel("Actual epochs")
    plt.ylabel("Loss/Acc")
    plt.plot(x, loss, 'r--', label='loss')
    plt.plot(x, acc, 'g--', label='acc')
    plt.legend()
    plt.savefig(os.path.join(
        out_dir, str(epochs) + "epochs_" + word_emb_type + "_" + model_type + ".png"
    ))

def draw_comparison(comparison, epochs, out_dir):
    x = list(range(1, epochs + 1))
    plt.subplot(1, 2, 1)
    plt.title(str(epochs) + "epochs_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(x, comparison["random_rnn_train_loss"], 'r--', label='random_rnn')
    plt.plot(x, comparison["random_cnn_train_loss"], 'y--', label='random_cnn')
    plt.plot(x, comparison["glove_rnn_train_loss"], 'b--', label='glove_rnn')
    plt.plot(x, comparison["glove_cnn_train_loss"], 'g--', label='glove_cnn')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(str(epochs) + "epochs_acc")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.plot(x, comparison["random_rnn_train_acc"], 'r--', label='random_rnn')
    plt.plot(x, comparison["random_cnn_train_acc"], 'y--', label='random_cnn')
    plt.plot(x, comparison["glove_rnn_train_acc"], 'b--', label='glove_rnn')
    plt.plot(x, comparison["glove_cnn_train_acc"], 'g--', label='glove_cnn')
    plt.legend()
    plt.savefig(os.path.join(
        out_dir, str(epochs) + "epochs_comparison.png"
    ))

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
    #wandb.init(project=config["proj_name"])

    runner = ModelHandler(config, device)
    t0 = time.time()
    train_loss, train_acc = runner.train()
    runner.test()
    draw_single(train_loss, train_acc, config["epochs"], config["word_emb_type"], config["model_type"], config["out_dir"])
    log = "Total run time: {:.2f}s\n".format(time.time() - t0)
    print(log)
    runner.logwriter.write(log)
    runner.logwriter.close()
    """
    comparison = {
        "random_rnn_train_loss": ,
        "random_cnn_train_loss": ,
        "glove_rnn_train_loss: ,
        "glove_cnn_train_loss": ,

        "random_rnn_train_acc": ,
        "random_cnn_train_acc": ,
        "glove_rnn_train_acc": ,
        "glove_cnn_train_acc": 
    }
    draw_comparison(comparison, config["epochs"], config["out_dir"])
    """

if __name__ == "__main__":
    config = {
        "seed": 2022,
        "gpu": True,
        "proj_name": "nlp-beginner-task2",
        "out_dir": "task2/out",

        "data_file": "task1/data/train.tsv",
        "data_size": None, # None means using all of them
        "test_rate": 0.2,
        "batch_size": 128,
        "num_workers": 0,

        "word_emb_type": "random",
        "pretrained_emb_file": "task2/pretrained_emb/glove.6B.50d.txt",
        "word_emb_size": 50,
        "model_type": "rnn",
        "num_layers": 1, # rnn
        "input_size": 50, # equal to word embedding size
        "hidden_size": 50, # rnn
        "categories": 5,
        "dropout": 0.2,

        "lr": 0.001,
        "epochs": 50,
        "patience": 10,
    }
    main(config)