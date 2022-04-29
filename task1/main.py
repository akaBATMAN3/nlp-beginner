import numpy as np
import torch
import wandb
import time
import os
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from utils import LogWriter
from dataset import MovieDataset
from model import Softmax


class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.logwriter = LogWriter(config["out_dir"])

        self._build_data()
        self._build_model()
        self._build_save_path()

    def _build_data(self):
        self.data = MovieDataset(
            data_file=self.config["data_file"],
            data_size=self.config["data_size"],
            text_rpst=self.config["text_rpst"],
            gram_num=self.config["gram_num"],
            test_rate=self.config["test_rate"]
        )
        log = f"Train size: {len(self.data.train_y)}, Test size: {len(self.data.test_y)}"
        print(log)
        self.logwriter.write(log)

    def _build_model(self):
        self.model = Softmax(
            input_size=len(self.data.train_y),
            category_size=len(self.data.category),
            vocab_size=len(self.data.vocab)
        )

    def _build_save_path(self):
        self.save_model_path = os.path.join(
            self.config["out_dir"],
            str(self.config["epochs"]) + "epochs_" + self.config["text_rpst"] + "_" + self.config["gradient"] + "_bestmodel.pt"
        )

    def train(self, lr):
        self.model.regression(
            x=self.data.train_x,
            y=self.data.train_y,
            gradient=self.config["gradient"],
            epochs=self.config["epochs"],
            lr=lr,
            mini_size=self.config["mini_size"]
        )
        prob = self.model.predict(self.data.train_x)
        pred = prob.argmax(axis=1)
        acc = sum([pred[i] == self.data.train_y[i] for i in range(len(self.data.train_y))]) / len(self.data.train_y)

        return acc

    def test(self):
        prob = self.model.predict(self.data.test_x)
        pred = prob.argmax(axis=1)
        acc = sum([pred[i] == self.data.test_y[i] for i in range(len(self.data.test_y))]) / len(self.data.test_y)
        return acc

def draw_single(epochs, lr, train_acc, test_acc, out_dir, text_rpst, gradient):
    plt.subplot()
    plt.title(str(epochs) + "epochs_acc")
    plt.xlabel("LR")
    plt.ylabel("Acc")
    plt.semilogx(lr, train_acc, 'r--', label='train')
    plt.semilogx(lr, test_acc, 'g--', label='test')
    plt.legend()
    plt.savefig(os.path.join(
        out_dir, str(epochs) + "epochs_" + text_rpst + "_" + gradient + ".png"
    ))

def draw_comparison(epochs, lr, comparison, out_dir):
    plt.subplot(1, 2, 1)
    plt.title(str(epochs) + "epochs_train_acc")
    plt.xlabel("LR")
    plt.ylabel("Acc")
    plt.semilogx(lr, comparison["bow_SGD_train"], 'r--', label="bow_SGD")
    plt.semilogx(lr, comparison["bow_BGD_train"], 'y--', label="bow_BGD")
    plt.semilogx(lr, comparison["bow_mini-batch_train"], 'b--', label="bow_mini-batch")
    plt.semilogx(lr, comparison["ngram_SGD_train"], 'g--', label="ngram_SGD")
    plt.semilogx(lr, comparison["ngram_BGD_train"], 'c', label="ngram_BGD")
    plt.semilogx(lr, comparison["ngram_mini-batch_train"], 'm', label="ngram_mini-batch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(str(epochs) + "epochs_test_acc")
    plt.xlabel("LR")
    plt.ylabel("Acc")
    plt.semilogx(lr, comparison["bow_SGD_test"], 'r--', label="bow_SGD")
    plt.semilogx(lr, comparison["bow_BGD_test"], 'y--', label="bow_BGD")
    plt.semilogx(lr, comparison["bow_mini-batch_test"], 'b--', label="bow_mini-batch")
    plt.semilogx(lr, comparison["ngram_SGD_test"], 'g--', label="ngram_SGD")
    plt.semilogx(lr, comparison["ngram_BGD_test"], 'c', label="ngram_BGD")
    plt.semilogx(lr, comparison["ngram_mini-batch_test"], 'm', label="ngram_mini-batch")
    plt.legend()
    plt.savefig(os.path.join(
        out_dir, str(epochs) + "epochs_comparison.png"
    ))

def main(config):
    np.random.seed(config["seed"]) # reappearance the experiment
    torch.manual_seed(config["seed"])
    #wandb.init(project=config["proj_name"]) # monitor hardware status

    runner = ModelHandler(config)
    train_acc = list()
    test_acc = list()
    best_acc = 0
    t0 = time.time()
    for lr in config["lr"]:
        acc = runner.train(lr)
        train_acc.append(acc)
        if acc > best_acc:
            torch.save(runner.model, runner.save_model_path)
        acc = runner.test()
        test_acc.append(acc)
    draw_single(config["epochs"], config["lr"], train_acc, test_acc, config["out_dir"], config["text_rpst"], config["gradient"])
    log = "Total run time: {:.2f}s\n".format(time.time() - t0)
    print(log)
    runner.logwriter.write(log)
    runner.logwriter.close()
    """
    comparison = {
        "bow_SGD_train": ,
        "bow_BGD_train": ,
        "bow_mini-batch_train": ,
        "ngram_SGD_train": ,
        "ngram_BGD_train": ,
        "ngram_mini-batch_train": ,

        "bow_SGD_test": ,
        "bow_BGD_test": ,
        "bow_mini-batch_test": ,
        "ngram_SGD_test": ,
        "ngram_BGD_test": ,
        "ngram_mini-batch_test": 
    }
    draw_comparison(config["epochs"], config["lr"], comparison, config["out_dir"])
    """

if __name__ == "__main__":
    config = {
        "seed": 2022,
        "proj_name": "nlp-beginner-task1",
        "out_dir": "task1/out",

        "data_file": "task1/data/train.tsv",
        "data_size": 1000, # none means all
        "text_rpst": "bow", # text representation
        "gram_num": 3, # if text_rpst is not ngram, this parameter won't help
        "test_rate": 0.3,

        "epochs": 10000,
        "lr": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        "gradient": "SGD", # gradient strategy
        "mini_size": 10 # if the gradient is not mini-batch, this parameter won't help
    }
    main(config)