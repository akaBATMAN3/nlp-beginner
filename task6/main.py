import numpy as np
import torch
import wandb
import time
import matplotlib.pyplot as plt
import os

import sys
sys.path.append(".")
from utils import LogWriter, cos_sim
from dataset import STSDateset
from models import InferSent


class ModelHandler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.logwriter = LogWriter(config["out_dir"])

        self._build_dataset()
        if config["method"] == "infersent":
            self._build_model()

    def _build_dataset(self):
        self.data = STSDateset(
            data_file=self.config["data_file"]["input"],
            data_size=self.config["data_size"],
            stopwords=self.config["stopwords"],
            method=self.config["method"],
            gram_num=self.config["gram_num"],
            ngram_threshold=self.config["ngram_threshold"],
            word_emb_type=self.config["word_emb_type"],
            pretrained_emb_file=self.config["pretrained_emb_file"],
            word_emb_size=self.config["word_emb_size"]
        )
        self.golden_score = list()
        with open(self.config["data_file"]["golden"]) as f:
            for line in f:
                self.golden_score.append(float(line.strip("\n")))
        assert len(self.data.x) == len(self.data.y) == len(self.golden_score)
        log = f"Data size: {len(self.data.x)}"
        print(log)
        self.logwriter.write(log)

    def _build_model(self):
        self.model = InferSent(self.config["params_model"]) # init
        self.model.load_state_dict(torch.load(self.config["model_path"]))
        self.model.to(self.device)
        if  self.config["params_model"]["version"] == 1 and self.config["word_emb_type"] is "glove":
            self.model.set_w2v_path(self.config["pretrained_emb_file"]["glove"])
        else:
            raise Exception("Check your version and word emb type, v1 matches glove and v2 matches fastText!")
        self.model.build_vocab_k_words(K=10000000)

    def calculate(self):
        """
        Compute similarity score between sentence pairs, then
        compute Pearson Correlation Coefficient between computed scores and golden scores.
        """
        score = list()
        if self.config["method"] is not "wmd":
            if self.config["method"] is "infersent":
                self.data.x = self.model.encode(self.data.x, tokenize=False, verbose=True)
                self.data.y = self.model.encode(self.data.y, tokenize=False, verbose=True)
            for i in range(len(self.data.x)):
                score.append(cos_sim(self.data.x[i], self.data.y[i])) # multiply by 5 because given golden_score scales from 0 to 5(actually it doesn't matter)
        else:
            for i in range(len(self.data.x)):
                score.append(-self.data.word_emb.wmdistance(self.data.x[i], self.data.y[i]))

        return np.corrcoef(score, self.golden_score)[0, 1]

def visualize(comparison, out_dir):
    x = [i for i in range(len(comparison))]
    fig, ax = plt.subplots()
    ax.set_title("Comparison")
    ax.set_xticks(x)
    ax.set_ylabel("Pearson Correlation Coefficient")
    i = 0
    for label, value in comparison.items():
        rects = ax.bar(i, value, label=label)
        ax.bar_label(rects)
        i += 1
    ax.legend(loc="lower left", prop={"size": 8})
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison.png"))

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

    t0 = time.time()
    runner = ModelHandler(config, device)
    r = runner.calculate()
    log = "Pearson Correlation Coefficient: {:.3f}, Total run time: {:.2f}s\n".format(r, time.time() - t0)
    print(log)
    runner.logwriter.write(log)
    runner.logwriter.close()
    """
    comparison = {
        "bow": ,
        "bow-stop": ,
        "ngram": ,
        "ngram-stop": ,
        "tfidf": ,
        "tfidf-stop": ,

        "avg-w2v-stop": ,
        "avg-glove-stop": ,
        "avg-tfidf-w2v-stop": ,
        "avg-tfidf-glove-stop": ,
        "wmd-w2v-stop": ,
        "wmd-glove-stop": ,
        "infersent-glove-stop": 
    }
    visualize(comparison, config["out_dir"])
    """

if __name__ == "__main__":
    config = {
        "seed": 2022,
        "gpu": True,
        "proj_name": "nlp-beginner-task6",
        "out_dir": "task6/out",

        "data_file": {
            "input": "task6/data/input.txt",
            "golden": "task6/data/golden.txt"
        },
        "data_size": None,
        "stopwords": True,
        "method": "avg-tfidf",
        "gram_num": 3, # if method is not ngram, ignore this parameter
        "ngram_threshold": 2, # if method is not ngram, ignore this parameter
        "word_emb_type": "glove",
        "pretrained_emb_file": {
            "word2vec": "task6/pretrained_emb/GoogleNews-vectors-negative300.bin",
            "glove": "task6/pretrained_emb/glove.840B.300d.w2v.txt"
        },
        "word_emb_size": 300,

        "params_model": {
            "bsize": 64,
            "word_emb_dim": 300,
            "enc_lstm_dim": 2048,
            "pool_type": "max",
            "dpout_model": 0.0,
            "version": 1
        },
        "model_path": "task6/encoder/infersent1.pkl"
    }
    main(config)