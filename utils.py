import torch
import os
import datetime as dt
from sklearn.metrics import f1_score
import numpy as np


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], torch.Tensor):
                data[i] = data[i].to(device)

    return data

class LogWriter:
    def __init__(self, dirname):
        os.makedirs(dirname, exist_ok=True) # create output dir
        self.fout = open(os.path.join(dirname, "log.txt"), "a")
        self.fout.writelines("\n" + dt.datetime.now().strftime('%F %T') + "\n")

    def write(self, text):
        self.fout.writelines(text + "\n")
        self.fout.flush()

    def close(self):
        self.fout.close()

class EarlyStopping:
    def __init__(self, save_model_path, patience=10):
        self.save_model_path = save_model_path
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            save_model_ckpt(model, self.save_model_path)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save_model_ckpt(model, self.save_model_path)
            self.counter = 0

        return self.early_stop

def save_model_ckpt(model, save_model_path):
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model, save_model_path)
    print(f"Saved model to {save_model_path}")

def f1_calc(y, pred):
    """Compute F1-score in one batch."""
    y = y.cpu()
    pred = pred.cpu()
    f1 = list()
    for i in range(len(y)):
        f1.append(f1_score(y[i], pred[i], average='macro'))
    
    return np.mean(f1)

def cos_sim(v1, v2):
    """
    Calculate cosine similarity between vectors,
    verified with sklearn.metrics.pairwise.cosine_similarity
    """
    num = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return (num / denom) if denom != 0 else 0