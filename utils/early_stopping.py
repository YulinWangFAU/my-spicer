# utils/early_stopping.py

import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_score, model):
        # 如果是 DataParallel 模型，保存 .module.state_dict()
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(state_dict)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(state_dict)
            self.counter = 0

    def save_checkpoint(self, state_dict):
        torch.save(state_dict, self.path)
        if self.verbose:
            print(f"✅ Model improved. Saving checkpoint to {self.path}")
