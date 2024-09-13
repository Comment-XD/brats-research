import torch

class Checkpoint:
    def __init__(self, model, optimizer=None, epoch=0, best_score=1):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.best_score = best_score

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.epoch = checkpoint["epoch"]
        self.best_score = checkpoint["best_score"]
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            for state in self.optimizer.state.values():
                  for k, v in state.items():
                           if torch.is_tensor(v):
                                    state[k] = v.cuda()

    def save(self, path):
        state_dict = self.model.module.state_dict()
        torch.save({"model_state": state_dict,
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                    "best_score": self.best_score}, path)