"""Epoch schedules; measurements always describe the rate actually used."""
import math


class LearningRateSchedule:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config
        self.kind = config.get("scheduler", "fixed")
        self.initial = float(config.get("learning_rate", .0005))
        self.minimum = float(config.get("min_learning_rate", self.initial * .01))
        self.warmup = int(config.get("warmup_epochs", 0))
        self.epochs = int(config["epochs"])
        self.next_rate = self.initial; self.best = None; self.bad = 0; self.epoch = 0

    def start_epoch(self, epoch):
        self.epoch = epoch
        rate = self.initial
        if self.kind == "plateau":
            rate = self.next_rate
        elif self.kind in {"cosine", "linear"}:
            if self.warmup and epoch <= self.warmup:
                rate = self.initial * epoch / self.warmup
            else:
                t = min(1., max(0., (epoch-self.warmup-1) / max(1,self.epochs-self.warmup-1)))
                factor = (1+math.cos(math.pi*t))/2 if self.kind == "cosine" else 1-t
                rate = self.minimum + (self.initial-self.minimum)*factor
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        return measured_rates(self.optimizer)

    def finish_epoch(self, score):
        if self.kind != "plateau": return
        if not isinstance(score, (int,float)) or not math.isfinite(score):
            raise ValueError("動態學習率需要有效的 Validation 指標")
        if self.best is None or score > self.best + 1e-6:
            self.best = float(score); self.bad = 0
        else:
            self.bad += 1
            if self.bad > int(self.config.get("lr_patience", 3)):
                self.next_rate = max(self.minimum, self.next_rate * float(self.config.get("lr_factor", .5)))
                self.bad = 0

    def state_dict(self):
        return {"config":self.config, "epoch":self.epoch, "best":self.best,
                "bad":self.bad, "next_rate":self.next_rate}


def measured_rates(optimizer):
    groups = getattr(optimizer, "param_groups", [])
    result = {}
    for i, group in enumerate(groups):
        value = group.get("lr")
        if isinstance(value, (int,float)) and math.isfinite(value):
            if i == 0: result["train/learning_rate"] = float(value)
            if i > 0: result[f"lr/group_{i}"] = float(value)
    return result
