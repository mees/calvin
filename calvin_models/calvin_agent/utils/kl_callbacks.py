from pytorch_lightning import Callback, LightningModule, Trainer
import torch


def sigmoid(scale: float, shift: float, x: int) -> float:
    return torch.sigmoid(torch.Tensor([(x - shift) / (scale / 12)])).item()


class KLSchedule(Callback):
    """
    Base class for KL Annealing
    """

    def __init__(self, start_epoch: int, end_epoch: int, max_kl_beta: float):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.max_kl_beta = max_kl_beta

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = pl_module.current_epoch
        kl_beta = self._anneal_fn(epoch)
        pl_module.set_kl_beta(kl_beta)  # type: ignore

    def _anneal_fn(self, epoch):
        raise NotImplementedError


class KLConstantSchedule(KLSchedule):
    def __init__(self):
        pass

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass

    def _anneal_fn(self, epoch: int) -> None:
        pass


class KLSigmoidSchedule(KLSchedule):
    def _anneal_fn(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            kl_beta = 0.0
        elif epoch > self.end_epoch:
            kl_beta = self.max_kl_beta
        else:
            scale = self.end_epoch - self.start_epoch
            shift = (self.end_epoch + self.start_epoch) / 2
            kl_beta = sigmoid(scale=scale, shift=shift, x=epoch) * self.max_kl_beta
        return kl_beta


class KLLinearSchedule(KLSchedule):
    def _anneal_fn(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            kl_beta = 0.0
        elif epoch > self.end_epoch:
            kl_beta = self.max_kl_beta
        else:
            kl_beta = self.max_kl_beta * (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return kl_beta


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")
    import numpy as np

    kl = KLLinearSchedule(10, 50, 0.1)
    x = np.arange(200)
    y = [kl._anneal_fn(i) for i in x]
    plt.plot(x, y)

    kl2 = KLSigmoidSchedule(10, 50, 0.1)
    x = np.arange(200)
    y = [kl2._anneal_fn(i) for i in x]
    plt.plot(x, y)

    plt.show()
