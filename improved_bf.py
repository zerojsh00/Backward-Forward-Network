import argparse
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score


class Net(nn.Module):
    def __init__(self, n_feature: int, n_hidden: int, n_output: int, layers: int) -> None:
        super().__init__()
        self.input = nn.Linear(n_feature, n_hidden)
        self.layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(layers)])
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.predict(x)


class BackwardForward(nn.Module):
    def __init__(self, backward_model: Net, forward_model: Net) -> None:
        super().__init__()
        self.backward_model = backward_model
        self.forward_model = forward_model

    def forward(self, y_data: torch.Tensor) -> torch.Tensor:
        x1 = self.backward_model(y_data)
        return self.forward_model(x1)


def generate_data(n_samples: int, elements: float) -> Tuple[np.ndarray, np.ndarray]:
    epsilon = np.random.normal(size=(n_samples))
    x_data = np.float32(np.random.uniform(-10.5, 10.5, n_samples))
    y_data = 7 * np.sin(elements * x_data) + 1.5 * x_data + epsilon * 0.5
    return x_data, y_data


def train(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor,
          x_valid: torch.Tensor, y_valid: torch.Tensor, optimizer: torch.optim.Optimizer,
          epoch: int):
    loss_fn = nn.MSELoss()
    train_loss, valid_loss = [], []
    for t in range(epoch):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        with torch.no_grad():
            pred_eval = model(x_valid)
            loss_eval = loss_fn(pred_eval, y_valid)
            valid_loss.append(loss_eval.item())
        if t % 500 == 0:
            print(f"{t} train loss: {loss.item():.4f} valid loss: {loss_eval.item():.4f}")
    return train_loss, valid_loss


def get_data(x: np.ndarray, y: np.ndarray, scaler: str, n_input: int, n_output: int):
    x_reshaped = np.float32(x).reshape(len(x), n_input)
    y_reshaped = np.float32(y).reshape(len(y), n_output)
    if scaler == 'minmax':
        scaler_fn = MinMaxScaler()
        scaler_fn.fit(x_reshaped)
        x_reshaped = scaler_fn.transform(x_reshaped)
    elif scaler == 'standard':
        scaler_fn = StandardScaler()
        scaler_fn.fit(x_reshaped)
        x_reshaped = scaler_fn.transform(x_reshaped)

    x_train, x_tmp, y_train, y_tmp = train_test_split(x_reshaped, y_reshaped, test_size=0.3, random_state=1)
    x_valid, x_test, y_valid, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=1)

    return (torch.from_numpy(x_train), torch.from_numpy(y_train),
            torch.from_numpy(x_valid), torch.from_numpy(y_valid),
            torch.from_numpy(x_test), torch.from_numpy(y_test))


def freeze_forward(bf: BackwardForward, forward_net: Net) -> BackwardForward:
    for param in bf.forward_model.parameters():
        param.requires_grad = False
    return bf


def plot_results(x_test: torch.Tensor, y_test: torch.Tensor, y_pred: np.ndarray,
                 x_hat: np.ndarray, y_hat: np.ndarray, forward_loss: list, bf_loss: list, file_name: str) -> None:
    plt.figure(figsize=(10, 12))
    plt.subplot(2, 2, 1)
    plt.scatter(x_test, y_test, alpha=0.2, label='real')
    plt.scatter(x_test, y_pred, alpha=0.2, label='pred')
    plt.legend()
    plt.title('Forward result')

    plt.subplot(2, 2, 2)
    plt.plot(forward_loss, label='forward loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.scatter(y_test, x_hat, alpha=0.2)
    plt.title('Backward result')

    plt.subplot(2, 2, 4)
    plt.plot(bf_loss, label='bf loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    print(f"Saved figure {file_name}")


@dataclass
class Config:
    n_samples: int = 2000
    n_input: int = 1
    n_output: int = 1
    n_forward_hidden: int = 128
    n_forward_layers: int = 3
    n_backward_hidden: int = 256
    n_backward_layers: int = 3
    forward_lr: float = 1e-3
    forward_epoch: int = 2000
    backward_lr: float = 1e-3
    backward_epoch: int = 5000
    scaler: str = 'minmax'


def main(cfg: Config) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y = generate_data(cfg.n_samples, 1.0)
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_data(x, y, cfg.scaler, cfg.n_input, cfg.n_output)

    forward_net = Net(cfg.n_input, cfg.n_forward_hidden, cfg.n_output, cfg.n_forward_layers).to(device)
    optim_fwd = torch.optim.Adam(forward_net.parameters(), lr=cfg.forward_lr)
    train_loss_fwd, _ = train(forward_net, x_train.to(device), y_train.to(device), x_valid.to(device), y_valid.to(device), optim_fwd, cfg.forward_epoch)

    y_pred = forward_net(x_test.to(device)).cpu().data.numpy()
    print('Forward r2:', r2_score(y_test.numpy(), y_pred))

    backward_net = Net(cfg.n_input, cfg.n_backward_hidden, cfg.n_output, cfg.n_backward_layers)
    bf = BackwardForward(backward_net, forward_net).to(device)
    bf = freeze_forward(bf, forward_net)
    optim_bf = torch.optim.Adam(bf.parameters(), lr=cfg.backward_lr)
    train_loss_bf, _ = train(bf, y_train.to(device), y_train.to(device), y_valid.to(device), y_valid.to(device), optim_bf, cfg.backward_epoch)

    x_hat = bf.backward_model(y_test.to(device)).cpu().data.numpy()
    y_hat = bf(y_test.to(device)).cpu().data.numpy()
    print('Backward r2:', r2_score(y_test.numpy(), y_hat))

    plot_results(x_test.numpy(), y_test.numpy(), y_pred, x_hat, y_hat, train_loss_fwd, train_loss_bf, 'result.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backward-Forward demo')
    args = parser.parse_args()
    cfg = Config()
    main(cfg)
