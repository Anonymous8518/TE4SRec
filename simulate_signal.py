import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import utils
from model.temporal_encoding import fourier_encoding, temporal_encoding

step_num = 1000
min_ts = 1419426000
max_ts = 1703422800
timestamps = np.linspace(min_ts, max_ts, num=step_num)
utils.global_seed(42)
sns.set_theme()
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 300


def random_dynamics(x):
    y = 2
    result = []
    for _ in x:
        result.append(y)
        y += np.random.normal(scale=0.05)
    return np.array(result)


class SignalDataset(Dataset):
    def __init__(self, timestamps, signal, d_hidden, min_ts, angular_velocity, is_fourier=False, pow_base=10000):
        self.timestamp = timestamps
        self.signal = signal
        if is_fourier:
            self.temporal_encodings = fourier_encoding(timestamps,
                                                       d_hidden,
                                                       min_ts,
                                                       angular_velocity)
        else:
            self.temporal_encodings = temporal_encoding(timestamps,
                                                        d_hidden,
                                                        min_ts,
                                                        angular_velocity,
                                                        pow_base)

    def __len__(self):
        return len(self.timestamp)

    def __getitem__(self, idx):
        return self.temporal_encodings[idx], self.signal[idx]


class TemporalSimulator(torch.nn.Module):
    def __init__(self, d_hidden):
        super(TemporalSimulator, self).__init__()
        self.linear = torch.nn.Linear(d_hidden, 1)

    def forward(self, x):
        return self.linear(x)


def simulate_signal(timestamps, signal, min_ts, d_hidden, is_fourier=False, alpha=0.5, device='cuda', batch_num=50):
    angular_velocity = 2 * math.pi * alpha / (max_ts - min_ts)

    dataset = SignalDataset(timestamps, signal, d_hidden, min_ts, angular_velocity, is_fourier=is_fourier)
    dataloader = DataLoader(dataset, batch_size=32)

    model = TemporalSimulator(d_hidden).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))

    for i in range(batch_num):
        loss_arr = []
        for x, y_l in dataloader:
            x = x.to(device)
            y_l = y_l.to(device)
            # print(y_l)
            y_p = model.forward(x).squeeze(-1)

            optimizer.zero_grad()
            loss = criterion(y_p, y_l)
            loss.backward()
            optimizer.step()

            loss_arr.append(loss.item())

        print(f'loss: {sum(loss_arr) / len(loss_arr)}')

    ts_tensor = torch.Tensor(dataset.temporal_encodings).to(device)
    pred_signal = model.forward(ts_tensor).to('cpu').detach().numpy().reshape(signal.shape)

    return pred_signal


def compare_simulate(timestamps, signal, min_ts, d_hidden, batch_num=50):
    print("Fourier Series")
    fs_signal = simulate_signal(timestamps, signal, min_ts, d_hidden, is_fourier=True, batch_num=batch_num)
    print("Temporal Encodings")
    te_signal = simulate_signal(timestamps, signal, min_ts, d_hidden, is_fourier=False, batch_num=batch_num)

    plt.figure(figsize=(8, 4))

    temporal_points = [datetime.fromtimestamp(ts) for ts in timestamps]
    sns.lineplot(x=temporal_points, y=signal, label='Original Signal')
    sns.lineplot(x=temporal_points, y=fs_signal, label='Fourier Series')
    sns.lineplot(x=temporal_points, y=te_signal, label='Temporal Encodings')

    plt.xlabel('Year')
    plt.ylabel('Signal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./image/simulate_{d_hidden}.png')
    plt.show()


if __name__ == '__main__':
    signal = np.array(random_dynamics(timestamps), dtype=np.float32)

    plt.figure(figsize=(8, 4))
    sns.lineplot(x=[datetime.fromtimestamp(ts) for ts in timestamps], y=signal, )

    compare_simulate(timestamps, signal, min_ts, d_hidden=32, batch_num=100)
    compare_simulate(timestamps, signal, min_ts, d_hidden=64, batch_num=100)
