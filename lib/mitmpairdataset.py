import torch
import matplotlib.pyplot as plt
import numpy as np

from .mitmdataset import MitmProxyLogsDataset

class MitmProxyPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]

    def plot(self, idx, prepermute=None):
        data = self.pairs[idx]

        if prepermute is not None:
            data = torch.permute(data, prepermute)

        plt.subplot(421)
        plt.plot(data[0][0])
        plt.ylabel("Latency")
        plt.ylim(-1, 1)
        plt.subplot(422)
        plt.plot(data[1][0])
        plt.ylim(-1, 1)
        plt.subplot(423)
        plt.plot(data[0][1])
        plt.ylabel("Bytes")
        plt.ylim(-1, 1)
        plt.subplot(424)
        plt.plot(data[1][1])
        plt.ylim(-1, 1)
        plt.subplot(425)
        plt.plot(data[0][2])
        plt.ylabel("FFT Lat.")
        plt.subplot(426)
        plt.plot(data[1][2])
        plt.subplot(427)
        plt.plot(data[0][3])
        plt.ylabel("FFT Bytes")
        plt.subplot(428)
        plt.plot(data[1][3])

def create_pairs_from_mitm_arrays(data, labels):
    x = data
    y = labels
    n = len(y)

    final_pair_tensor = torch.zeros((n * 2, 2, x.shape[1], x.shape[2]))
    final_pair_labels = torch.zeros(n * 2)
    final_pair_labels[:n] = 1

    for i in range(n):
        current_label = y[i]
        matching_indices = torch.where(y == current_label)
        # Select a random index from the matching indices
        random_index = np.random.choice(matching_indices[0].numpy())
        final_pair_tensor[i][0] = x[i]
        final_pair_tensor[i][1] = x[random_index]

    # Add non-matching pairs
    for i in range(n, n * 2):
        current_label = y[i % n]
        non_matching_indices = torch.where(y != current_label)
        random_index = np.random.choice(non_matching_indices[0].numpy())
        final_pair_tensor[i][0] = x[i % n]
        final_pair_tensor[i][1] = x[random_index]

    return final_pair_tensor, final_pair_labels

def create_pairs_from_mitm_dataset(dataset: MitmProxyLogsDataset):
    x = dataset.data_tensor
    y = dataset.label_tensor

    return create_pairs_from_mitm_array(x, y)