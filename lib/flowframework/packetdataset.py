import torch
import matplotlib.pyplot as plt

from pandas import DataFrame

from torch.utils.data import Dataset

class PacketDataset(Dataset):
    def __init__(self, dataframe: DataFrame, chunk_size: int = 1024,  overlap: int = 512):
        self.data_tensor = None
        self.label_tensor = None

        self.chunk_size = chunk_size
        self.overlap = overlap

        self.__create_tensors(dataframe)

    def __create_tensors(self, dataframe):
        print("Creating tensor from dataframe with size: ", dataframe.shape)

        # Group by label, sort by receive time and create chunks
        grouped = dataframe.groupby('label')

        final_amount_of_chunks = 0
        num_packages_per_label = len(dataframe[dataframe['label'] == 0])
        num_chunks_per_label = round((num_packages_per_label // self.chunk_size) * (self.chunk_size / self.overlap)) + 5
        final_amount_of_chunks = num_chunks_per_label * dataframe['label'].nunique()

        self.data_tensor = torch.zeros((final_amount_of_chunks, self.chunk_size, 4), dtype=torch.float32)
        self.label_tensor = torch.zeros((final_amount_of_chunks, 1), dtype=torch.int64)

        data_tensors = []
        label_tensors = []

        for label, group in grouped:
            # Sort the group by receive_time_ms
            group = group.sort_values(by='receive_time_ms')

            receive_time_diff_tensor = torch.tensor(group['receive_time_diff'].values, dtype=torch.float32)
            packet_size_tensor = torch.tensor(group['packet_size'].values, dtype=torch.float32)
            packet_count_tensor = torch.tensor(group['packet_count'].values, dtype=torch.float32)
            is_to_tensor = torch.tensor(group['is_to'].values, dtype=torch.float32)

            is_to_tensor[is_to_tensor.isnan()] = 1

            # Split into chunks
            receive_time_diff_tensor = receive_time_diff_tensor.unfold(0, self.chunk_size, self.overlap)
            packet_size_tensor = packet_size_tensor.unfold(0, self.chunk_size, self.overlap)
            packet_count_tensor = packet_count_tensor.unfold(0, self.chunk_size, self.overlap)
            is_to_tensor = is_to_tensor.unfold(0, self.chunk_size, self.overlap)

            # Concatenate the tensors
            chunk_tensor = torch.stack([receive_time_diff_tensor, packet_size_tensor, packet_count_tensor, is_to_tensor], dim=2)
            label_tensor = torch.ones(chunk_tensor.shape[0], 1) * label

            data_tensors.append(chunk_tensor)
            label_tensors.append(label_tensor)

        self.data_tensor = torch.cat(data_tensors, dim=0)
        self.label_tensor = torch.cat(label_tensors, dim=0)

        print("Created tensor with shape: ", self.data_tensor.shape)

    def plot(self, idx):
        Xs = self[idx][0]
        y = self[idx][1]

        plt.figure(figsize=(10, 5))
        plt.suptitle(f"Label: {y.item()}", fontsize=12)
        ax = plt.subplot(221)
        ax.set_title('Receive Time Diff')
        ax.plot(Xs[:, 0].numpy())
        ax.set_ylabel('Receive Time Diff')
        ax = plt.subplot(222)
        ax.set_title('Packet Size')
        ax.plot(Xs[:, 1].numpy())
        ax.set_ylabel('Packet Size')
        ax = plt.subplot(223)
        ax.set_title('Packet Count')
        ax.plot(Xs[:, 2].numpy())
        ax.set_ylabel('Packet Count')
        ax = plt.subplot(224)
        ax.set_title('Is To')
        ax.plot(Xs[:, 3].numpy())
        ax.set_ylabel('Is To [0/1]')
        ax.set_xlabel('Packet Index')
        plt.tight_layout()

    def __len__(self):
        return self.data_tensor.shape[0]

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.label_tensor[idx]
