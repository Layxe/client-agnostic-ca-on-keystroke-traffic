import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FlowDataset(Dataset):
    DATASET_PACKET_MEAN    = 0
    DATASET_PACKET_STD     = -1
    DATASET_BYTES_MEAN     = 0
    DATASET_BYTES_STD      = -1
    DATASET_DIFF_TIME_MEAN = 0
    DATASET_DIFF_TIME_STD  = -1
    DATASET_CONSTANTS_SET  = False

    def __init__(self, dataframe: pd.DataFrame, training=True, chunk_size=64, overlap=32, verbose=False):
        self.dataframe  = dataframe.copy()
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.verbose    = verbose

        # Remove time skips
        self.dataframe = self.__remove_timeskips(self.dataframe)

        if training:
            self.__get_dataset_constants(self.dataframe)

        if FlowDataset.DATASET_CONSTANTS_SET:
            self.__prepare_data(self.dataframe)
            self.data, self.labels = self.__generate_tensors(self.dataframe, chunk_size, overlap)
        else:
            raise Exception("Dataset constants not set, load training data first.")

    def __remove_timeskips(self, dataframe: pd.DataFrame):
        dataframe["receive_time_diff"] = dataframe.groupby("label")["receive_time"].diff()
        # Remove time skips larger than 10 minutes
        dataframe = dataframe[dataframe["receive_time_diff"] < 600]
        # Drop all NaN values
        dataframe = dataframe.dropna()

        return dataframe

    def __get_dataset_constants(self, dataframe: pd.DataFrame):
        FlowDataset.DATASET_PACKET_MEAN    = dataframe["packets"].mean()
        FlowDataset.DATASET_PACKET_STD     = dataframe["packets"].std()
        FlowDataset.DATASET_BYTES_MEAN     = dataframe["bytes"].mean()
        FlowDataset.DATASET_BYTES_STD      = dataframe["bytes"].std()
        FlowDataset.DATASET_DIFF_TIME_MEAN = dataframe["receive_time_diff"].mean()
        FlowDataset.DATASET_DIFF_TIME_STD  = dataframe["receive_time_diff"].std()
        FlowDataset.DATASET_CONSTANTS_SET  = True

        if self.verbose:
            print("Dataset Constants")
            print("----------------------------------------")
            print(f"Packets mean: {round(self.DATASET_PACKET_MEAN, 2)} | std: {round(self.DATASET_PACKET_STD, 2)}")
            print(f"Bytes mean: {round(self.DATASET_BYTES_MEAN, 2)} | std: {round(self.DATASET_BYTES_STD, 2)}")
            print(f"Diff time mean: {round(self.DATASET_DIFF_TIME_MEAN, 2)} | std: {round(self.DATASET_DIFF_TIME_STD, 2)}")
            print("----------------------------------------")

    def __normalize(self, series, mean, std):
        return (series - mean) / std

    def __generate_tensors(self, orig_dataframe: pd.DataFrame, chunk_size=64, overlap=32):
        # Generate tensors
        unique_labels = orig_dataframe["label"].unique()

        data_tensors  = []
        label_tensors = []

        for label in unique_labels:
            dataframe = orig_dataframe[orig_dataframe["label"] == label]

            packets      = torch.tensor(dataframe["packets"].values, dtype=torch.float32)
            bytes_tensor = torch.tensor(dataframe["bytes"].values, dtype=torch.float32)
            diff         = torch.tensor(dataframe["receive_time_diff"].values, dtype=torch.float32)
            # Label
            # label = torch.tensor(dataframe["label"].values, dtype=torch.float32)
            # label =

            # Split into chunks
            packets        = packets.unfold     (0, chunk_size, overlap)
            bytes_tensor   = bytes_tensor.unfold(0, chunk_size, overlap)
            diff           = diff.unfold        (0, chunk_size, overlap)
            # label          = label.unfold       (0, chunk_size, overlap)

            # FFT
            fft1 = torch.fft.rfft(packets, dim=1).abs()
            fft2 = torch.fft.rfft(diff, dim=1).abs()

            # Remove last entry from both FFTs so they have the dimension of CHUNK_SIZE/2
            fft1 = fft1[:, :-1]
            fft2 = fft2[:, :-1]

            # Concatenate FFTs
            fft = torch.cat([fft1, fft2], dim=1)

            # Remove the last chunk
            packets      = packets[:-1]
            bytes_tensor = bytes_tensor[:-1]
            diff         = diff[:-1]
            fft          = fft[:-1]

            # Concat into one tensor
            data_tensor = torch.stack([packets, bytes_tensor, diff, fft], dim=2)
            label_tensor = torch.ones(len(data_tensor)) * label
            label_tensor = label_tensor.long()

            data_tensors.append(data_tensor)
            label_tensors.append(label_tensor)

        final_data_tensor = torch.cat(data_tensors, dim=0)
        final_label_tensor = torch.cat(label_tensors, dim=0)

        return final_data_tensor, final_label_tensor

    def __prepare_data(self, dataframe: pd.DataFrame):
        dataframe["packets"] = self.__normalize(
            dataframe["packets"],
            FlowDataset.DATASET_PACKET_MEAN,
            FlowDataset.DATASET_PACKET_STD
        )
        dataframe["bytes"] = self.__normalize(
            dataframe["bytes"],
            FlowDataset.DATASET_BYTES_MEAN,
            FlowDataset.DATASET_BYTES_STD
        )
        dataframe["receive_time_diff"] = self.__normalize(
            dataframe["receive_time_diff"],
            FlowDataset.DATASET_DIFF_TIME_MEAN,
            FlowDataset.DATASET_DIFF_TIME_STD
        )

    def plot_datapoint(self, idx):
        test_vals = self.data[idx]
        title_fontdict = {"fontsize": 12}

        # Plot example test item
        plt.figure(figsize=(16, 9))
        ax1 = plt.subplot(221)
        plt.plot(test_vals[:, 0], label="Packets")
        plt.xlabel("Flow index [#]")
        plt.ylabel("Packet Count Normalized")

        ax2 = plt.subplot(222)
        plt.plot(test_vals[:, 1], label="Bytes")
        plt.xlabel("Flow index [#]")
        plt.ylabel("Byte Count Normalized")

        ax3 = plt.subplot(223)
        plt.plot(test_vals[:, 2], label="Diff time")
        plt.xlabel("Flow index [#]")
        plt.ylabel("Time difference Normalized")

        ax4 = plt.subplot(224)
        plt.plot(test_vals[:, 3], label="FFT")
        plt.xlabel("Frequency [Bins]")
        plt.ylabel("Magnitude")
        plt.vlines(self.chunk_size // 2, 0, test_vals[:, 3].max(), color="black")

        ax1.set_title("Packets", fontdict=title_fontdict)
        ax2.set_title("Bytes", fontdict=title_fontdict)
        ax3.set_title("Difference receive time", fontdict=title_fontdict)
        ax4.set_title("FFT of packets and receive time diff", fontdict=title_fontdict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
