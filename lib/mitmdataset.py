import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

class MitmProxyLogsDataset(torch.utils.data.Dataset):
    MAX_VALUE_DIFF = -1
    MAX_VALUE_BYTES = -1

    def __init__(self, df=None, training: bool = True, chunk_size=64, scale_zero_to_one=True, subtract_mean=False, apply_highpass=False, use_highpass_as_time_series=False, scale_negative_separately=False):

        self.scale_zero_to_one = scale_zero_to_one
        self.subtract_mean = subtract_mean
        self.apply_highpass = apply_highpass
        self.use_highpass_as_time_series = use_highpass_as_time_series
        self.scale_negative_separately = scale_negative_separately

        if df is not None:
            self.df = df
            self.chunk_size = chunk_size

            if training:
                self.get_normalization_values()

            self.__preprocess_dataframe(self.df)

        print("-------------------------------------------------------")

    def apply_highpass_filter(self, array):
        array = array.numpy().copy()
        b, a = signal.butter(8, 0.1, btype='high', analog=False)
        return signal.filtfilt(b, a, array)

    def get_normalization_values(self):
        diff_arr = self.df.groupby('filename')['receive_time_us'].diff().dropna()

        MitmProxyLogsDataset.MAX_VALUE_DIFF  = diff_arr.max().value
        MitmProxyLogsDataset.MAX_VALUE_BYTES = self.df['bytes'].max()
        MitmProxyLogsDataset.MEAN_VALUE_DIFF = diff_arr.mean().value
        MitmProxyLogsDataset.MEAN_VALUE_BYTES = self.df['bytes'].mean()
        MitmProxyLogsDataset.STD_VALUE_DIFF = diff_arr.std().value
        MitmProxyLogsDataset.STD_VALUE_BYTES = self.df['bytes'].std()
        MitmProxyLogsDataset.MIN_VALUE_BYTES = self.df['bytes'].min()

        print("Dataset Statistics")
        print("-------------------------------------------------------")
        print(f"Max value diff:   {MitmProxyLogsDataset.MAX_VALUE_DIFF}")
        print(f"Max value bytes:  {MitmProxyLogsDataset.MAX_VALUE_BYTES}")
        print(f"Mean value diff:  {MitmProxyLogsDataset.MEAN_VALUE_DIFF}")
        print(f"Mean value bytes: {MitmProxyLogsDataset.MEAN_VALUE_BYTES}")
        print(f"Std value diff:   {MitmProxyLogsDataset.STD_VALUE_DIFF}")
        print(f"Std value bytes:  {MitmProxyLogsDataset.STD_VALUE_BYTES}")


    def __preprocess_dataframe(self, df):
        label_index = 0

        minimum_group_size = 1_000_000_000

        for _, group in df.groupby('filename'):
            if len(group) // self.chunk_size < minimum_group_size:
                minimum_group_size = len(group) // self.chunk_size

        print(f"Chunks per group: {minimum_group_size}")

        self.data_tensor = torch.zeros((0, 4, self.chunk_size))
        self.label_tensor = torch.zeros((0))

        for name, group in df.groupby('filename'):
            group = group.sort_values(by=['receive_time_us'])
            group = group.reset_index(drop=True)

            tensor_length = len(group) // self.chunk_size

            data_tensor = torch.zeros((minimum_group_size, 4, self.chunk_size))
            label_tensor = torch.ones((minimum_group_size)) * label_index

            group['receive_time_us_int'] = group['receive_time_us'].astype(int)
            group['receive_time_us_int'] = group['receive_time_us_int'] - group['receive_time_us_int'].iloc[0]

            group['diff'] = group['receive_time_us_int'].diff().fillna(0)
            group['label'] = label_index

            # Normalize diff and bytes
            if self.scale_zero_to_one:
                if not self.scale_negative_separately:
                    group['diff_normalized'] = group['diff'] / MitmProxyLogsDataset.MAX_VALUE_DIFF
                    group['bytes_normalized'] = group['bytes'] / MitmProxyLogsDataset.MAX_VALUE_BYTES
                else:
                    group['diff_normalized'] = group['diff'] / MitmProxyLogsDataset.MAX_VALUE_DIFF
                    group['bytes_normalized'] = group['bytes']
                    group['bytes_normalized'] = group['bytes_normalized'].astype(float)

                    for index, row in group.iterrows():
                        bytes_val = row['bytes']
                        if bytes_val < 0:
                            group.at[index, 'bytes_normalized'] = bytes_val / (MitmProxyLogsDataset.MIN_VALUE_BYTES * -1)
                        else:
                            group.at[index, 'bytes_normalized'] = bytes_val / MitmProxyLogsDataset.MAX_VALUE_BYTES
            else:
                # Scale with mean and std
                group['diff_normalized'] = (group['diff'] - MitmProxyLogsDataset.MEAN_VALUE_DIFF) / MitmProxyLogsDataset.STD_VALUE_DIFF
                group['bytes_normalized'] = (group['bytes'] - MitmProxyLogsDataset.MEAN_VALUE_BYTES) / MitmProxyLogsDataset.STD_VALUE_BYTES

            inter_arrival_times_arr = group['diff_normalized'].values
            bytes_arr = group['bytes_normalized'].values

            # Create tensor
            inter_arrival_times_tensor = torch.tensor(inter_arrival_times_arr)
            bytes_tensor = torch.tensor(bytes_arr)

            # Create chunks with unfold
            inter_arrival_times_chunks = inter_arrival_times_tensor.unfold(0, self.chunk_size, self.chunk_size)
            bytes_chunks = bytes_tensor.unfold(0, self.chunk_size, self.chunk_size)

            inter_arrival_times_chunks_highpass = None
            bytes_chunks_highpass = None

            if self.apply_highpass:
                inter_arrival_times_chunks_highpass = torch.ones_like(inter_arrival_times_chunks)
                bytes_chunks_highpass = torch.ones_like(bytes_chunks)

                for i in range(len(inter_arrival_times_chunks)):
                    arrival_times_highpass = self.apply_highpass_filter(inter_arrival_times_chunks[i])
                    bytes_chunk_highpass = self.apply_highpass_filter(bytes_chunks[i])

                    inter_arrival_times_chunks_highpass[i] = torch.tensor(arrival_times_highpass.copy())
                    bytes_chunks_highpass[i] = torch.tensor(bytes_chunk_highpass.copy())


            if self.subtract_mean:
                # Subtract the mean from the chunks
                inter_arrival_times_chunks = inter_arrival_times_chunks - inter_arrival_times_chunks.mean(dim=1).unsqueeze(1)
                bytes_chunks = bytes_chunks - bytes_chunks.mean(dim=1).unsqueeze(1)

            # Get random indices, to create a balanced dataset with random chunks from every user
            indices = torch.randperm(tensor_length)
            tensor_index = 0

            for i in indices[:minimum_group_size]:
                arrival_time_chunk = inter_arrival_times_chunks[i]
                bytes_chunk = bytes_chunks[i]

                data_tensor[tensor_index, 0, :] = inter_arrival_times_chunks[i]
                data_tensor[tensor_index, 1, :] = bytes_chunks[i]

                if self.use_highpass_as_time_series:
                    data_tensor[tensor_index, 0, :] = inter_arrival_times_chunks_highpass[i]
                    data_tensor[tensor_index, 1, :] = bytes_chunks_highpass[i]

                if self.apply_highpass:
                    arrival_time_fft = torch.fft.fft(inter_arrival_times_chunks_highpass[i])
                    bytes_fft = torch.fft.fft(bytes_chunks_highpass[i])
                else:
                    arrival_time_fft = torch.fft.fft(arrival_time_chunk)
                    bytes_fft = torch.fft.fft(bytes_chunk)

                data_tensor[tensor_index, 2, :] = torch.abs(arrival_time_fft)
                data_tensor[tensor_index, 3, :] = torch.abs(bytes_fft)

                tensor_index += 1

            self.data_tensor = torch.cat((self.data_tensor, data_tensor), 0)
            self.label_tensor = torch.cat((self.label_tensor, label_tensor), 0)

            label_index += 1

    def from_tensors(self, data, labels):
        self.data_tensor = data
        self.label_tensor = labels
        self.chunk_size = data.shape[2]

    def save_dataset_as_numpy_array(self, file_path):
        np.save(file_path + "-data", self.data_tensor.numpy())
        np.save(file_path + "-labels", self.label_tensor.numpy())

    def plot_datapoint(self, idx):
        data = self.data_tensor[idx]
        label = self.label_tensor[idx]

        plt.title(f"Label: {label} Index: {idx}")

        plt.subplot(221)
        plt.plot(data[0], label='Inter arrival times')
        plt.ylim(0, 1)
        plt.subplot(222)
        plt.plot(data[1], label='Bytes')
        plt.ylim(0, 1)
        plt.subplot(223)
        plt.plot(data[2], label='Inter arrival times FFT')
        plt.subplot(224)
        plt.plot(data[3], label='Bytes FFT')

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.label_tensor[idx]

def train_test_split_mitmproxydataset(dataset: MitmProxyLogsDataset, test_size=0.2):
    n       = len(dataset)
    n_test  = int(n * test_size)
    n_train = n - n_test

    x = dataset.data_tensor
    y = dataset.label_tensor

    random_indices = torch.randperm(n)

    training_x = x[random_indices[:n_train]]
    training_y = y[random_indices[:n_train]]

    testing_x = x[random_indices[n_train:]]
    testing_y = y[random_indices[n_train:]]

    training_dataset = MitmProxyLogsDataset()
    training_dataset.from_tensors(training_x, training_y)

    testing_dataset = MitmProxyLogsDataset()
    testing_dataset.from_tensors(testing_x, testing_y)

    return training_dataset, testing_dataset