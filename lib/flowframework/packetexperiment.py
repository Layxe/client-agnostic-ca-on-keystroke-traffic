import pandas as pd

from pandas import DataFrame
from typing import Literal

from lib.flowframework.packetdataset import PacketDataset

# Constants
# ##################################################################################################

NormalizationMethod = Literal['minmax', 'meanstd']

# Functions
# ##################################################################################################

# This function needs to be outside of the class, so it can be used in multiprocessing
def get_flow_for_subgroup(group, label, fold, is_to, flow_sampling_rate):
    flow_start_time = group['receive_time_ms'].min()
    group = group.sort_values(by='receive_time_ms')
    results = []

    total_packet_size = 0
    packet_count = 0

    for row in group.itertuples():
        receive_time_ms = getattr(row, 'receive_time_ms')
        packet_size = getattr(row, 'packet_size')

        if receive_time_ms - flow_start_time >= flow_sampling_rate * 1000:
            # Store the current flow in a new dataframe
            entry = {
                'receive_time_ms': flow_start_time,
                'packet_size': total_packet_size,
                'label': label,
                'fold': fold,
                'packet_count': packet_count,
                'is_to': is_to,
                'session_id': getattr(row, 'session_id'),
            }

            results.append(entry)

            # Reset the flow
            packet_count = 1
            total_packet_size = packet_size
            flow_start_time = receive_time_ms
        else:
            packet_count += 1
            total_packet_size += packet_size

    # Add the last flow
    if packet_count > 0:
        entry = {
            'receive_time_ms': flow_start_time,
            'packet_size': total_packet_size,
            'label': label,
            'fold': fold,
            'packet_count': packet_count
        }
        results.append(entry)

    return results


# Class
# ##################################################################################################

class PacketDataExperiment:

    ORIGINAL_SAMPLING_RATE_HZ = 20

    def __init__(self, dataframe: DataFrame, normalization_method: NormalizationMethod = 'minmax', flow_sampling_rate_s: float = None, chunk_size: int = 1024, overlap: int = 512):

        print("Initializing PacketDataExperiment... ", end=' ')
        dataframe = dataframe.copy()
        print("Dataframe copied")

        if 'packet_count' not in dataframe.columns:
            dataframe['packet_count'] = 1 # Add packet count, if not present

        self.dataframe = dataframe
        self.normalization_method = normalization_method
        self.flow_sampling_rate = flow_sampling_rate_s
        self.chunk_size = chunk_size
        self.overlap = overlap

        if self.flow_sampling_rate is not None:
            assert self.flow_sampling_rate > 0, "flow_sampling_rate must be greater than 0"
            assert self.flow_sampling_rate > 1/PacketDataExperiment.ORIGINAL_SAMPLING_RATE_HZ, \
            "Since the original data is sampled at 20 Hz, the flow_sampling_rate must be greater than 1/20"

            self.dataframe = self.__subsample_dataframe(dataframe)

        print("Adding receive_time_diff to dataframe... ", end=' ')
        self.__add_receive_time_diff_to_dataframe(self.dataframe)
        print("receive_time_diff added")

    def __subsample_dataframe(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame), "dataframe must be a pandas DataFrame"
        assert 'receive_time_diff' not in dataframe.columns, "Perform subsampling before adding receive_time_diff"

        print("Starting subsampling...")

        # Group by label and fold, then summarize packets accordingly
        grouped = dataframe.groupby(['label', 'fold', 'is_to'])

        argument_list = []

        for label, fold, is_to in grouped.groups.keys():
            group = grouped.get_group((label, fold, is_to))

            argument_list.append((group.copy(), label, fold, is_to, self.flow_sampling_rate))

        # Use multiprocessing to speed up the process
        from multiprocessing import Pool
        with Pool() as pool:
            results = pool.starmap(get_flow_for_subgroup, argument_list)
            results = [pd.DataFrame(result) for result in results]

        new_dataframe = pd.concat(results, ignore_index=True)
        new_dataframe = new_dataframe.sort_values(by='receive_time_ms')
        new_dataframe = new_dataframe.reset_index(drop=True)

        print(f"Subsampling at {self.flow_sampling_rate} seconds, resulting in {len(new_dataframe)} rows from {len(dataframe)} rows")

        return new_dataframe

    def __add_receive_time_diff_to_dataframe(self, dataframe):
        # Group by fold and label, then sort by receive_time_ms and calculate the diff
        dataframe['receive_time_diff'] = dataframe.groupby(['fold', 'label'])['receive_time_ms'].diff()
        # Fill NaN values with 0
        dataframe['receive_time_diff'] = dataframe['receive_time_diff'].fillna(0)
        # Clamp the diff to a maximum value of 10 seconds
        dataframe['receive_time_diff'] = dataframe['receive_time_diff'].clip(upper=10000)
        # Convert to milliseconds
        dataframe['receive_time_diff'] = dataframe['receive_time_diff'].astype('int32')

    def __get_normalization_metrics_from_dataframe(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame), "dataframe must be a pandas DataFrame"
        assert 'receive_time_diff' in dataframe.columns, "dataframe must contain the column 'receive_time_diff'"

        max_value_diff = dataframe['receive_time_diff'].max()
        min_value_diff = dataframe['receive_time_diff'].min()
        mean_value_diff = dataframe['receive_time_diff'].mean()
        std_value_diff = dataframe['receive_time_diff'].std()

        max_value_packet_size = dataframe['packet_size'].max()
        min_value_packet_size = dataframe['packet_size'].min()
        mean_value_packet_size = dataframe['packet_size'].mean()
        std_value_packet_size = dataframe['packet_size'].std()

        max_value_packet_count = dataframe['packet_count'].max()
        min_value_packet_count = dataframe['packet_count'].min()
        mean_value_packet_count = dataframe['packet_count'].mean()
        std_value_packet_count = dataframe['packet_count'].std()

        return {
            'receive_time_diff': {
                'max': max_value_diff, 'min': min_value_diff,
                'mean': mean_value_diff, 'std': std_value_diff
            },
            'packet_size': {
                'max': max_value_packet_size, 'min': min_value_packet_size,
                'mean': mean_value_packet_size, 'std': std_value_packet_size
            },
            'packet_count': {
                'max': max_value_packet_count, 'min': min_value_packet_count,
                'mean': mean_value_packet_count, 'std': std_value_packet_count
            }
        }

    def __create_dataset_from_dataframe(self, dataframe, metrics):
        if self.normalization_method == 'minmax':
            receive_time_min = metrics['receive_time_diff']['min']
            receive_time_max = metrics['receive_time_diff']['max']
            packet_size_min = metrics['packet_size']['min']
            packet_size_max = metrics['packet_size']['max']

            dataframe['receive_time_diff'] = (dataframe['receive_time_diff'] - receive_time_min) / (receive_time_max - receive_time_min)
            dataframe['packet_size'] = (dataframe['packet_size'] - packet_size_min) / (packet_size_max - packet_size_min)
            dataframe['packet_count'] = (dataframe['packet_count'] - metrics['packet_count']['min']) / (metrics['packet_count']['max'] - metrics['packet_count']['min'])

        if self.normalization_method == 'meanstd':
            receive_time_mean = metrics['receive_time_diff']['mean']
            receive_time_std = metrics['receive_time_diff']['std']
            packet_size_mean = metrics['packet_size']['mean']
            packet_size_std = metrics['packet_size']['std']

            dataframe['receive_time_diff'] = (dataframe['receive_time_diff'] - receive_time_mean) / receive_time_std
            dataframe['packet_size'] = (dataframe['packet_size'] - packet_size_mean) / packet_size_std
            dataframe['packet_count'] = (dataframe['packet_count'] - metrics['packet_count']['mean']) / metrics['packet_count']['std']

        dataframe = dataframe.drop(columns=['fold'])

        # Convert the dataframe to a PyTorch dataset
        dataset = PacketDataset(dataframe, chunk_size=self.chunk_size, overlap=self.overlap)

        return dataset

    def get_training_and_test_dataset(self, fold):
        train_df = self.dataframe[self.dataframe['fold'] != fold]
        test_df = self.dataframe[self.dataframe['fold'] == fold]

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        normalization_metrics = self.__get_normalization_metrics_from_dataframe(train_df)

        training_dataset = self.__create_dataset_from_dataframe(train_df, normalization_metrics)
        test_dataset = self.__create_dataset_from_dataframe(test_df, normalization_metrics)

        return training_dataset, test_dataset