import torch

from torch.utils.data import Dataset

from typing import Literal
from lib.flowframework.packetdataset import PacketDataset

TripletGenerator = Literal['random', 'hard']

class PacketDatsetTriplet(Dataset):
    def __init__(self, packet_dataset: PacketDataset, inflation_factor: int = 10,
                 triplet_generator: TripletGenerator = 'random'):
        """Create a triplet dataset from a packet dataset.

        Args:
            packet_dataset (PacketDataset): Packet dataset to create triplet dataset from.
            inflation_factor (int, optional): How many samples per entry. Defaults to 10.
            triplet_generator (TripletGenerator, optional): Way of generation. Defaults to 'random'.
        """
        self.inflation_factor = inflation_factor
        self.triplet_generator = triplet_generator

        if triplet_generator == 'random':
            self.__create_triplet_dataset_random(packet_dataset)

        if triplet_generator == 'hard':
            self.__create_triplet_dataset_hard(packet_dataset)

    def __create_triplet_dataset_random(self, packet_dataset: PacketDataset):
        from tqdm import tqdm

        N = len(packet_dataset)
        self.data_tensor = torch.zeros(
            (N * self.inflation_factor, 3, packet_dataset.chunk_size, 4),
            dtype=torch.float32
        )

        dataset_labels = packet_dataset.label_tensor

        print("Generating triplet dataset with random sampling...")
        progress_bar = tqdm(
            total = N * self.inflation_factor,
            desc  = "Generating triplets",
            unit  = "triplet"
        )
        progress_bar.colour = 'green'

        for i in range(N):
            current_label = dataset_labels[i].numpy()[0]
            for j in range(self.inflation_factor):
                # Randomly select a positive and negative sample
                positive_indices = torch.where(dataset_labels == current_label)[0]
                negative_indices = torch.where(dataset_labels != current_label)[0]

                # Drop the current index from the positive indices
                positive_indices = positive_indices[positive_indices != i]

                random_pos_idx = torch.randint(0, len(positive_indices), (1,)).item()
                random_neg_idx = torch.randint(0, len(negative_indices), (1,)).item()

                positive_idx = positive_indices[random_pos_idx]
                negative_idx = negative_indices[random_neg_idx]

                tensor_index = i * self.inflation_factor + j

                if i == positive_idx or i == negative_idx or positive_idx == negative_idx:
                    assert False, "Triplet generation failed. Indices are not unique."

                # Create the triplet
                self.data_tensor[tensor_index, 0] = packet_dataset.data_tensor[i]
                self.data_tensor[tensor_index, 1] = packet_dataset.data_tensor[positive_idx]
                self.data_tensor[tensor_index, 2] = packet_dataset.data_tensor[negative_idx]

                progress_bar.update(1)

        progress_bar.close()
        print("Triplet dataset created with shape: ", self.data_tensor.shape)

    def __create_triplet_dataset_hard(self, packet_dataset: PacketDataset):
        assert False, "Hard triplet generation is not implemented yet."
        pass

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return self.data_tensor[idx]