from torch.utils.data import Dataset
import h5py
from typing import Dict, Any, Tuple, Protocol
import torch


class ActivationLoader(Protocol):
    def __call__(self, neuron_index: int) -> torch.Tensor:
        """
        Load activations for a specific neuron index.

        Args:
            neuron_index (int): The index of the neuron to load activations for.

        Returns:
            torch.Tensor: The activations for the specified neuron.
        """
        ...



class DictionaryDataset(Dataset):
    def __init__(self, data_dict: Dict[Any, Any]):
        """
        Initialize a dataset from a dictionary.

        Args:
            data_dict (Dict[Any, Any]): A dictionary of data items to be used in the dataset.
        """
        # enumerate the dictionary items
        self.data_items = list(data_dict.items())

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        """
        Get a data item by index from the newly created dataset.

        Args:
            idx (int): Index of the data item to be retrieved.

        Returns:
            Tuple[Any, Any]: A tuple containing the key and value of the data item from the dictionary.
        """
        key, value = self.data_items[idx]
        return key, value
