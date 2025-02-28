from torch.utils.data import Dataset
import h5py
from typing import Dict, Any, Tuple


class CachedActivationLoader:
    def __init__(self, file_path: str):
        """
        Initialize the cached activation loader.

        Args:
            file_path (str): Path to the H5Py file containing cached activations.
        """
        self.cache_path = file_path
        self.cache_file = h5py.File(file_path, "r")

    def __del__(self):
        """
        Close the cache file when the object is deleted.
        """
        if hasattr(self, 'cache_file') and self.cache_file:
            self.cache_file.close()

    def load_activations(self, neuron_index: int) -> Any:
        """
        Load activations for a specific neuron index.

        Args:
            neuron_index (int): The index of the neuron to load activations for.

        Returns:
            Any: The activations for the specified neuron.
        """
        try:
            return self.cache_file[f"neuron_{neuron_index}"][:]
        except KeyError:
            raise KeyError(f"Neuron index {neuron_index} not found in cache file")


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
