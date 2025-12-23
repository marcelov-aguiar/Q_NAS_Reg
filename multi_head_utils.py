from torch.utils.data import Dataset
from typing import Dict, List
from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    def __init__(self, params: dict, info: dict = {}):
        self.params = params
        self.info_dict = info

    @abstractmethod
    def get_train_dataset_info(self) -> Dict:
        """Retorna informações do dataset de treino"""
        pass
    
    @abstractmethod
    def get_train_val_test_dataset(self,
                                   individual: List[str]
                                 ) -> Dict:
        """Retorna dataset de treino, validação e teste"""
        pass
 

class CustomDatasetMultiHead(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (list of np.array): List with one array per sensor, where each array has shape (num_samples, num_windows, window_size, 1)
            y (np.array): Array of labels with shape (num_samples,)
        """
        self.X = X  # List of 14 arrays (one per sensor)
        self.y = y  # Label array, same length as number of samples (e.g., 15807)

        # Sanity check: ensure all sensors have the same number of samples
        num_samples_per_sensor = [x.shape[0] for x in X]
        assert all(n == num_samples_per_sensor[0] for n in num_samples_per_sensor), "All sensors must have the same number of samples."

    def __len__(self):
        # Number of samples (e.g., 15807)
        return self.X[0].shape[0]

    def __getitem__(self, idx):
        # For the given idx, retrieve the idx-th sample from each sensor
        sample_x = [sensor[idx] for sensor in self.X] # List of tensors: [sensor1_sample, sensor2_sample, ..., sensor14_sample]
        sample_y = self.y[idx]
        return sample_x, sample_y
