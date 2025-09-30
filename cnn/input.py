""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Input module - Generic data loader for PyTorch, supporting various datasets and data augmentation.

"""
import torch
import random
import auxiliar
from multi_head.utils import network_launcher
import util
import os
import numpy as np
import pandas as pd
from time import time
import torchvision.datasets
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, TrivialAugmentWide
from sklearn.model_selection import StratifiedShuffleSplit
import medmnist
from medmnist import INFO
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod

cifar10_info = {
  'dataset': 'CIFAR10',
  'mean': [0.491400808095932, 0.48215898871421814, 0.44653093814849854],
  'std': [0.24703224003314972, 0.24348513782024384, 0.26158785820007324],
  'shape': [3, 32, 32], 
  'num_classes': 10,
  'task': 'classification',
  'balanced_train': True
}

cifar100_info = {
  'dataset': 'CIFAR100',
  'mean': [0.5070757865905762, 0.48655030131340027, 0.4409191310405731],
  'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967],
  'shape': [3, 32, 32],
  'num_classes': 100,
  'task': 'classification',
  'balanced_train': True
}

atletasaxial_info = {
  'dataset': 'ATLETA_AXIAL',
  'mean': [0.485, 0.456, 0.406],
  'std': [0.229, 0.224, 0.225],
  'shape': [3, 128, 128],
  'num_classes': 3,
  'task': 'classification',
  'balanced_train': False
}

atletascoronal_info = {
  'dataset': 'ATLETA_CORONAL',
  'mean': [0.485, 0.456, 0.406],
  'std': [0.229, 0.224, 0.225],
  'shape': [3, 128, 128],
  'num_classes': 3,
  'task': 'classification',
  'balanced_train': False
}

available_datasets = {
  'cifar10': cifar10_info,
  'cifar100': cifar100_info,
  'atleta_axial': atletasaxial_info,
  'atleta_coronal': atletascoronal_info
}


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

class IdentityTransform:
    def __call__(self, x):
        return x

class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
          x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


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

class TurbofanMultiHeadDataLoader(BaseDataLoader):
  def __init__(self,
               params: dict,
               info: dict = {}):
    super().__init__(params, info)

  def get_train_dataset_info(self) -> dict:
    data = pd.read_csv(os.path.join(self.params['data_path'],
                                    f"{self.params['dataset']}.{self.params['file_extension']}"))
    #TODO: Remover os Nones. Não é possível remover pois a priori não sabemos o tamanho da janela
    # pois ela é definida pelo indivíduo.
    n_window, window_length, n_channel = None, None, None
    self.info_dict['num_sensors'] = len(data.columns) - len(self.params["extra_params"]["cols_non_sensor"])

    self.info_dict['shape'] = [n_window, window_length, n_channel]
    self.info_dict['num_classes'] = self.params["num_classes"]
    self.info_dict['task'] = self.params["task"]
    return self.info_dict

  def get_train_val_test_dataset(self,
                                 individual: List[str]) -> \
                                  Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dataset, valid_dataset = self._get_train_val_dataset(individual)
    test_dataset = self._get_test_dataset(individual)

    return train_dataset, valid_dataset, test_dataset

  def _get_test_dataset(self,
                        individual: List[str]) -> pd.DataFrame:
    data = pd.read_csv(
       os.path.join(self.params['data_path'],
                    f"{self.params['extra_params']['dataset_test']}.{self.params['extra_params']['file_extension_test']}"))
    
    RUL_FD_path = os.path.join(self.params['data_path'],
                              f"{self.params['extra_params']['RUL_FD_test_file']}")

    window_length = self._get_windows_length_from_individual(individual)

    test_input, test_input_label = \
      network_launcher().rmse_test_input_generator(
          dataframe_norm=data,
          cols_non_sensor=self.params["extra_params"]["cols_non_sensor"],
          rul_file_path=RUL_FD_path,
          sequence_length=self.params["extra_params"]["sequence_length"],
          stride=self.params["extra_params"]["stride"],
          window_length=window_length,
          piecewise_lin_ref=self.params["extra_params"]["piecewise_lin_ref"])

    X_test = [sensor.astype(np.float32) for sensor in test_input]
    y_test = test_input_label.astype(np.float32)

    X_test = [torch.tensor(sensor, dtype=torch.float32) for sensor in X_test]
    test_dataset = CustomDatasetMultiHead(X_test,
                                          torch.tensor(y_test, dtype=torch.float32))
    return test_dataset

  def _get_train_val_dataset(self,
                             individual: List[str]) -> \
                              Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the training and validation dataset.
    steps:
    - Gerar dataset csv no formato esperado do trabalho relacionado (semelhante ao QIEA)
    - Carregar o dataset csv no formato esperado do trabalho relacionado (semelhante ao QIEA)
    - Pré-processar os dados conforme necessário: tamanho da janela e sequência (receber como parâmetro), semelhante ao trabalho relacionado
    - Retornar o dataset no formato esperado do PyTorch
    Parameters
    ----------
    train : bool, optional
        _description_, by default True

    Returns
    -------
    pd.DataFrame
        _description_
    """
    data = pd.read_csv(os.path.join(self.params['data_path'],
                                    f"{self.params['dataset']}.{self.params['file_extension']}"))
    
    window_length = self._get_windows_length_from_individual(individual)

    training_input, training_input_label = \
      network_launcher().opt_network_input_generator(
         dataframe_norm=data,
         cols_non_sensor=self.params["extra_params"]["cols_non_sensor"],
         sequence_length=self.params["extra_params"]["sequence_length"],
         stride=self.params["extra_params"]["stride"],
         window_length=window_length,
         piecewise_lin_ref=self.params["extra_params"]["piecewise_lin_ref"])

    train_inputs, val_inputs, train_targets, val_targets = \
      self.__split_train_val(
         training_input,
         training_input_label,
         val_ratio=self.params["extra_params"]["val_split"])

    X_train = [sensor.astype(np.float32) for sensor in train_inputs]
    y_train = train_targets.astype(np.float32)
    X_val = [sensor.astype(np.float32) for sensor in val_inputs]
    y_val = val_targets.astype(np.float32)
      
    X_train = [torch.tensor(sensor, dtype=torch.float32) for sensor in X_train]
    train_dataset = CustomDatasetMultiHead(X_train,
                                           torch.tensor(y_train, dtype=torch.float32))
    
    X_val = [torch.tensor(sensor, dtype=torch.float32) for sensor in X_val]
    valid_dataset = CustomDatasetMultiHead(X_val,
                                           torch.tensor(y_val, dtype=torch.float32))
    return train_dataset, valid_dataset

  def _get_windows_length_from_individual(self,
                                          individual: List[str]) -> int:
    """
    Extracts the window length from the first convolutional layer
    encoded in an individual's genotype.

    This function assumes that each gene in the genotype is represented as
    a string with the format: 'conv_<kernel>_<stride>_<filters>', where:
        - <kernel>  (int): size of the convolutional kernel
        - <stride>  (int): stride of the convolution
        - <filters> (int): number of convolutional filters

    The window length is defined here as the kernel size of the *first*
    convolutional layer in the genotype. Only the first element of the list
    is parsed to extract this value.

    Parameters
    ----------
    individual : list of str
        The genotype of the individual, where each element encodes a
        convolutional layer in the format described above.
        Example: ['conv_2_1_3', 'conv_3_2_64']

    Returns
    -------
    int
        The window length, i.e., the kernel size of the first convolution.
    """
    for ind in individual:
      if ind.startswith('conv'):
          parts = ind.split('_')
          if len(parts) >= 4:
              return int(parts[1])
    return 5 # Default window length if no conv layer is found

  def __split_train_val(self,
                        inputs: List[np.ndarray],
                        targets: np.ndarray, val_ratio=0.1):
    """
    Divide os dados de entrada (por sensor) e os rótulos (contínuos) entre treino e validação,
    mantendo a ordem temporal.

    Args:
        inputs (list of np.ndarray): Lista com os dados dos sensores, cada um com shape (N, ...).
        targets (np.ndarray): Array contínuo com os rótulos, shape (total_N, 1).
        val_ratio (float): Proporção dos dados para validação (ex: 0.1 = 10%).

    Returns:
        Tuple: (train_inputs, val_inputs, train_targets, val_targets)
               train_inputs e val_inputs são listas com mesmo comprimento de `inputs`;
               train_targets e val_targets são np.ndarrays.
    """
    train_inputs, val_inputs = [], []
    train_targets_list, val_targets_list = [], []

    total_samples = 0

    for sensor_data in inputs:
        n_samples = sensor_data.shape[0]
        split_idx = int(n_samples * (1 - val_ratio))

        # Divide os dados de entrada por sensor
        train_inputs.append(sensor_data[:split_idx])
        val_inputs.append(sensor_data[split_idx:])

        # Seleciona a fatia correspondente dos rótulos globais
        train_targets_list.append(targets[total_samples:total_samples + split_idx])
        val_targets_list.append(targets[total_samples + split_idx:total_samples + n_samples])

        total_samples += n_samples

    # Concatena as listas em arrays únicos
    train_targets = np.concatenate(train_targets_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    return train_inputs, val_inputs, train_targets, val_targets

class GenericDataLoader:
  """A generic data loader for PyTorch, supporting various datasets and data augmentation."""
  def __init__(self, params: dict, seed=None):
    """
    Initialize the GenericDataLoader.
    
    Parameters:
      params (dict): Dictionary containing parameters.
      train_split (float): Split ratio for training data.
      seed (int): Seed for randomization.
      info (dict): Additional information about the dataset.
      
    Returns:
      None
    """
    self.params = params

    if seed is None:
        seed = int(time())
        random.seed(seed)
        torch.manual_seed(seed)
    self.info_dict = {'dataset': f'{self.params["dataset"]}'}
    self.info_dict['seed'] = seed

    self.num_classes = self.params["num_classes"]
    self.task = self.params["task"]

    dataloader_cls: type[BaseDataLoader] = globals()[self.params['dataloader_class']]

    dataloader: BaseDataLoader = dataloader_cls(params=self.params, info=self.info_dict)

    self.info_dict = dataloader.get_train_dataset_info()

    util.create_info_file(out_path=self.params['data_path'], info_dict=self.info_dict)

  def get_loader(self, individual=None, for_train=True, pin_memory_device="cuda"):
    """
    Get data loader for training or validation/testing.

    Parameters:
      for_train (bool): If True, returns the training and val loader; otherwise, returns the testing loader.

    Returns:
      DataLoader: PyTorch DataLoader.
    """
    # create the dataset

    dataloader_cls: type[BaseDataLoader] = globals()[self.params['dataloader_class']]

    dataloader: BaseDataLoader = dataloader_cls(params=self.params, info=self.info_dict)

    train_dataset, valid_dataset, test_dataset = \
      dataloader.get_train_val_test_dataset(individual=individual)
    
    drop_last = True if self.params['dataset'].lower() == 'organamnist' else False
        
    if not for_train:
      test_loader = DataLoader(
        test_dataset,
        batch_size=self.params['eval_batch_size'],
        num_workers=self.params['num_workers'],
        shuffle=False,
        pin_memory=True,
        pin_memory_device=pin_memory_device)
      return test_loader

    train_loader = DataLoader(
      train_dataset,
      batch_size=self.params['batch_size'],
      num_workers=self.params['num_workers'],
      shuffle= True,
      drop_last=drop_last,
      pin_memory=True, 
      pin_memory_device=pin_memory_device)

    val_loader = DataLoader(
      valid_dataset,
      batch_size=self.params['eval_batch_size'],
      num_workers=self.params['num_workers'],
      shuffle=True,
      pin_memory=True, 
      pin_memory_device=pin_memory_device)
  
    self.info_dict['train_records'] = len(train_dataset)
    self.info_dict['valid_records'] = len(valid_dataset)
    self.info_dict['test_records'] = len(test_dataset)
            
    util.create_info_file(out_path=self.params['data_path'], info_dict=self.info_dict)
    
    return train_loader, val_loader

  # def get_dataset(self, train=True) -> pd.DataFrame:
  #   train_FD = pd.read_csv(os.path.join(self.params['data_path'], f"{self.params['dataset']}.csv"))

  #   return train_FD
