from typing import List
import os
import pandas as pd
import numpy as np
import joblib

import importlib
import torch
from sklearn.model_selection import train_test_split

from air_quality.preprocessing.air_quality_utils import AirQualityNetworkLauncher
from multi_head_utils import BaseDataLoader, CustomDatasetMultiHead


class AirQualityMultiHeadDataLoader(BaseDataLoader):
	def __init__(self, params: dict, info: dict = {}):
		super().__init__(params, info)

	def get_train_dataset_info(self) -> dict:
		# Caminho para o arquivo final_train.parquet
		data_path = os.path.join(self.params['data_path'],
								 f"{self.params['dataset']}.{self.params['file_extension']}")
		# Leitura rápida só para pegar colunas
		data = pd.read_csv(data_path)

		# Num sensors = Total - (Metadados Excluídos) - (Estruturais Mantidas)
		cols_non_sensor = self.params["extra_params"]["cols_non_sensor"]

		cols_sensors = [c for c in data.columns if c not in cols_non_sensor]
		self.info_dict['num_sensors'] = len(cols_sensors)

		# Params task
		self.info_dict['shape'] = [None, None, None]
		self.info_dict['num_classes'] = self.params["num_classes"]
		self.info_dict['task'] = self.params["task"]
		return self.info_dict

	def get_train_val_test_dataset(self, individual: List[str]):
		"""
		Carrega os parquets processados e aplica apenas a Janela Deslizante.
		"""
		data_train_path = os.path.join(self.params['data_path'],
								 f"{self.params['dataset']}.{self.params['file_extension']}")

		df_train = pd.read_csv(data_train_path)

		df_train["unit_nr"] = 10

		data_val_path = os.path.join(self.params['data_path'],
								 f"{self.params['extra_params']['dataset_val']}.{self.params['extra_params']['file_extension_val']}")
		
		df_val = pd.read_csv(data_val_path)

		df_val["unit_nr"] = 10

		data_test_path = os.path.join(self.params['data_path'],
								 f"{self.params['extra_params']['dataset_test']}.{self.params['extra_params']['file_extension_test']}")
		
		df_test = pd.read_csv(data_test_path)

		df_test["unit_nr"] = 10

		train_dataset = self.get_train_val_dataset(df_train, individual)

		valid_dataset = self.get_train_val_dataset(df_val, individual)

		test_dataset = self.get_train_val_dataset(df_test, individual)

		return train_dataset, valid_dataset, test_dataset

	def get_train_val_dataset(self, df_data: pd.DataFrame, individual):
		window_length = self._get_windows_length_from_individual(individual)

		cols_non_sensor = self.params["extra_params"]["cols_non_sensor"]

		cols_sensors = [c for c in df_data.columns if c not in cols_non_sensor]

		input_data, input_data_label = \
		  AirQualityNetworkLauncher().opt_network_input_generator(
				df=df_data,
				cols_sensors=cols_sensors,
				campaign_name=self.params["extra_params"]["campaign_name"],
				target_name=self.params["extra_params"]["target_name"],
				sequence_length=self.params["extra_params"]["sequence_length"],
				stride=self.params["extra_params"]["stride"],
				window_length=window_length,
				horizon=self.params["num_classes"])

		X_data = [sensor.astype(np.float32) for sensor in input_data]
		y_data = input_data_label.astype(np.float32)

		
		X_data = [torch.tensor(sensor, dtype=torch.float32) for sensor in X_data]
		dataset = CustomDatasetMultiHead(X_data,
	                                     torch.tensor(y_data, dtype=torch.float32))

		return dataset

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
