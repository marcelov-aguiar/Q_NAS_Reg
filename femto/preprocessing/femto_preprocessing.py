from typing import List
import os
import pandas as pd
import numpy as np
import joblib

import joblib
import importlib
import torch
from sklearn.model_selection import train_test_split

from femto.preprocessing.femto_utils import FemtoNetworkLauncher, FemtoPrep
from femto.preprocessing import femto_const
from multi_head_utils import BaseDataLoader, CustomDatasetMultiHead


class FemtoMultiHeadDataLoader(BaseDataLoader):
    def __init__(self, params: dict, info: dict = {}):
        super().__init__(params, info)
        self.processed_train_path = \
          os.path.join(self.params['data_path'], 'temp', f"{self.params['dataset']}_temp_{self.params['exp']}.parquet")
        self.processed_test_path = \
          os.path.join(self.params['data_path'], 'temp', f"{self.params['extra_params']['dataset_test']}_temp_{self.params['exp']}.parquet")

    def get_train_dataset_info(self) -> dict:
        # Caminho para o arquivo final_train.parquet
        data_path = os.path.join(self.params['data_path'],
                                 f"{self.params['dataset']}.{self.params['file_extension']}")
        # Leitura rápida só para pegar colunas
        data = pd.read_parquet(data_path)
        
        cols_non_sensor = self.params['extra_params']['cols_non_sensor']
        cols_to_drop = self.params['extra_params']['cols_to_drop']

        # Filtra para garantir que existem no arquivo (segurança)
        existing_drop = [c for c in cols_to_drop if c in data.columns]
        existing_non_sensor = [c for c in cols_non_sensor if c in data.columns]
        
        # Num sensors = Total - (Metadados Excluídos) - (Estruturais Mantidas)
        self.info_dict['num_sensors'] = len(data.columns) - len(existing_drop) - len(existing_non_sensor)
        
        # Params task
        self.info_dict['shape'] = [None, None, None]
        self.info_dict['num_classes'] = self.params["num_classes"]
        self.info_dict['task'] = self.params["task"]
        return self.info_dict

    def get_train_val_test_dataset(self, individual: List[str]):
        """
        Carrega os parquets processados e aplica apenas a Janela Deslizante.
        """
        window_length = self._get_windows_length_from_individual(individual)

        cols_non_sensor = self.params['extra_params']['cols_non_sensor']

        df_train_raw = pd.read_parquet(self.processed_train_path)

        cols_sensors = [c for c in df_train_raw.columns if c not in cols_non_sensor]

        launcher = FemtoNetworkLauncher()

        if self.params["extra_params"]["type_validation"] == "split_train_val_bearing":
          X_train, X_val, y_train, y_val = \
            self._split_train_val_bearing(df_train_raw, launcher, cols_sensors, window_length)
        elif self.params["extra_params"]["type_validation"] == "split_train_val_temporal":
          X_train, X_val, y_train, y_val = \
            self._split_train_val_temporal(df_train_raw, launcher, cols_sensors, window_length, val_ratio=self.params["extra_params"]["val_ratio"])
        elif self.params["extra_params"]["type_validation"] == "split_train_val_shuffle":
          X_train, X_val, y_train, y_val = \
            self._split_train_val_shuffle(df_train_raw, launcher, cols_sensors, window_length, val_ratio=self.params["extra_params"]["val_ratio"])
        else:
           raise ValueError("Tipo de validação desconhecido")

        X_train = [torch.tensor(s, dtype=torch.float32) for s in X_train]
        X_val = [torch.tensor(s, dtype=torch.float32) for s in X_val]
        
        train_dataset = CustomDatasetMultiHead(X_train, torch.tensor(y_train, dtype=torch.float32))
        valid_dataset = CustomDatasetMultiHead(X_val, torch.tensor(y_val, dtype=torch.float32))

        # --- TESTE ---
        df_test_raw = pd.read_parquet(self.processed_test_path)
        
        X_test_np, y_test_np = launcher.opt_network_input_generator(
            df_test_raw,
            cols_sensors=cols_sensors,
            sequence_length=self.params["extra_params"]["sequence_length"],
            stride=self.params["extra_params"]["stride"],
            window_length=window_length
        )
        
        X_test = [torch.tensor(s, dtype=torch.float32) for s in X_test_np]
        test_dataset = CustomDatasetMultiHead(X_test, torch.tensor(y_test_np, dtype=torch.float32))

        return train_dataset, valid_dataset, test_dataset

    def get_train_val_test_dataset_old(self, individual: List[str]):
        # Carrega hiperparâmetros do indivíduo (Q-NAS) ou fixos
        # No Q-NAS o tamanho da janela (kernel) define o 'window_length' da CNN
        window_length = self._get_windows_length_from_individual(individual)
        cols_to_drop = self.params['extra_params']['cols_to_drop']
        cols_non_sensor = self.params['extra_params']['cols_non_sensor']

        # --- TREINO e VALIDAÇÃO ---
        train_path = os.path.join(self.params['data_path'],
                                 f"{self.params['dataset']}.{self.params['file_extension']}")
        
        # Chama o Launcher
        launcher = FemtoNetworkLauncher()
        df_train_raw, cols_sensors = launcher.process_data_train(
           train_path,
           cols_to_drop=cols_to_drop,
           cols_non_sensor=cols_non_sensor,
           scaler_path=os.path.join(self.params['data_path'], self.params['extra_params']["scaler_name"]),
           piecewise_lin_ref=self.params["extra_params"]["piecewise_lin_ref"],
           train=True
        )
        
        df_train_raw = self.save_target_scaler(df_train_raw)
        
        if self.params["extra_params"]["type_validation"] == "split_train_val_bearing":
          # split baseado no bearing inteiro para validação
          X_train, X_val, y_train, y_val = \
            self._split_train_val_bearing(df_train_raw,
                                          launcher,
                                          cols_sensors,
                                          window_length)
        elif self.params["extra_params"]["type_validation"] == "split_train_val_temporal":
          # Split Treino/Validação (Mantendo ordem temporal independente do bearing)
          X_train, X_val, y_train, y_val = self._split_train_val_temporal(df_train_raw,
                                          launcher,
                                          cols_sensors,
                                          window_length,
                                          val_ratio=0.1)
        elif self.params["extra_params"]["type_validation"] == "split_train_val_shuffle":
          # Split Treino/Validação com shuffle igual a True (apos criacao das janelas)
          X_train, X_val, y_train, y_val = self._split_train_val_shuffle(df_train_raw,
                                          launcher,
                                          cols_sensors,
                                          window_length,
                                          val_ratio=0.1)
        else:
           X_train, X_val, y_train, y_val = None, None, None, None
           raise

        # Converte para Tensores
        X_train = [torch.tensor(s, dtype=torch.float32) for s in X_train]
        X_val = [torch.tensor(s, dtype=torch.float32) for s in X_val]
        
        train_dataset = CustomDatasetMultiHead(X_train, torch.tensor(y_train, dtype=torch.float32))
        valid_dataset = CustomDatasetMultiHead(X_val, torch.tensor(y_val, dtype=torch.float32))

        # --- TESTE ---
        test_path = os.path.join(self.params['data_path'],
                                 f"{self.params['extra_params']['dataset_test']}.{self.params['extra_params']['file_extension_test']}")

        df_test_raw = self._process_data_test(data_path=test_path,
                                              sequence_length=self.params["extra_params"]["sequence_length"],
                                              cols_to_drop=cols_to_drop,
                                              cols_non_sensor=cols_non_sensor,
                                              cols_sensors=cols_sensors)
        
        df_test_raw = self._norm_target_test(df_test_raw)

        X_test_np, y_test_np = launcher.opt_network_input_generator(
            df_test_raw,
            cols_sensors=cols_sensors,
            sequence_length=self.params["extra_params"]["sequence_length"],
            stride=self.params["extra_params"]["stride"],
            window_length=window_length
        )
        
        X_test = [torch.tensor(s, dtype=torch.float32) for s in X_test_np]
        test_dataset = CustomDatasetMultiHead(X_test, torch.tensor(y_test_np, dtype=torch.float32))

        return train_dataset, valid_dataset, test_dataset

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

    def _split_train_val_bearing(self,
                                 df_train_raw,
                                 launcher: FemtoNetworkLauncher,
                                 cols_sensors,
                                 window_length):
      val_bearing_ids = self.params["extra_params"]["validation_bearing_id"]
      mask_val = df_train_raw['bearing_id'].isin(val_bearing_ids)
          
      df_val = df_train_raw[mask_val].copy()
      df_train = df_train_raw[~mask_val].copy()
       
      X_train, y_train = launcher.opt_network_input_generator(
          df_train,
          cols_sensors=cols_sensors,
          sequence_length=self.params["extra_params"]["sequence_length"],
          stride=self.params["extra_params"]["stride"],
          window_length=window_length
      )

      X_val, y_val = launcher.opt_network_input_generator(
              df_val,
              cols_sensors=cols_sensors,
              sequence_length=self.params["extra_params"]["sequence_length"],
              stride=self.params["extra_params"]["stride"],
              window_length=window_length
      )

      return X_train, X_val, y_train, y_val

    def _split_train_val_shuffle(self, df_train_raw,
                            launcher: FemtoNetworkLauncher,
                            cols_sensors,
                            window_length,
                            val_ratio=0.3):
      """
      Faz a divisão com shuffle igual a True após criação das janelas
      """
      X_full, y_full = launcher.input_generator(
            df_train_raw,
            cols_sensors=cols_sensors,
            sequence_length=self.params["extra_params"]["sequence_length"],
            stride=self.params["extra_params"]["stride"],
            window_length=window_length
        )

      num_samples = y_full.shape[0]
      indices = np.arange(num_samples)

      # 2. Faz o split 70/30 nos ÍNDICES (shuffle=True é o padrão, mas deixei explícito)
      train_idx, val_idx = train_test_split(indices, test_size=val_ratio, shuffle=True)

      # 3. Aplica os índices aos dados
      # Como X_full é uma lista de arrays (um por sensor/head), precisamos iterar sobre ela
      X_train = [x[train_idx] for x in X_full]
      X_val   = [x[val_idx]   for x in X_full]

      y_train = y_full[train_idx]
      y_val   = y_full[val_idx]
      return X_train, X_val, y_train, y_val

    def _split_train_val_temporal(self,
                            df_train_raw,
                            launcher: FemtoNetworkLauncher,
                            cols_sensors,
                            window_length,
                            val_ratio=0.3):
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

        # split de forma temporal
        inputs, targets = launcher.opt_network_input_generator(
            df_train_raw,
            cols_sensors=cols_sensors,
            sequence_length=self.params["extra_params"]["sequence_length"],
            stride=self.params["extra_params"]["stride"],
            window_length=window_length
        )
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

    def _filter_benchmark_windows(self, df_raw: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
        """
        Filtra o DataFrame de teste para manter APENAS os dados necessários para criar
        a última janela de cada rolamento (terminando exatamente no RPT).
        """
        filtered_chunks = []
        
        # Garante que bearing_id é string para bater com as chaves do dicionário
        df_raw['bearing_id'] = df_raw['bearing_id'].astype(str)
        
        for bearing_id, rpt_idx in femto_const.RPT_INDICES.items():
            # Seleciona dados do rolamento atual
            # Nota: bearing_id no df pode ser "1_1" ou "11", ajuste conforme seu dado
            bearing_mask = df_raw['bearing_id'] == bearing_id
            
            if not bearing_mask.any():
                continue
                
            # Define o intervalo exato da janela: [RPT - seq_len + 1, RPT]
            # Ex: Se seq=30 e RPT=1801, pegamos do 1772 ao 1801.
            start_idx = rpt_idx - sequence_length + 1
            end_idx = rpt_idx
            
            # Filtra pelos índices
            idx_mask = (df_raw['sample_idx'] >= start_idx) & (df_raw['sample_idx'] <= end_idx)
            
            chunk = df_raw[bearing_mask & idx_mask].copy()
            
            # Validação: Só adiciona se tivermos a janela completa
            if len(chunk) >= sequence_length:
                filtered_chunks.append(chunk)
            else:
                print(f"AVISO: Bearing {bearing_id} tem dados insuficientes para janela de tamanho {sequence_length} no RPT {rpt_idx}.")

        if not filtered_chunks:
            raise ValueError("Nenhum dado de teste restou após a filtragem de Benchmark!")
            
        return pd.concat(filtered_chunks, ignore_index=True)

    def prepare_dataset_once(self):
        """
        Deve ser chamado na thread principal antes do início da evolução.
        """
        cols_to_drop = self.params['extra_params']['cols_to_drop']
        cols_non_sensor = self.params['extra_params']['cols_non_sensor']

        train_path = os.path.join(self.params['data_path'], f"{self.params['dataset']}.{self.params['file_extension']}")
        test_path = os.path.join(self.params['data_path'],
                                 f"{self.params['extra_params']['dataset_test']}.{self.params['extra_params']['file_extension_test']}")

        scaler_name = self.params['extra_params']["scaler_name"].replace('.save', '_'+self.params['exp']+'.save')
        input_features_scaler_path = os.path.join(self.params['data_path'], 'scaler', scaler_name)
        
        target_scaler_name = self.params['target_normalization']['path'].replace('.save', '_'+self.params['exp']+'.save')
        target_scaler_path = os.path.join(self.params['data_path'], 'scaler', target_scaler_name)


        launcher = FemtoNetworkLauncher()
        
        # 1. Processa Treino (Features + Fit Scaler Input)
        df_train_raw, cols_sensors = launcher.process_data_train(
           train_path,
           cols_to_drop=cols_to_drop,
           cols_non_sensor=cols_non_sensor,
           input_features_scaler_path=input_features_scaler_path,
           piecewise_lin_ref=self.params["extra_params"]["piecewise_lin_ref"],
           train=True
        )
        
        # 2. Fit e Save do Target Scaler (RUL)
        df_train_raw = self.save_target_scaler(df_train_raw, target_scaler_path)

        # 3. Salva dataset intermediário de treino
        df_train_raw.to_parquet(self.processed_train_path, index=False)

        # 4. Processa dados de Teste
        df_test_raw = self._process_data_test(
            data_path=test_path,
            sequence_length=self.params["extra_params"]["sequence_length"],
            cols_to_drop=cols_to_drop,
            cols_non_sensor=cols_non_sensor,
            cols_sensors=cols_sensors,
            input_features_scaler_path=input_features_scaler_path
        )
        
        df_test_raw = self._norm_target_test(df_test_raw, target_scaler_path)
        
        # 5. Salva dataset intermediário de teste
        df_test_raw.to_parquet(self.processed_test_path, index=False)


    def _process_data_test(self,
                           data_path: str,
                           sequence_length: int,
                           cols_to_drop: List[str],
                           cols_non_sensor: List[str],
                           cols_sensors: List[str],
                           input_features_scaler_path):
       # 1. Carregar DataFrame Bruto (ainda tem sample_idx)
        df_test_full = FemtoPrep().load_data(data_path)
        
        # 2. FILTRAGEM DE BENCHMARK
        df_test_bench = self._filter_benchmark_windows(df_test_full, sequence_length)
        
        # 3. Preparar (Clip RUL e Remove Metadados)
        df_test_clean = FemtoPrep().df_preparation(
            df_test_bench, 
            cols_to_drop, 
            piecewise_lin_ref=self.params["extra_params"]["piecewise_lin_ref"]
        )
        
        # 4. Normalizar (Apenas Features, usando o scaler salvo)
        df_test_norm, _ = FemtoPrep().df_preprocessing(
            df_test_clean, 
            cols_non_sensor, 
            input_features_scaler_path=input_features_scaler_path,
            cols_sensors=cols_sensors,
            train=False
        )

        return df_test_norm

    def save_target_scaler(self, df_data, target_scaler_path) -> pd.DataFrame:

      if self.params['target_normalization']["name"] is not None:
        scaler_config = self.params['target_normalization']
        full_class_path = scaler_config["name"]  # Ex: "sklearn.preprocessing.MinMaxScaler"

        # 1. Separa o caminho do módulo da classe
        # "sklearn.preprocessing" | "MinMaxScaler"
        module_name, class_name = full_class_path.rsplit(".", 1)

        # 2. Importação Dinâmica
        try:
            module = importlib.import_module(module_name)
            scaler_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Não foi possível importar {class_name} de {module_name}. Erro: {e}")

        params_dict = scaler_config.get("params") or {}
        if params_dict == 'None':
           params_dict = {}
        scaler = scaler_cls(**params_dict)

        df_data["RUL"] = \
          scaler.fit_transform(df_data["RUL"].values.reshape(-1, 1)).flatten()

        joblib.dump(scaler, target_scaler_path)
        print(f"Target Scaler ({class_name}) salvo com sucesso em: {target_scaler_path}")
      return df_data

    def _norm_target_test(self, df_test: pd.DataFrame, target_scaler_path: str) -> pd.DataFrame:
      if self.params['target_normalization']["name"] is not None:
        target_scaler = joblib.load(target_scaler_path)
        df_test["RUL"] = \
          target_scaler.transform(df_test["RUL"].values.reshape(-1, 1)).flatten()
      return df_test
