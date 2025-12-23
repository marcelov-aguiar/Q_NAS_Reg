from typing import List
import pandas as pd
from sklearn import preprocessing
import joblib

from femto.preprocessing.femto_window import FemtoWindow


class FemtoNetworkLauncher(object):
    def __init__(self):
        pass

    def process_data_train(self,
                     data_path,
                     cols_to_drop,
                     cols_non_sensor,
                     input_features_scaler_path: str,
                     piecewise_lin_ref=None,
                     train=True):
        """
        Orquestra o carregamento, limpeza e normalização.
        Args:
            cols_to_drop: Lista de colunas para jogar fora (metadados).
            cols_non_sensor: Lista de colunas para manter mas não normalizar (ID, RUL).
        """
        # 1. Carregar
        df = FemtoPrep().load_data(data_path)
        
        # 2. Preparar (Clip RUL e Remove Metadados)
        df = FemtoPrep().df_preparation(df, cols_to_drop, piecewise_lin_ref)
        
        # 3. Normalizar (Apenas Features)
        df, cols_sensors = FemtoPrep().df_preprocessing(df, cols_non_sensor, input_features_scaler_path, None, train)
        
        return df, cols_sensors

    def opt_network_input_generator(self, df, cols_sensors, sequence_length=30, stride=1, window_length=3):
        """
        Gera o input para o modelo Multi-Head.
        Args:
            cols_sensors: Colunas que são features (usado para identificar o X).
        """
        n_window = int((sequence_length - window_length) / stride + 1)
        
        # 1. Gera X (Features janeladas)
        # O FemtoWindow usa cols_sensors para saber o que é Feature
        seq_array = FemtoWindow().seq_generation(df, cols_sensors, sequence_length)
        
        # 2. Formata para Multi-Head (List of Arrays para cada sensor)
        # Shape final: Lista de [Samples, 1, Window, 1] ou similar dependendo da arquitetura
        network_input = FemtoWindow().networkinput_generation(seq_array, stride, n_window, window_length)
        
        # 3. Gera Y (Labels janelados)
        network_label = FemtoWindow().label_generation(df, sequence_length)
        
        return network_input, network_label


class FemtoPrep(object):
    def __init__(self):
        pass

    def load_data(self, data_path):
        '''
        Lê o arquivo Parquet concatenado.
        '''
        df = pd.read_parquet(data_path)
        return df

    def df_preparation(self, df, cols_to_drop, piecewise_lin_ref=None):
        '''
        - Cria coluna RUL a partir da rul_seconds limitando ao valor piecewise_lin_ref
        - Limpa o dataset, mantendo apenas Features, ID e RUL.
        '''
        # 1. Garantir tipo do ID
        if 'bearing_id' in df.columns:
            df['bearing_id'] = df['bearing_id'].astype(int)

        # 2. Criar a coluna alvo 'RUL' baseada em 'rul_seconds'
        # Aplica o Piecewise Linear (Clip no teto) se solicitado
        if piecewise_lin_ref is not None:
            df['RUL'] = df['rul_seconds'].clip(upper=piecewise_lin_ref)
        else:
            df['RUL'] = df['rul_seconds']

        # 3. Remover colunas de metadados (Lixo)
        # Remove apenas as que existem no DF para evitar erros de KeyError
        existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing_cols_to_drop)
        
        return df

    def df_preprocessing(self,
                         df: pd.DataFrame,
                         cols_non_sensor,
                         input_features_scaler_path: str,
                         cols_sensors: List[str] = None,
                         train=True):
        '''
        Normaliza apenas as colunas de features.
        Args:
            cols_non_sensor: Colunas estruturais (ex: ['bearing_id', 'RUL']) que NÃO devem ser normalizadas.
        '''
        min_max_scaler = preprocessing.MinMaxScaler()
        
        if train:
            # Fit e Transform no Treino
            cols_sensors = sorted(list(df.columns.difference(cols_non_sensor)))
            norm_values = min_max_scaler.fit_transform(df[cols_sensors])
            norm_df = pd.DataFrame(norm_values, columns=cols_sensors, index=df.index)
            joblib.dump(min_max_scaler, input_features_scaler_path)
        else:
            # Apenas Transform no Teste/Validação
            min_max_scaler = joblib.load(input_features_scaler_path)
            norm_values = min_max_scaler.transform(df[cols_sensors])
            norm_df = pd.DataFrame(norm_values, columns=cols_sensors, index=df.index)
            
        # Reconstrói o DataFrame: Junta Estruturais (intocadas) + Features (normalizadas)
        # Garante que cols_non_sensor existam no df atual
        valid_non_sensor = [c for c in cols_non_sensor if c in df.columns]
        join_df = df[valid_non_sensor].join(norm_df)
        
        # Reordena colunas (opcional, mas bom para consistência)
        # df = join_df.reindex(columns=df.columns) # Pode falhar se removemos colunas, melhor usar o join_df direto
        
        return join_df, cols_sensors
