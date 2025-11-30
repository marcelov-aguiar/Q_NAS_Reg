from femto.preprocessing.femto_preprocessing import FemtoPrep
from femto.preprocessing.femto_window import FemtoWindow


class FemtoNetworkLauncher(object):
    def __init__(self):
        pass

    def process_data(self, data_path, cols_to_drop, cols_non_sensor, scaler_path: str, piecewise_lin_ref=None):
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
        df, cols_sensors = FemtoPrep().df_preprocessing(df, cols_non_sensor, scaler_path)
        
        return df, cols_sensors

    def input_generator(self, df, cols_non_sensor, sequence_length=30, stride=1, window_length=3):
        """
        Gera o input para o modelo Multi-Head.
        Args:
            cols_non_sensor: Colunas que NÃO são features (usado para identificar o X).
        """
        n_window = int((sequence_length - window_length) / stride + 1)
        
        # 1. Gera X (Features janeladas)
        # O FemtoWindow usa cols_non_sensor para saber o que é Feature
        seq_array = FemtoWindow().seq_generation(df, cols_non_sensor, sequence_length)
        
        # 2. Formata para Multi-Head (List of Arrays para cada sensor)
        # Shape final: Lista de [Samples, 1, Window, 1] ou similar dependendo da arquitetura
        network_input = FemtoWindow().networkinput_generation(seq_array, stride, n_window, window_length)
        
        # 3. Gera Y (Labels janelados)
        network_label = FemtoWindow().label_generation(df, sequence_length)
        
        return network_input, network_label
