import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing

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
        Limpa o dataset, mantendo apenas Features, ID e RUL.
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
                         scaler_path: str):
        '''
        Normaliza apenas as colunas de features.
        Args:
            cols_non_sensor: Colunas estruturais (ex: ['bearing_id', 'RUL']) que NÃO devem ser normalizadas.
        '''
        # Identifica colunas de features (Tudo que sobra no DF menos as estruturais)
        cols_normalize = sorted(list(df.columns.difference(cols_non_sensor)))
        
        # min_max_scaler = preprocessing.MinMaxScaler()
        
        #if train:
        #    # Fit e Transform no Treino
        #    norm_values = min_max_scaler.fit_transform(df[cols_normalize])
        #    norm_df = pd.DataFrame(norm_values, columns=cols_normalize, index=df.index)
        #    joblib.dump(min_max_scaler, scaler_filename)
        #else:
        # Apenas Transform no Teste/Validação
        min_max_scaler = joblib.load(scaler_path)
        norm_values = min_max_scaler.transform(df[cols_normalize])
        norm_df = pd.DataFrame(norm_values, columns=cols_normalize, index=df.index)
            
        # Reconstrói o DataFrame: Junta Estruturais (intocadas) + Features (normalizadas)
        # Garante que cols_non_sensor existam no df atual
        valid_non_sensor = [c for c in cols_non_sensor if c in df.columns]
        join_df = df[valid_non_sensor].join(norm_df)
        
        # Reordena colunas (opcional, mas bom para consistência)
        # df = join_df.reindex(columns=df.columns) # Pode falhar se removemos colunas, melhor usar o join_df direto
        
        return join_df, cols_normalize
