import numpy as np

class FemtoWindow(object):
    
    @staticmethod
    def gen_sequence(id_df, seq_length, seq_cols):
        """ 
        Gera janelas deslizantes (Features).
        CORREÇÃO: Adicionado +1 no range para incluir a última janela possível (Current Step).
        """
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        
        # Antes: range(0, num_elements - seq_length) -> Parava antes do fim
        # Agora: range(0, num_elements - seq_length + 1) -> Vai até o fim
        for start in range(0, num_elements - seq_length + 1):
            stop = start + seq_length
            yield data_matrix[start:stop, :]

    @staticmethod
    def gen_labels(id_df, seq_length, label):
        """ 
        Gera labels alinhados.
        CORREÇÃO: Alinha o label com o ÚLTIMO passo da janela, não com o próximo.
        """
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
        
        # Antes: data_matrix[seq_length:num_elements, :] -> Pegava o próximo (t+1)
        # Agora: data_matrix[seq_length - 1:num_elements, :] -> Pega o atual (t)
        return data_matrix[seq_length - 1:num_elements, :]

    def seq_generation(self, df, cols_sensors, sequence_length):
        sequence_cols = cols_sensors.copy()
        
        seq_gen = (list(FemtoWindow.gen_sequence(df[df['bearing_id'] == id], sequence_length, sequence_cols))
                   for id in df['bearing_id'].unique())

        # clean_seq_gen = [x for x in seq_gen if len(x) > 0]
        # 
        # if not clean_seq_gen:
        #      # Retorna array vazio com shape correto para evitar erro de concat
        #      return np.empty((0, sequence_length, len(sequence_cols)), dtype=np.float32)
             
        seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
        return seq_array

    def label_generation(self, df, sequence_length):
        label_gen = [FemtoWindow.gen_labels(df[df['bearing_id'] == id], sequence_length, ['RUL'])
                     for id in df['bearing_id'].unique()]
        
        # clean_label_gen = [x for x in label_gen if len(x) > 0]
        # 
        # if not clean_label_gen:
        #     return np.empty((0, 1), dtype=np.float32)

        label_array = np.concatenate(label_gen).astype(np.float32)
        return label_array

    def networkinput_generation(self, seq_array, stride, n_window, window_length):
        '''
        Transforma (Samples, TimeSteps, Features) em (Samples, Heads, Window, 1).
        Essa lógica é perfeita para o seu Multi-Head (Uma feature = Uma Head).
        '''
        # seq_array shape: [num_samples, sequence_length, num_features]
        # num_features será 12 (no seu caso)
        
        train_FD_sensor = []
        as_strided = np.lib.stride_tricks.as_strided

        # Itera sobre as features (cada feature vira um input para uma Head)
        for s_i in range(seq_array.shape[2]):
            window_list = []
            
            for seq in range(seq_array.shape[0]):
                S = stride
                s0 = seq_array[seq, :, s_i].strides
                
                # Cria sub-janelas (se aplicável) para a CNN
                seq_sensor = as_strided(seq_array[seq, :, s_i], (n_window, window_length),
                                        strides=(S * s0[0], s0[0]))
                window_list.append(seq_sensor)

            window_array = np.stack(window_list, axis=0)
            # Reshape para (Samples, SubWindows, WindowLen, 1) -> Formato de Imagem 1D
            window_array = np.reshape(window_array,
                                      (window_array.shape[0], window_array.shape[1], window_array.shape[2], 1))
            train_FD_sensor.append(window_array)

        # Retorna uma LISTA de arrays. Cada item da lista é o input de uma Head.
        return train_FD_sensor