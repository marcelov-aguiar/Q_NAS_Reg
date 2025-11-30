import numpy as np

class FemtoWindow(object):
    
    @staticmethod
    def gen_sequence(id_df, seq_length, seq_cols):
        """ 
        Mesma lógica do original: fatia os dados sem padding.
        """
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        
        for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]

    @staticmethod
    def gen_labels(id_df, seq_length, label):
        """ 
        Mesma lógica do original: alinha o label com o fim da janela.
        """
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
        return data_matrix[seq_length:num_elements, :]

    def seq_generation(self, df, cols_non_sensor, sequence_length):
        '''
        Gera sequências para TODOS os rolamentos no DataFrame.
        MUDANÇA: Usa 'bearing_id' em vez de 'unit_nr'.
        '''
        sequence_cols = df.columns.difference(cols_non_sensor)
        
        # Itera sobre cada rolamento único
        # Isso garante que a janela não misture dados do Bearing1_1 com Bearing1_2
        seq_gen = (list(FemtoWindow.gen_sequence(df[df['bearing_id'] == id], sequence_length, sequence_cols))
                   for id in df['bearing_id'].unique())

        # Concatena tudo num arrayzão numpy
        # O try/except é para evitar erro se algum rolamento for menor que a janela
        clean_seq_gen = [x for x in seq_gen if len(x) > 0]
        seq_array = np.concatenate(clean_seq_gen).astype(np.float32)

        return seq_array

    def label_generation(self, df, sequence_length):
        '''
        Gera labels para TODOS os rolamentos.
        '''
        label_gen = [FemtoWindow.gen_labels(df[df['bearing_id'] == id], sequence_length, ['RUL'])
                     for id in df['bearing_id'].unique()]
        
        clean_label_gen = [x for x in label_gen if len(x) > 0]
        label_array = np.concatenate(clean_label_gen).astype(np.float32)

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