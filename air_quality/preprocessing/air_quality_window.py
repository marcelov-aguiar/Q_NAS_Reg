import numpy as np

class AirQualityWindow(object):
    
    @staticmethod
    def gen_sequence(id_df, seq_length, horizon, seq_cols):
        """
        Generate input sequences considering forecast horizon.
        """
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]

        max_t = num_elements - horizon

        for start in range(0, max_t - seq_length + 1):
            stop = start + seq_length
            yield data_matrix[start:stop, :]

    @staticmethod
    def gen_labels(id_df, seq_length, horizon, label):
        """
        Generate labels for multi-step forecasting.

        Parameters
        ----------
        id_df : pandas.DataFrame
            DataFrame for a single campaign / id.
        seq_length : int
            Input sequence length.
        horizon : int
            Forecast horizon.
        label : list[str]
            Target column name.

        Returns
        -------
        numpy.ndarray
            Shape: [samples, horizon]
        """
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]

        labels = []

        for start in range(seq_length, num_elements - horizon + 1):
            labels.append(data_matrix[start:start + horizon, 0])

        return np.array(labels)

    def seq_generation(
        self,
        train_FD_norm,
        cols_sensors,
        sequence_length,
        horizon,
        campaign_name
    ):
        seq_gen = (
            list(
                AirQualityWindow.gen_sequence(
                    train_FD_norm[train_FD_norm[campaign_name] == id],
                    sequence_length,
                    horizon,
                    cols_sensors
                )
            )
            for id in train_FD_norm[campaign_name].unique()
        )
    
        seq_array_train = np.concatenate(list(seq_gen)).astype(np.float32)
    
        return seq_array_train

    def label_generation(
        self,
        train_FD_norm,
        sequence_length,
        horizon,
        campaign_name,
        target_name
    ):
        label_gen = [
            AirQualityWindow.gen_labels(
                train_FD_norm[train_FD_norm[campaign_name] == id],
                sequence_length,
                horizon,
                [target_name]
            )
            for id in train_FD_norm[campaign_name].unique()
        ]

        label_array_train = np.concatenate(label_gen).astype(np.float32)

        return label_array_train

    def networkinput_generation(self, seq_array_train, stride, n_window, window_length):
        '''
        :param numpy array of sequence (sliced time series)
        :return: numpy array of network input for training
        '''
        # for each sensor: reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
        train_FD_sensor = []

        as_strided = np.lib.stride_tricks.as_strided

        for s_i in range(seq_array_train.shape[2]):
            window_list = []
            window_array = np.array([])

            for seq in range(seq_array_train.shape[0]):
                S = stride
                s0 = seq_array_train[seq, :, s_i].strides
                seq_sensor = as_strided(seq_array_train[seq, :, s_i], (n_window, window_length),
                                        strides=(S * s0[0], s0[0]))
                #         print (seq_sensor)
                #         window_array = np.concatenate((window_array, seq_sensor), axis=1)
                window_list.append(seq_sensor)

            window_array = np.stack(window_list, axis=0)
            window_array = np.reshape(window_array,
                                      (window_array.shape[0], window_array.shape[1], window_array.shape[2], 1))
            # print(window_array.shape)
            train_FD_sensor.append(window_array)

        return train_FD_sensor