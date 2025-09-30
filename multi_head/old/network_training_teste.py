import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding


class network_train:
    @staticmethod
    def conv2d_block(x, filters, kernel_size, strides):
        """Applies two depthwise separable Conv2D layers with BatchNorm and ReLU."""
        for _ in range(2):
            x = layers.SeparableConv2D(filters=filters,
                                       kernel_size=(1, kernel_size),
                                       strides=(1, strides),
                                       padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        return x

    @staticmethod
    def TD_CNNBranch(n_filters, window_length, n_window, n_channel, strides_len, kernel_size, n_conv_layer):
        '''
        Defining time distributed cnn layers for each input branch of multi-head network
        :param : please refer to the description of arguments in utils.py
        :return: cnn architecture with following parameter settings.
        '''
        cnn = Sequential()
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same'),
                                input_shape=(n_window, window_length, n_channel)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))

        if n_conv_layer == 1:
            pass

        else:
            for loop in range(n_conv_layer-1):
                cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same')))
                cnn.add(TimeDistributed(BatchNormalization()))
                cnn.add(TimeDistributed(Activation('relu')))

        cnn.add(TimeDistributed(Flatten()))
        # print(cnn.summary())

        return cnn

    @staticmethod
    def multi_head_cnn(sensor_input_model, n_filters, window_length, n_window,
                       n_channel, strides_len, kernel_size, n_conv_layer):
        '''
        Defining multi-head CNN network
        :param : please refer to the description of arguments in utils.py
        :return: multi-head CNN network  architecture with following parameter settings.
        '''

        cnn_out_list = []
        cnn_branch_list = []

        for sensor_input in sensor_input_model:
            cnn_branch_temp = network_train.TD_CNNBranch(n_filters, window_length, n_window,
                                                         n_channel, strides_len, kernel_size, n_conv_layer)
            cnn_out_temp = cnn_branch_temp(sensor_input)

            cnn_branch_list.append(cnn_branch_temp)
            cnn_out_list.append(cnn_out_temp)

        return cnn_out_list, cnn_branch_list

    @staticmethod
    def sensors_input_tensor(cols_sensors, n_window, window_length, n_channel):
        """Creates one input tensor per sensor."""
        return [layers.Input(shape=(n_window, window_length, n_channel), name=name)
                for name in cols_sensors]


if __name__ == "__main__":
    # Reprodutibilidade
    np.random.seed(42)
    tf.random.set_seed(42)

    # Parâmetros
    cols_sensors = ["s1", "s2", "s3"]
    batch_size = 2
    n_window = 4
    window_length = 2
    n_channel = 1
    n_filters = 3
    kernel_size = 2
    strides_len = 1
    n_conv_layer = 1
    LSTM1_units = 8
    LSTM2_units = 4
    n_outputs = 1
    bidirec = False

    # Criar dados de entrada fixos
    input_data_dict = []
    for sensor_name in cols_sensors:
        data = np.random.randn(batch_size, n_window, window_length, n_channel).astype(np.float32)
        input_data_dict.append(data)

    # Criar input tensors
    sensor_input_model = network_train.sensors_input_tensor(cols_sensors, n_window, window_length, n_channel)

    # Multi-Head CNN
    cnn_out_list, cnn_branch_list = network_train.multi_head_cnn(sensor_input_model, n_filters,
                                                                  window_length, n_window, n_channel,
                                                                  strides_len, kernel_size, n_conv_layer)

    # Concatenar saídas das CNNs
    x = tf.keras.layers.Concatenate()(cnn_out_list)

    # # LSTM stack
    # if bidirec:
    #     x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=LSTM1_units, return_sequences=True))(x)
    #     x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=LSTM2_units, return_sequences=False))(x)
    # else:
    #     x = tf.keras.layers.LSTM(units=LSTM1_units, return_sequences=True)(x)
    #     x = tf.keras.layers.LSTM(units=LSTM2_units, return_sequences=False)(x)

    # x = tf.keras.layers.Dropout(0.0)(x)  # Desligar dropout para teste determinístico
    main_output = tf.keras.layers.Dense(n_outputs, activation='linear', name='main_output')(x)

    # Criar modelo final
    model = tf.keras.Model(inputs=sensor_input_model, outputs=main_output)

    # Atribuir pesos fixos (mesmo valor para todas as camadas)
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            new_weights = [np.ones_like(w) * 0.5 for w in weights]
            layer.set_weights(new_weights)

    # Rodar predição com entrada fixa
    output = model.predict(input_data_dict)

    # Mostrar saída
    print("Saída do modelo TensorFlow com pesos fixos e entrada fixa:")
    print(output)
