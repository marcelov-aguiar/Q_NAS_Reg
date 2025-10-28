import torch
import torch.nn as nn
import numpy as np
import random
from model_teste import MultiHeadNetworkGraphNew, Conv1DBlock  # Altere para o nome real do seu módulo
# from cnn.model import MultiHeadNetworkGraph, Conv1DBlock  # Altere para o nome real do seu módulo

# ----------------------------
# Função para fixar os seeds
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# Simula entrada dummy igual à do Keras
# ----------------------------
def generate_dummy_input(batch_size, n_window, window_length, num_sensors, n_channel=1):
    """Gera uma lista com tensores por sensor no formato exigido pela MultiHeadNetworkGraph"""
    dummy_input = []
    for _ in range(num_sensors):
        # Tensor com shape (batch_size, num_windows, window_size, 1)
        data = np.random.randn(batch_size, n_window, window_length, n_channel).astype(np.float32)
        data_torch = torch.from_numpy(data)
        dummy_input.append(data_torch)
    return dummy_input


def init_weights_with_constant(model, value=0.5):
    """
    Inicializa todos os pesos e biases do modelo com um valor constante.
    Similar a set_weights no Keras com np.ones_like * value.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.constant_(module.weight, value)
            if module.bias is not None:
                nn.init.constant_(module.bias, value)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, value)  # gamma
            nn.init.constant_(module.bias, value)    # beta
            module.running_mean.fill_(value)
            module.running_var.fill_(value)
        elif isinstance(module, nn.LSTM):
            for param_name, param in module.named_parameters():
                if "weight" in param_name:
                    nn.init.constant_(param, value)
                elif "bias" in param_name:
                    nn.init.constant_(param, value)



# ----------------------------
# Define a arquitetura manualmente (Conv1DBlock)
# ----------------------------
def build_fn_dict():
    return {
        'Conv1DBlock': {
            'function': 'Conv1DBlock',
            'params': {
                'kernel': 2,
                'filters': 3,
                'strides': 1
            }
        }
    }

# ----------------------------
# Mapeia os nomes de blocos para classes
# ----------------------------
functions_dict = {
    'Conv1DBlock': Conv1DBlock
}

# ----------------------------
# Teste principal
# ----------------------------
def test_multihead_model():
    set_seed(42)

    # Configurações
    batch_size = 4
    num_windows = 4
    window_size = 2
    num_sensors = 3
    in_channels = 1
    num_classes = 1
    lstm_1 = 8
    lstm_2 = 4

    # Arquitetura
    fn_dict = build_fn_dict()
    net_list = ['Conv1DBlock']

    # Instancia o modelo
    model = MultiHeadNetworkGraphNew(
        num_classes=num_classes,
        network_config='default',
        network_gap=False,
        num_lstm_cells_1=lstm_1,
        num_lstm_cells_2=lstm_2,
        in_channels=in_channels,
        num_sensors=num_sensors
    )

    # Insere o dicionário de funções e arquitetura
    model.create_functions(fn_dict=fn_dict, net_list=net_list)
    init_weights_with_constant(model, value=0.5)

    # Gera entrada dummy
    x = generate_dummy_input(batch_size, num_windows, window_size, num_sensors)

    # Forward
    model.eval()
    output = model(x)
    init_weights_with_constant(model, value=0.5)
    output = model(x)
    
    # Resultado
    print("Saída do modelo:")
    print(output.detach().numpy())

if __name__ == "__main__":
    test_multihead_model()
