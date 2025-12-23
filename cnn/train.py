""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Compute the fitness of a model_net using the evolved networks.

Documentation:

    - Automatic mixed precision training (AMP): 
        - https://pytorch.org/docs/stable/amp.html, 
        - https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision
    
"""
import os
import time
import numpy as np
import joblib
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Union, Any
from cnn import model, input, metrics, fitness_utils
from util import create_info_file, init_log, load_yaml
from torch.cuda.amp import GradScaler
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR, MultiStepLR
from torch.optim.lr_scheduler import LambdaLR
from cnn.train_detailed import TFScheduler, keras_style_scheduler


TRAIN_TIMEOUT = 5400

current_directory = os.path.dirname(os.path.dirname(__file__))
log_directory = os.path.join(current_directory, 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file = os.path.join(log_directory, 'train.log')
LOGGER = init_log("INFO", name=__name__, file_path=log_file)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # para evitar sqrt(0)

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


class MetricTracker:
    """
    A class for tracking evaluation metrics during training and validation,
    supporting both classification and regression tasks.

    Parameters
    ----------
    task : str
        The type of task. Must be either 'multi-class' for classification or 'regression' for regression problems.

    Attributes
    ----------
    task : str
        The task type.
    total : float
        Accumulated number of samples or accumulated loss value.
    correct : int
        Accumulated number of correct predictions (only for classification).
    count : int
        Number of batches (used for averaging loss in regression).
    """

    def __init__(self, task: str):
        self.task = task
        self.total = 0
        self.correct = 0
        self.count = 0

    def update(self, y_logits: torch.Tensor, labels: torch.Tensor):
        """
        Update the internal state based on model predictions and ground truth labels.

        For classification tasks ('multi-class'), accumulates the number of correct predictions and total samples.
        For regression tasks ('regression'), accumulates the root mean squared error (RMSE) over batches.

        Parameters
        ----------
        y_logits : torch.Tensor
            Model predictions. Shape: (batch_size, num_classes) for classification,
            (batch_size, 1) or (batch_size,) for regression.

        labels : torch.Tensor
            Ground truth labels. Shape should match y_logits accordingly.
        """
        if self.task == 'multi-class' or self.task == 'classification':
            _, predicted = y_logits.max(1)
            self.total += labels.size(0)
            self.correct += predicted.eq(labels).sum().item()
        elif self.task == 'regression':
            sse = torch.nn.functional.mse_loss(y_logits, labels, reduction='sum').item()
            self.total += sse
            self.count += labels.size(0)
        else:
            raise ValueError(f"Unknown task type: {self.task}")

    def result(self) -> float:
        """
        Compute the final metric based on accumulated values.

        Returns
        -------
        float
            The average metric:
            - Accuracy (%) for classification.
            - Root Mean Squared Error (RMSE) for regression.
        """
        if self.task == 'multi-class' or self.task == 'classification':
            return 100.0 * self.correct / self.total if self.total > 0 else 0.0
        elif self.task == 'regression':
            mse = self.total / self.count if self.count > 0 else 0.0
            return float(np.sqrt(mse))


def lr_schedule(epoch):
    if epoch == 10:
        return 0.1
    elif epoch == 15:
        return 0.1
    elif epoch == 20:
        return torch.exp(torch.tensor(-0.1)).item()
    else:
        return 1.0

def init_weights(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        # Glorot/Xavier uniform initialization (como no Keras)
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        init.ones_(module.weight)
        init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)  # ← Keras usa Orthogonal aqui
            elif 'bias' in name:
                init.zeros_(param)
                # Set forget gate bias to 1 (assuming default PyTorch LSTM layout)
                hidden_size = param.shape[0] // 4
                param.data[hidden_size:2*hidden_size] = 1.0


def train_epoch(model, criterion, optimizer, data_loader, params, scaler, target_scaler=None):
    model.train()
    train_loss = 0.0
    device = torch.device(params['device'])
    amp_device = device.type  # 'cuda' or 'cpu'
    
    metric_tracker = MetricTracker(params['task'])

    for inputs, labels in data_loader:
        if "dataset_type" in params and params["dataset_type"] == "multihead":
            inputs = [inp.to(device) for inp in inputs]
            labels = labels.to(device)
        else:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=params['mixed_precision']):
            y_logits = model(inputs)
            if params['task'] == 'multi-class':
                labels = labels.squeeze().long()
            loss = criterion(y_logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        # 2. Desnormalização
        if params['task'] == 'regression' and target_scaler is not None:
            # Converte para numpy e garante shape (N, 1) exigido pelo sklearn
            y_pred_np = y_logits.detach().cpu().numpy().reshape(-1, 1)
            labels_np = labels.detach().cpu().numpy().reshape(-1, 1)

            y_pred_real = target_scaler.inverse_transform(y_pred_np)
            labels_real = target_scaler.inverse_transform(labels_np)

            metric_tracker.update(torch.tensor(y_pred_real, dtype=torch.float32).to(device), 
                                 torch.tensor(labels_real, dtype=torch.float32).to(device))
        else:
            metric_tracker.update(y_logits, labels)
        
    avg_loss = train_loss / len(data_loader)
    avg_metric = metric_tracker.result()
    return avg_loss, avg_metric

def evaluate(model, criterion, data_loader, params, target_scaler=None):
    model.eval()
    validation_loss = 0.0
    device = torch.device(params['device'])
    amp_device = device.type  # 'cuda' or 'cpu'

    metric_tracker = MetricTracker(params['task'])

    with torch.no_grad():
        for inputs, labels in data_loader:
            if "dataset_type" in params and params["dataset_type"] == "multihead":
                inputs = [inp.to(device) for inp in inputs]
                labels = labels.to(device)
            else:
                inputs, labels = inputs.to(device), labels.to(device)
            with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=params['mixed_precision']):
                y_logits = model(inputs)
                if params['task'] == 'multi-class':
                    labels = labels.squeeze().long() # medmnist
                loss = criterion(y_logits, labels)
            validation_loss += loss.item()

            # 2. Desnormalização
            if params['task'] == 'regression' and target_scaler is not None:
                # Converte para numpy e garante shape (N, 1) exigido pelo sklearn
                y_pred_np = y_logits.cpu().numpy().reshape(-1, 1)
                labels_np = labels.cpu().numpy().reshape(-1, 1)

                y_pred_real = target_scaler.inverse_transform(y_pred_np)
                labels_real = target_scaler.inverse_transform(labels_np)

                metric_tracker.update(torch.tensor(y_pred_real, dtype=torch.float32).to(device), 
                                 torch.tensor(labels_real, dtype=torch.float32).to(device))
            else:
                metric_tracker.update(y_logits, labels)

    avg_metric = metric_tracker.result()
    validation_loss /= len(data_loader)

    return validation_loss, avg_metric

def train(model:torch.nn.Module,
          criterion:torch.nn.Module,
          optimizer:torch.optim.Optimizer, 
          train_loader:torch.utils.data.DataLoader,
          val_loader:torch.utils.data.DataLoader, 
          params:Dict, debug=False,
          target_scaler = None) -> Dict:
    """
    Train a neural network model.

    Args:
        model: Model to be trained.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        params: Dictionary with parameters necessary for training
            - max_epochs: Number of epochs to train.
            - epochs_to_eval: Number of epochs before starting validation.
            - t0: Time when the training started.
        device: Device to run the training on (CPU or GPU).

    Returns:
        training_results: Dictionary with the training results.
        
            -training_losses: List of training losses for each epoch.
            -validation_losses: List of validation losses for each epoch.
            -best_accuracy: Best validation accuracy achieved.
    """
    model.train()
    training_losses = []
    training_metrics = []
    validation_losses = []
    validation_metrics = []
    #best_accuracy = 0.0
    best_validation_loss = float('inf')
    
    training_results = {}
    max_epochs = params['max_epochs']
    epochs_to_eval = params['epochs_to_eval']
    start_eval = max_epochs - epochs_to_eval
    
    # Automatic mixed precision training (AMP)
    scaler = GradScaler(enabled=params['mixed_precision'])

    # Define o caminho do modelo
    best_model_path = os.path.join(params['model_path'], 'temp_best_model_6e31.pth')
    # Variável para segurar os pesos na memória RAM
    best_model_state = None

    milestones = [int(0.5 * max_epochs), int(0.75 * max_epochs)]
    scheduler_type = params['lr_scheduler_train']
    if scheduler_type == 'exponential':
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_type == 'reduce_on_plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.1)
    elif scheduler_type == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0, last_epoch=-1)
    elif scheduler_type == 'multistep':
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif scheduler_type == 'LambdaLR':
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    elif scheduler_type == 'TFScheduler':
        lr_scheduler = TFScheduler(optimizer, keras_style_scheduler)
    else:
        lr_scheduler = None

    # Check which metric to calculate
    if params['task'] == 'multi-class' or params['task'] == 'classification':
        metric_name = "accuracy"
        best_val_metric = float('-inf')
    elif params['task'] == 'regression':
        best_val_metric = float('inf')
        metric_name = "rmse"
    else:
        best_val_metric = None
        raise ValueError(f"Unknown task type: {params['task']}")
    
    LOGGER.info(f"Training {params['task']} model with {metric_name} metric")
    for epoch in range(1, max_epochs + 1):
        train_loss, train_metric = train_epoch(model, criterion, optimizer, train_loader, params, scaler, target_scaler)
        training_losses.append(train_loss)
        training_metrics.append(train_metric)

        if epoch < start_eval and (time.time() - params['t0']) > TRAIN_TIMEOUT:
            print("Timeout reached")
            raise TimeoutError()

        validation_loss = None
        val_metric = None
        if epoch > start_eval:
            validation_loss, val_metric = evaluate(model, criterion, val_loader, params, target_scaler)
            validation_losses.append(validation_loss)
            validation_metrics.append(val_metric)

            if params['task'] == 'regression':
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    create_info_file(params['model_path'], {f'best_{metric_name}': best_val_metric}, f'best_{metric_name}.txt')
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    create_info_file(params['model_path'], {f'best_{metric_name}': best_val_metric}, f'best_{metric_name}.txt')
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                create_info_file(params['model_path'], {'best_validation_loss': best_validation_loss}, 'best_validation_loss.txt')
        
        if lr_scheduler is not None:
            if scheduler_type == 'reduce_on_plateau':
                if (validation_loss is not None):
                    lr_scheduler.step(validation_loss)                
            else:
                lr_scheduler.step()

    if debug:
        if epoch >= start_eval and val_metric is not None:
            print(f"Epoch [{epoch}/{max_epochs}] - Training Loss: {train_loss:.4f} - Validation Loss: {validation_loss:.4f} - Validation {metric_name}: {val_metric:.2f}%")
        elif epoch % 5 == 0:
            print(f"Epoch [{epoch}/{max_epochs}] - Training Loss: {train_loss:.4f}")

    # SALVAMENTO FINAL NO DISCO (Apenas 1 vez)
    if best_model_state is not None:
        torch.save(best_model_state, best_model_path)
        LOGGER.info(f"Best model saved to {best_model_path}")

    params['t1'] = time.time()
    params['training_time'] = params['t1'] - params['t0']
    
    model_metrics = metrics.ModelMetrics(model, device=params['device'])
    
    if "dataset_type" in params and params["dataset_type"] == "multihead":
        batch = next(iter(val_loader))
        inference_images = [inp[:10].to(params['device']) for inp in batch[0]]
    else:
        inference_images = next(iter(val_loader))[0][:10].to(params['device'])

    input_shape = params['input_shape']
    
    cuda_inference_time = model_metrics.measure_inference_time(inference_images)
    model_memory_usage = model_metrics.measure_memory(input_shape) / (1024 ** 2)  # Convert bytes to MB
    total_trainable_params = model_metrics.measure_parameters()
    total_flops = model_metrics.measure_flops(input_shape)
        
    fitness_metric = params['fitness_metric']
    mo_base_metric = params['mo_metric_base']

    if fitness_metric == f'best_{metric_name}' or (fitness_metric == 'scalar_multi_objective' and mo_base_metric == metric_name):
        metric_value = best_val_metric
        metric_type = metric_name
    elif fitness_metric == 'best_loss' or (fitness_metric == 'scalar_multi_objective' and mo_base_metric == 'loss'):
        metric_value = best_validation_loss
        metric_type = 'loss'
    else:
        raise ValueError(f"Invalid fitness_metric: {fitness_metric}")   
        
    # Scalarized multi-objective function
    scalar_multi_objective = fitness_utils.mofitness(metric_value=metric_value,params=total_trainable_params,inference_time=cuda_inference_time,
                                    T_p=params['max_params'], T_t=params['max_inference_time'],metric_type=metric_type)
    
    # Lower loss leads to higher fitness - Reciprocal Transformation
    fitness_val_loss = - best_validation_loss if params['task'] == 'regression' else (1 / (1 + best_validation_loss))*100.0 
    
    params['total_trainable_params'] = total_trainable_params
    params['cuda_inference_time'] = cuda_inference_time
    params['model_memory_usage'] = model_memory_usage
    params['total_flops'] = total_flops
    params[f'best_{metric_name}'] = best_val_metric
    params['best_validation_loss'] = best_validation_loss
    params['fitness_val_loss'] = fitness_val_loss
    params['scalar_multi_objective'] = scalar_multi_objective

    
    LOGGER.info(f"Cuda Inference time: {cuda_inference_time} microseconds")
    LOGGER.info(f"Total trainable parameters: {round(total_trainable_params / 1e6,2)}M")
    
    create_info_file(params['model_path'], params, 'training_params.txt')
    
    training_results['training_losses'] = training_losses
    training_results[f'training_{metric_name}'] = training_metrics
    training_results['validation_losses'] = validation_losses
    training_results[f'validation_{metric_name}'] = validation_metrics
    training_results['cuda_inference_time'] = cuda_inference_time # in microseconds
    training_results['model_memory_usage'] = model_memory_usage # in MB
    training_results['total_trainable_params'] = total_trainable_params / 1e6 # in millions
    training_results['total_flops'] = total_flops / 1e6  # Convert to MFLOPs
    training_results[f'best_{metric_name}'] = best_val_metric
    training_results['fitness_val_loss'] = fitness_val_loss
    training_results['scalar_multi_objective'] = scalar_multi_objective        
    return training_results


def fitness_calculation(id_num:str,
                        params:Dict[str, Any], 
                        fn_dict:Dict[str, Any],
                        net_list:List[str],
                        decoded_params:Dict[str, Any],
                        train_loader:torch.utils.data.DataLoader,
                        val_loader:torch.utils.data.DataLoader,
                        return_val,
                        dataset_info:Dict[str, Any]=None,
                        debug:bool=False) -> Dict[str, Union[List[float], float]]:
    """Train and evaluate a model using evolved hyperparameters.

    This function trains and evaluates a convolutional neural network model using the specified
    configuration and evolved hyperparameters.

    Args:
        id_num (str): A string identifying the generation number and the individual number.
        params (Dict[str, Any]): A dictionary with parameters necessary for training, including
            the evolved hyperparameters.
        fn_dict (Dict[str, Any]): A dictionary with definitions of the possible layers, including
            their names and parameters.
        net_list (List[str]): A list with names of layers defining the network, in the order they appear.

    Returns:
        Dict[str, Union[List[float], float]]: A dictionary containing the training results.

        - 'training_losses' (List[float]): List of training losses for each epoch.
        - 'validation_losses' (List[float]): List of validation losses for each epoch.
        - 'best_accuracy' (float): Best validation accuracy achieved.

    Raises:
        TimeoutError: If the training process takes too long to complete.
    """

    device = params['device']
    params['net_list'] = net_list
    params['decoded_params'] = decoded_params
    model_path = os.path.join(params['experiment_path'], id_num)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    params['model_path'] = model_path
    params['generation'] = id_num.split('_')[0]
    params['individual'] = id_num.split('_')[1]

    LOGGER.info(f"Training model {id_num} on device {device} ...")
    # Load data info
    if params['dataset'].lower() in input.available_datasets and dataset_info is not None:
        dataset_info = input.available_datasets[params['dataset'].lower()]
    # else:
    #     dataset_info = load_yaml(os.path.join(params['data_path'], 'data_info.txt'))
    
    params['num_classes'] = dataset_info['num_classes']
    params['task'] = dataset_info['task']
    dataset_info['shape'] = list(train_loader.dataset.X[0].shape[1:])

    # check if cbam is a key in the fn_dict
    has_cbam_key = any(key.startswith('cbam') for key in fn_dict)

    num_sensors_data = dataset_info['num_sensors']
    num_sensors_config = params['extra_params']['num_sensors']

    if num_sensors_data != num_sensors_config:
        raise ValueError(
            f"Mismatch in number of sensors: dataset reports {num_sensors_data}, "
            f"but config specifies {num_sensors_config}."
        )

    # Create the model
    if "dataset_type" in params and params["dataset_type"] == "multihead":
        # If multi-head architecture is enabled, create a multi-head network
        lstm_multiplier = params['extra_params']['lstm_multiplier']
        model_net = model.MultiHeadNetworkGraphNew(num_classes=dataset_info['num_classes'], 
                                                network_config=params['network_config'], 
                                                network_gap=params['network_gap'],
                                                in_channels=params['extra_params']['in_channels'],
                                                num_sensors=num_sensors_data,
                                                num_lstm_cells_1=int(decoded_params['lstm_1']*lstm_multiplier),
                                                num_lstm_cells_2=int(decoded_params['lstm_2']*lstm_multiplier),
                                                shared_head_architecture=params['extra_params']['shared_head_architecture'])

        single_sensor_shape = [params['batch_size']] + dataset_info['shape']
        input_shape = [single_sensor_shape] * dataset_info["num_sensors"]

        inputs = [torch.randn(single_sensor_shape) for _ in range(dataset_info["num_sensors"])]

    else:
        model_net = model.NetworkGraph(num_classes=dataset_info['num_classes'], 
                                       network_config=params['network_config'], 
                                       network_gap=params['network_gap'],
                                       in_channels=dataset_info["shape"][0])
        # Add the fully connected layer to the model
        input_shape =  [params['batch_size']] + dataset_info['shape']
        inputs = torch.randn(input_shape)

    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    model_net.create_functions(fn_dict=filtered_dict, net_list=net_list, cbam=has_cbam_key)

    model_net.eval()
    with torch.no_grad():
        _ = model_net(inputs)
    model_net.to(device)
    

    model_net.apply(init_weights)

    params['input_shape'] = input_shape

    # define criterion
    crit_config = params["criterion"]

    # If your loss is inside torch.nn
    if hasattr(nn, crit_config["name"]):
        criterion_cls = getattr(nn, crit_config["name"])
    else:
        # If it's a custom one (e.g., RMSELoss defined in code)
        criterion_cls = globals()[crit_config["name"]]

    criterion = criterion_cls(**crit_config.get("params", {}))

    # define optimizer
    opt_config = params["optimizer"]

    optimizer_cls = getattr(torch.optim, opt_config["name"])

    optimizer = optimizer_cls(model_net.parameters(), **opt_config.get("params", {}))

    # Training time start counting here.
    params['t0'] = time.time()

    # Check which metric to calculate
    if params['task'] == 'multi-class' or params['task'] == 'classification':
        metric_name = "accuracy"
    elif params['task'] == 'regression':
        metric_name = "rmse"
    else:
        raise ValueError(f"Unknown task type: {params['task']}")

    target_scaler = None
    if ('target_normalization' in params) and (params['target_normalization']["name"] is not None):
        target_scaler_name = params['target_normalization']['path'].replace('.save', '_'+params['exp']+'.save')
        target_scaler_path = os.path.join(params['data_path'], 'scaler', target_scaler_name)
        if os.path.exists(target_scaler_path):
            target_scaler = joblib.load(target_scaler_path)
            LOGGER.info(f"Target scaler loaded for denormalization from {target_scaler_path}")

    # Train the model in fitness scheme
    try:
        results_dict = train(model_net,
                             criterion,
                             optimizer,
                             train_loader,
                             val_loader,
                             params,
                             debug,
                             target_scaler)
        if debug:
            result = results_dict
            return result
        else:
            #return_val.value = results_dict['best_accuracy']
            if params['fitness_metric'] == f'best_{metric_name}':
                return_val[0] = results_dict[f'best_{metric_name}']
                return_val[1] = results_dict['total_trainable_params']
                return_val[2] = results_dict['cuda_inference_time']
            elif params['fitness_metric'] == 'best_loss':
                return_val[0] = results_dict['fitness_val_loss'] # 1 - best_validation_loss
                return_val[1] = results_dict['total_trainable_params']
                return_val[2] = results_dict['cuda_inference_time']
            elif params['fitness_metric'] == 'scalar_multi_objective':
                return_val[0] = results_dict['scalar_multi_objective']
                return_val[1] = results_dict['total_trainable_params']
                return_val[2] = results_dict['cuda_inference_time']
            else:
                raise ValueError(f"Invalid fitness metric: {params['fitness_metric']}")
        LOGGER.info(f"Training of model {id_num} finished, best {params['fitness_metric']}: {round(return_val[0], 2)}")
        
    except (TimeoutError, MemoryError) as e:
        LOGGER.error(f"Exception: {e}")
        return_val[:] = [0.0, 0.0, 0.0]
    except Exception as e:
        if "out of memory" in str(e):
            LOGGER.error(f"CUDA out of memory exception, error: {e}")
            return_val[:] = [0.0, 0.0, 0.0]
        else:
            LOGGER.error(f"Exception: {e}")
            return_val[:] = [0.0, 0.0, 0.0]
        raise e