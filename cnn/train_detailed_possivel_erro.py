""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Compute the fitness of a model_net using the evolved networks.


"""
import os
import time
import numpy as np
import torch
from medmnist import INFO, Evaluator
import torch.nn as nn
from tqdm.notebook import tqdm
from typing import Dict, List, Union, Any
from sklearn.metrics import confusion_matrix
from cnn import model, input, metrics
from util import create_info_file, init_log, load_yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR, MultiStepLR
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.init as init
import random
import pandas as pd

current_directory = os.path.dirname(os.path.dirname(__file__))
log_directory = os.path.join(current_directory, 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    
log_file = os.path.join(log_directory, 'retrain.log')
LOGGER = init_log("INFO", name=__name__, file_path=log_file)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
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
            rmse = torch.sqrt(torch.nn.functional.mse_loss(y_logits, labels, reduction='mean')).item()
            self.total += rmse
            self.count += 1
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
            root_mean_squared_error = self.total / self.count if self.count > 0 else 0.0
            return torch.tensor(root_mean_squared_error).item()


def lr_schedule(epoch):
    if epoch == 10:
        return 0.1
    elif epoch == 15:
        return 0.1
    elif epoch == 20:
        return torch.exp(torch.tensor(-0.1)).item()
    else:
        return 1.0


def init_weights2(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

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


def release_gpu_memory(gpu_name='cuda:0'):
    """
    Release GPU memory.
    
    Args:
        gpu_name (str): The name of the GPU device (default is 'cuda').
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU memory to release.")
        return
    
    if gpu_name == 'cuda':
        gpu_name = 'cuda:0'

    device = torch.device(gpu_name)
    torch.cuda.set_device(device)

    # Get memory usage before clearing the cache
    memory_allocated_before = torch.cuda.memory_allocated(device)
    memory_reserved_before = torch.cuda.memory_reserved(device)

    # Clear the cache
    torch.cuda.empty_cache()

    # Check if there was a significant change
    memory_allocated_after = torch.cuda.memory_allocated(device)
    memory_reserved_after = torch.cuda.memory_reserved(device)

    # Verificar si hubo un cambio significativo
    if memory_allocated_before != memory_allocated_after or memory_reserved_before != memory_reserved_after:
        print("Cache was cleared.")
    else:
        print("Cache was already empty.")

def compute_metrics(model, data_loader, params):
    model.eval()
    all_labels = []
    all_predictions = []
    auc, acc = 0, 0
    y_score = torch.tensor([]).to(params['device'])
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(params['device']), labels.to(params['device'])
            y_logits = model(inputs)
            _, predicted = y_logits.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            if params['task'] == 'multi-class':
                output = y_logits.softmax(dim=-1)
                y_score = torch.cat((y_score, output), 0)

        if params['task'] == 'multi-class':
            y_score = y_score.cpu().detach().numpy()
            evaluator = Evaluator(params['dataset'], split='test', root=params['data_path'])
            metrics = evaluator.evaluate(y_score)
            auc, acc = metrics

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    return conf_matrix, auc, acc

def reset_and_load_best_model(params, best_model_path):
    # Reinitialize the original model
    
    # Create the model
    if "dataset_type" in params and params["dataset_type"] == "multihead":
        # If multi-head architecture is enabled, create a multi-head network
        best_model = model.MultiHeadNetworkGraph(num_classes=params['num_classes'], 
                                                network_config=params['network_config'], 
                                                network_gap=params['network_gap'],
                                                in_channels=1,
                                                num_sensors=params['num_sensors'],
                                                num_lstm_cells_1=int(params['decoded_params']['lstm_1']*20),
                                                num_lstm_cells_2=int(params['decoded_params']['lstm_2']*20))

        single_sensor_shape = params['input_shape'][0]

        input_random = [torch.randn(single_sensor_shape) for _ in range(params["num_sensors"])]

    else:
        best_model = model.NetworkGraph(num_classes=params["num_classes"],
                                        network_config=params['network_config'], 
                                        network_gap=params['network_gap'],
                                        in_channels=params["input_shape"][1])
        input_random = torch.randn(params['input_shape'])

    filtered_dict = {key: item for key, item in params['fn_dict'].items() if key in params['net_list']}
    best_model.create_functions(fn_dict=filtered_dict, net_list=params['net_list'])

    best_model.eval()
    with torch.no_grad():
        _ = best_model(input_random)
    # Load the state dictionary of the best model into the new model
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(params['device'])

    return best_model

def train_epoch(model, criterion, optimizer, data_loader, params):
    model.train()
    train_loss = 0.0
    device = torch.device(params['device'])

    metric_tracker = MetricTracker(params['task'])

    for inputs, labels in data_loader:
        if "dataset_type" in params and params["dataset_type"] == "multihead":
            inputs = [inp.to(device) for inp in inputs]
            labels = labels.to(device)
        else:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        y_logits = model(inputs)
        
        if params['task'] == 'multi-class':
            labels = labels.squeeze().long() # medmnist
            
        loss = criterion(y_logits, labels)
        loss.backward()       
        optimizer.step()
        train_loss += loss.item()
        metric_tracker.update(y_logits, labels)
        
    avg_metric = metric_tracker.result()
    train_loss /= len(data_loader)
    return train_loss, avg_metric

def evaluate(model, criterion, data_loader, params, test=False):
    model.eval()
    eval_loss = 0.0
    device = torch.device(params['device'])

    metric_tracker = MetricTracker(params['task'])

    with torch.no_grad():
        for inputs, labels in data_loader:
            if "dataset_type" in params and params["dataset_type"] == "multihead":
                inputs = [inp.to(device) for inp in inputs]
                labels = labels.to(device)
            else:
                inputs, labels = inputs.to(device), labels.to(device)
            y_logits = model(inputs)
            
            if params['task'] == 'multi-class':
                labels = labels.squeeze().long() # medmnist
                
            loss = criterion(y_logits, labels)
            eval_loss += loss.item()
            metric_tracker.update(y_logits, labels)

    avg_metric = metric_tracker.result()
    eval_loss /= len(data_loader)
    
    if test:
        if params['task'] == 'regression':
            return eval_loss, avg_metric
        else:
            confusion_matrix, auc, acc = compute_metrics(model, data_loader, params)
            return eval_loss, avg_metric, auc, acc , confusion_matrix

    return eval_loss, avg_metric

def train(model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        params: Dict[str, Union[int, float, str]]) -> Dict[str, Union[List[float], float]]:
    """
    Retrain a convolutional neural network model.

    Args:
        model (Module): Model to be trained.
        criterion (Module): Loss function.
        optimizer (Optimizer): Optimization algorithm.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        params (Dict[str, Union[int, float, str]]): Dictionary with parameters necessary for training.
            - 'max_epochs' (int): Number of epochs to train.
            - 'model_path' (str): Path to save the trained model.
            - 'lr_scheduler' (str): Learning rate scheduler to use.
            - 'experiment_path' (str): Path to save the experiment.
            - 'device' (str): Device to use for training.

    Returns:
        Dict[str, Union[List[float], float]]: Dictionary with the training results.
        
        - 'training_losses' (List[float]): List of training losses for each epoch.
        - 'training_accuracies' (List[float]): List of training accuracies for each epoch.
        - 'validation_losses' (List[float]): List of validation losses for each epoch.
        - 'validation_accuracies' (List[float]): List of validation accuracies for each epoch.
        - 'best_accuracy' (float): Best validation accuracy achieved.
        - 'test_loss' (float): Loss on the test set.
        - 'test_accuracy' (float): Accuracy on the test set.
        - 'auc_score' (float): AUC score on the test set.
        - 'acc_medmnist' (float): Accuracy on the test set.
        - 'confusion_matrix' (numpy.ndarray): Confusion matrix on the test set.
        - 'total_trainable_params' (int): Total number of trainable parameters in the model.
    """
    model.train()
    training_losses = []
    training_metrics = []
    validation_losses = []
    validation_metrics = []
    auc_value = 0.0
    acc_med = 0.0
    training_results = {}
    max_epochs = params['max_epochs']
    milestones = [0.5 * max_epochs, 0.75 * max_epochs]

    best_model_path = os.path.join(params['model_path'], 'best_model.pth')
    
    if params['lr_scheduler'] == 'exponential':
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif params['lr_scheduler'] == 'reduce_on_plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.1)
    elif params['lr_scheduler'] == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0, last_epoch=-1)
    elif params['lr_scheduler'] == 'multistep':
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif params['lr_scheduler'] == 'LambdaLR':
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
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

    #for epoch in tqdm(range(1, max_epochs + 1), desc="Retrain Scheme"):
    for epoch in range(1, max_epochs + 1):
        train_loss, train_metric = train_epoch(model, criterion, optimizer, train_loader, params)
        training_losses.append(train_loss)
        training_metrics.append(train_metric)
        
        validation_loss, val_metric = evaluate(model, criterion, val_loader, params)
        validation_losses.append(validation_loss)
        validation_metrics.append(val_metric)

        test_loss_temp, test_metric_temp = evaluate(model, criterion, test_loader, params, test=True)
        print("Test Loss", test_loss_temp)
        print("Test Metric", test_metric_temp)
        
        if params['task'] == 'regression':
            if val_metric < best_val_metric: 
                best_val_metric = val_metric
                torch.save(model.state_dict(), best_model_path)
                create_info_file(params['model_path'], {f'best_{metric_name}': best_val_metric}, f'best_{metric_name}.txt')
        else:
            if val_metric > best_val_metric: 
                best_val_metric = val_metric
                torch.save(model.state_dict(), best_model_path)
                create_info_file(params['model_path'], {f'best_{metric_name}': best_val_metric}, f'best_{metric_name}.txt')

        if epoch % 25 == 0:
            LOGGER.info(f"Experiment: {params['experiment_path']} - Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss:.2f} - Validation loss: {validation_loss:.2f} - Validation {metric_name}: {val_metric:.2f}%")
            #print(f"Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss} - Validation loss: {validation_loss} - Validation accuracy: {accuracy}%")

        if lr_scheduler is not None:
            if params['lr_scheduler'] == 'reduce_on_plateau':
                lr_scheduler.step(validation_loss)                
            else:
                lr_scheduler.step()
        
    best_model_loaded = reset_and_load_best_model(params, best_model_path)

    if params['task'] == 'regression':
        test_loss, test_metric = evaluate(best_model_loaded, criterion, test_loader, params, test=True)
        auc_value, acc_med, confusion_matrix = None, None, None
    else:
        test_loss, test_metric, auc_value, acc_med, confusion_matrix = evaluate(best_model_loaded, criterion, test_loader, params, test=True)
    
    LOGGER.info(f"Experiment: {params['experiment_path']} - Test loss: {test_loss:.2f} - Test {metric_name}: {test_metric:.2f}%")
    #print(f"Test loss: {test_loss} - Test accuracy: {test_accuracy}%")
            
    params['t1'] = time.time()
    
    create_info_file(params['model_path'], params, 'retraining_params.txt')
    
    model_metrics = metrics.ModelMetrics(best_model_loaded, device=params['device'])

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
        
    training_results['total_trainable_params'] = total_trainable_params
    training_results['cuda_inference_time'] = cuda_inference_time
    training_results['total_flops'] = total_flops
    training_results['model_memory_usage'] = model_memory_usage
    training_results['training_losses'] = training_losses
    training_results[f'training_{metric_name}'] = training_metrics
    training_results['validation_losses'] = validation_losses
    training_results[f'validation_{metric_name}'] = validation_metrics
    training_results[f'best_{metric_name}'] = best_val_metric
    training_results['test_loss'] = test_loss
    training_results[f'test_{metric_name}'] = test_metric
    training_results['auc_score'] = auc_value
    training_results['acc_medmnist'] = acc_med
    training_results['confusion_matrix'] = confusion_matrix.tolist() if (confusion_matrix is not None) else confusion_matrix
            
    return training_results


def train_and_eval(params: Dict[str, Any], 
                   fn_dict: Dict[str, Any],
                   net_list:List[str],
                   best_individual_info: Dict[str, Any],
                   train_loader:torch.utils.data.DataLoader, 
                   val_loader:torch.utils.data.DataLoader,
                   test_loader:torch.utils.data.DataLoader) -> Dict[str, Union[List[float], float]]:
    """
    This function retrains and evaluates a convolutional neural network model using the specified
    configuration.

    Args:
        params (Dict[str, Any]): A dictionary with parameters necessary for training, including.
        fn_dict (Dict[str, Any]): A dictionary with definitions of the possible layers, including
            their names and parameters.
        net_list (List[str]): A list with names of layers defining the network, in the order they appear.

    Returns:
        Dict[str, Union[List[float], float]]: Dictionary with the training results.
        
        - 'training_losses' (List[float]): List of training losses for each epoch.
        - 'training_accuracies' (List[float]): List of training accuracies for each epoch.
        - 'validation_losses' (List[float]): List of validation losses for each epoch.
        - 'validation_accuracies' (List[float]): List of validation accuracies for each epoch.
        - 'best_accuracy' (float): Best validation accuracy achieved.
        - 'test_loss' (float): Loss on the test set.
        - 'test_accuracy' (float): Accuracy on the test set.
        - 'auc_score' (float): AUC score on the test set.
        - 'acc_medmnist' (float): Accuracy on the test set.
        - 'confusion_matrix' (numpy.ndarray): Confusion matrix on the test set.
        - 'total_trainable_params' (int): Total number of trainable parameters in the model.
    """
    
    device = params['device']
    model_path = os.path.join(params['experiment_path'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    params['model_path'] = model_path
    params['decoded_params'] = best_individual_info['decoded_params']

    LOGGER.info(f"Start retraining of the experiment: {params['experiment_path']}")
    # Load data information
    if params['dataset'].lower() in input.available_datasets:
        dataset_info = input.available_datasets[params['dataset'].lower()]
    else:
        dataset_info = load_yaml(os.path.join(params['data_path'], 'data_info.txt'))

    dataset_info['shape'] = list(train_loader.dataset.X[0].shape[1:])

    # Create the model
    if "dataset_type" in params and params["dataset_type"] == "multihead":
        # If multi-head architecture is enabled, create a multi-head network
        model_net = model.MultiHeadNetworkGraph(num_classes=dataset_info['num_classes'], 
                                                network_config=params['network_config'], 
                                                network_gap=params['network_gap'],
                                                in_channels=1,
                                                num_sensors=dataset_info['num_sensors'],
                                                num_lstm_cells_1=int(best_individual_info['decoded_params']['lstm_1']*20),
                                                num_lstm_cells_2=int(best_individual_info['decoded_params']['lstm_2']*20))

        single_sensor_shape = [params['batch_size']] + dataset_info['shape']
        input_shape = [single_sensor_shape] * dataset_info["num_sensors"]
        params["num_sensors"] = dataset_info["num_sensors"]
        inputs = [torch.randn(single_sensor_shape) for _ in range(dataset_info["num_sensors"])]
    else:
        model_net = model.NetworkGraph(num_classes=dataset_info['num_classes'], 
                                       network_config=params['network_config'], 
                                       network_gap=params['network_gap'],
                                       in_channels=dataset_info["shape"][0])
        input_shape =  [params['batch_size']] + dataset_info['shape']
        inputs = torch.randn(input_shape)

    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    model_net.create_functions(fn_dict=filtered_dict, net_list=net_list)

    #params['model_net'] = model_net
    params['net_list'] = net_list
    params['fn_dict'] = fn_dict
    params['num_classes'] = dataset_info["num_classes"]
    params['task'] = dataset_info["task"]
    
    # Add the fully connected layer to the model
    model_net.eval()
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
    
    try:
        results_dict = train(model_net, criterion, optimizer, train_loader, val_loader, test_loader, params)
    except RuntimeError as e:
        if "out of memory" in str(e):
            LOGGER.error(f"Out of memory error: {e}")
            results_dict = None
            release_gpu_memory(gpu_name=params['device'])
        else:
            LOGGER.error(f"Runtime error during training: {e}")
            raise
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred during training: {e}")
        raise
    
    release_gpu_memory(gpu_name=params['device'])
    
    return results_dict
