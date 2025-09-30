""" Copyright (c) 2023, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using multiple processes.
"""
import os

from typing import Dict, Any, List
import numpy as np
from cnn import train
from util import init_log, load_yaml
from multi_head.task import SimpleNeuroEvolutionTask
from cnn import input
import time
import torch


class EvalPopulationTF(object):
    """
    Evaluate a population using multiple processes.

    This class is designed to distribute the evaluation of a population of models
    using multiple processes.
    
    Parameters
    ----------
    params : dict
        A dictionary containing parameters for the evaluation process.
    fn_dict : dict
        A dictionary containing definitions of the functions.
    log_level : str, optional
        The logging level for the internal logger (default is 'INFO').

    Attributes
    ----------
    train_params : dict
        Parameters for the training and evaluation process.
    fn_dict : dict
        Definitions of the functions used in the evaluation.
    timeout : int
        Timeout value for the Dask operations.
    logger : logger
        Internal logger for logging messages.
    gpus : list
        List of GPU devices available for evaluation.
    client : Client
        Dask client for managing the distributed computation.

    Methods
    -------
    __call__(decoded_params, decoded_nets, generation)
        Perform the evaluation of the population.
    
    """
    def __init__(self, params: dict, fn_dict: dict, log_level: str = 'INFO'):
        """
        Initialize the EvalPopulation object.

        Arguments:
        params : dict
            A dictionary containing parameters for the evaluation process.
        fn_dict : dict
            A dictionary containing definitions of the functions.
        log_level : str, optional
            The logging level for the internal logger (default is 'INFO').
        """
        
        self.train_params = params
        self.fn_dict = fn_dict
        self.logger = init_log(log_level, name=__name__)
        self.gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        self.loader = input.GenericDataLoader(params=self.train_params)
        # mp.set_start_method('spawn') # This is necessary for the multiprocessing to work on Windows
        self.logger.info(f"Evaluation process initialized with {len(self.gpus)} GPUs")        
        
    def __call__(self, decoded_params: list, decoded_nets: list, generation: int):
        """
        Evaluate the population.

        Parameters
        ----------
        decoded_params : list
            List of dictionaries containing the parameters for each model.
        decoded_nets : list
            List of lists containing the network architectures for each model.
        generation : int
            The generation number for tracking purposes.

        Returns
        -------
        np.ndarray
            An array containing the evaluations for each model.

        Raises
        ------
        TimeoutError
            If the Dask operations exceed the specified timeout.
        """
        pop_size = len(decoded_nets)
        
        if len(decoded_params) != 0:
            pop_size = len(decoded_params)        
        
        selected_thread = 0
        fitness_to_eval = [float('-inf') for x in range(pop_size)]
        print("\n")
        self.logger.info(f"Starting the Generation {generation} with {pop_size} individuals")
        evol_time_start = time.perf_counter()

        
        dataset_info = load_yaml(os.path.join(self.train_params['data_path'], 'data_info.txt'))

        train_loader = None
        val_loader = None
        self.run_individuals(generation,
                             self.train_params,
                             self.fn_dict,
                             train_loader,
                             val_loader,
                             dataset_info,
                             selected_thread,
                             decoded_nets,
                             decoded_params,
                             fitness_to_eval
                           )
            
        evol_end = time.perf_counter()
        time_elapsed_min = (evol_end-evol_time_start)/60
        time_elapsed_sec = (evol_end-evol_time_start)%60
        self.logger.info(f"Time elapsed for {pop_size} individuals: {time_elapsed_min:.0f}m {time_elapsed_sec:.0f}s")

        return fitness_to_eval
            
            
    def run_individuals(self,
                        generation: int,
                        train_params,
                        fn_dict,
                        train_loader,
                        val_loader,
                        dataset_info = None,
                        selected_thread = 0,
                        decoded_nets = None,
                        decoded_params = None,
                        fitness_to_eval: List[float] = [None]):
        # TODO: implementar o get_dataset para o tensorflow
        train_FD = self.loader.get_dataset(train=True)
        cols_sensors = [col for col in list(train_FD.columns) if col not in self.train_params['cols_non_sensor']]
        task = SimpleNeuroEvolutionTask(
                dataframe_norm=train_FD,
                cols_non_sensor=self.train_params['cols_non_sensor'],
                cols_sensors=cols_sensors,
                model_path=self.train_params['model_path'],
                fitness_mode=self.train_params['fitness_mode'],
                log_file_path=self.train_params['log_file_path'],
                dropout=self.train_params['dropout'],
                # n_window=n_window,
                sequence_length=self.train_params['sequence_length'],
                n_channel=self.train_params['n_channel'],
                strides_len=self.train_params['strides_len'],
                n_outputs=self.train_params['n_outputs'],
                cross_val=self.train_params['cross_val'],
                k_value_fold=self.train_params['k_value_fold'],
                val_split=self.train_params['val_split'],
                max_epoch=self.train_params['max_epoch'],
                patience=self.train_params['patience'],
                bidirec=self.train_params['bidirec'],
                test_engine_idx=self.train_params['test_engine_idx'],
                stride=self.train_params['stride'],
                piecewise_lin_ref=self.train_params['piecewise_lin_ref'],
                batch_size=self.train_params['batch_size'],
                experiment=self.train_params['experiment']
                )   
        for idx, param in enumerate(decoded_params):
            if all(value == -1.0 for value in decoded_params[idx].values()):
                continue
            param['lstm_2'] = param['lstm_2'] if param['lstm_2'] < param['lstm_1'] else param['lstm_1']  # Saturation
            genotype = [param['window_length'], param['n_filters'], param['n_conv_layer'], param['lstm_1'], param['lstm_2']]
            genotype = [int(x) for x in genotype]

            fitness_to_eval_temp = task.evaluate(genotype=genotype)

            fitness_to_eval[idx] = - fitness_to_eval_temp[0]
            self.logger.info(f"Calculated fitness of individual {idx} on thread {selected_thread} with "
                            f"Best Metric: {round(fitness_to_eval[idx], 3)}")
        
        best_metric = max(fitness_to_eval)
        best_id = fitness_to_eval.index(best_metric)
        path_file = os.path.join(os.getcwd(), self.train_params['experiment_path'], f'{generation}_{best_id}')
        if not os.path.exists(path_file):
            os.makedirs(path_file)
        path_file_txt = os.path.join(path_file, f'{generation}_{best_id}.txt')
        with open(path_file_txt, 'w') as file:
            file.write(f'Best Metric: {round(best_metric, 3)}\n')
