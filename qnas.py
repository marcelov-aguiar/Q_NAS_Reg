""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Q-NAS algorithm class.
"""

import datetime
import os
from pickle import dump, HIGHEST_PROTOCOL

import numpy as np
import time
import hashlib
from population import QPopulationNetwork, QPopulationParams
from util import delete_old_dirs, init_log, load_pkl, calculate_time
from evaluation import EvalPopulation
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class EvalEntry:
    fitness: float
    penalty: float
    net_pop: np.array

@dataclass
class EvalEntryParams:
    fitness: float
    penalty: float
    params_pop: np.array


class QNAS(object):
    """ Quantum Inspired Neural Architecture Search """

    def __init__(self, eval_func: EvalPopulation, experiment_path, log_file, log_level, data_file):
        """ Initialize QNAS.

        Args:
            eval_func: function that will be used to evaluate individuals.
            experiment_path: (str) path to the folder where logs and models will be saved.
            log_file: (str) path to the file to keep logs.
            log_level: (str) one of "INFO", "DEBUG" or "NONE".
        """

        self.dtype = np.float64                 # Type of all arrays excluding fitnesses
        self.tolerance = 1.e-15                 # Tolerance to compare floating point

        self.best_so_far = -np.inf                  # Best fitness so far
        self.best_so_far_id = [0, 0]            # id = [generation, position in the population]
        self.current_best_id = [0, 0]
        self.current_gen = 0                    # Current generation number
        self.data_file = data_file
        self.eval_func = eval_func
        self.experiment_path = experiment_path
        self.fitnesses = None                   # TF calculates accuracy with float32 precision
        self.generations = None
        self.update_quantum_gen = None
        self.logger = init_log(log_level, name=__name__, file_path=log_file)
        self.penalties = None
        self.penalize_number = None
        self.random = 0.0
        self.raw_fitnesses = None
        self.reducing_fns_list = []
        self.replace_method = None
        self.save_data_freq = np.inf
        self.total_eval = 0
        self.early_stopping_counter = 0

        self.qpop_params = None
        self.qpop_net = None

        self.eval_history: Dict[str, EvalEntry] = {}
        self.eval_history_params: Dict[str, EvalEntryParams] = {}
        self.allow_duplicate_architectures = True
        self.code_no_op = None

    def initialize_qnas(self, num_quantum_ind, params_ranges, params_type, repetition, max_generations,
                        crossover_rate, update_quantum_gen, replace_method, fn_list,
                        initial_probs, update_quantum_rate, max_num_nodes, reducing_fns_list,
                        patience,early_stopping, save_data_freq=0, penalize_number=0, crossover_frequency = 5,
                        en_pop_crossover=False,pop_crossover_rate=0.25, pop_crossover_method='hux',
                        allow_duplicate_architectures=True):

        """ Initialize algorithm with several parameter values.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            params_ranges: {'parameter_name': [parameter_lower_limit, parameter_upper_limit]}.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            max_generations: (int) number of generations to run the evolution.
            crossover_rate: (float) crossover rate for numerical part of the chromosomes.
            update_quantum_gen: (int) the width of the quantum genes will be updated in a
                interval of *update_quantum_gen* generations.
            replace_method: (str) one of 'best' or 'elitism', indicating which method to
                substitute the population.
            fn_list: list of possible functions.
            initial_probs: list defining the initial probabilities for each function; if empty,
                the algorithm will give the same probability for each function.
            update_quantum_rate: (float) probability that a quantum gene will be updated,
                if using update_center() and/or update_width_decay().
            max_num_nodes: (int) initial number of nodes in the network to be evolved (the 
                classifier fc layer is always included).
            save_data_freq: generation frequency in which train loss and accuracy of the best
                model (of current generation) will be extracted from events.out.tfevents file
                and saved in a csv file.
            penalize_number: (int) defines the minimum number of reducing layers an individual
                can have without being penalized. The penalty is proportional to the number of
                exceeding reducing layers. If 0, no penalization will be applied.
            reducing_fns_list: (list) list of reducing functions (stride > 2) names.
            patience: (int) number of generations without improvement in the best fitness to
                stop the evolution.
            early_stopping: (bool) if True, the evolution will stop if the best fitness does not
                improve by at least 0.005 (0.5%) for *patience* generations.
            en_pop_crossover: (bool) if True, there will be crossover between the best individuals
                of the current and new populations.
            crossover_frequency: (int) frequency of crossover in the population of networks.
            pop_crossover_rate: (float) crossover rate for the population of networks, used to select
                the number of offspring to generate in the crossover [0 - 1].
            pop_crossover_method: (str) one of 'hux' or 'uniform', indicating the method to
                apply crossover in the population of networks.
        """

        self.generations = max_generations
        self.update_quantum_gen = update_quantum_gen
        self.replace_method = replace_method
        self.penalize_number = penalize_number
        self.patience = patience
        self.early_stopping = early_stopping
        self.en_pop_crossover = en_pop_crossover
        self.pop_crossover_rate = pop_crossover_rate
        self.crossover_frequency = crossover_frequency
        self.allow_duplicate_architectures = allow_duplicate_architectures
        if reducing_fns_list:
            self.penalties = np.zeros(shape=(num_quantum_ind * repetition))
            self.reducing_fns_list = [i for i in range(len(fn_list))
                                    if fn_list[i] in reducing_fns_list]

        if save_data_freq:
            self.save_data_freq = save_data_freq

        self.qpop_params = QPopulationParams(num_quantum_ind=num_quantum_ind,
                                            params_ranges=params_ranges,
                                            params_type=params_type,
                                            repetition=repetition,
                                            crossover_rate=crossover_rate,
                                            update_quantum_rate=update_quantum_rate)

        self.qpop_net = QPopulationNetwork(num_quantum_ind=num_quantum_ind,
                                            max_num_nodes=max_num_nodes,
                                            repetition=repetition,
                                            update_quantum_rate=update_quantum_rate,
                                            fn_list=fn_list,
                                            initial_probs=initial_probs, 
                                            crossover_method=pop_crossover_method)
                                            
        self.code_no_op = self.get_id_fn_dict("no_op")

    def replace_pop(self, new_pop_params, new_pop_net, new_fitnesses, raw_fitnesses):
        """ Replace the individuals of old population using one of two methods: elitism or
            replace the worst. In *elitism*, only the best individual of the old population is
            maintained, while all the others are replaced by the new population. In *best*,
            only the best of the union of both populations individuals are kept.

        Args:
            new_pop_params: float ndarray representing a classical population of parameters.
            new_pop_net: int ndarray representing a classical population of networks.
            new_fitnesses: float numpy array representing the fitness of each individual in
                *new_pop*.
            raw_fitnesses: float numpy array representing the fitness of each individual in
                *new_pop* before the penalization method. Note that, if no penalization method
                is applied, *raw_fitnesses* = *new_fitnesses*.
        """

        if self.current_gen == 0:
            # In the 1st generation, the current population is the one that was just generated.
            self.qpop_params.current_pop = new_pop_params
            self.qpop_net.current_pop = new_pop_net

            self.fitnesses = new_fitnesses
            self.raw_fitnesses = raw_fitnesses
            self.update_best_id(new_fitnesses)
        else:
            # Checking if the best so far individual has changed in the current generation
            self.update_best_id(new_fitnesses)

            if self.replace_method == 'elitism':
                select_new = range(new_fitnesses.shape[0] - 1)
                new_fitnesses, raw_fitnesses, new_pop_params, \
                    new_pop_net = self.order_pop(new_fitnesses,
                                                new_pop_params,
                                                new_pop_net,
                                                select_new)
                selected = range(1)
            elif self.replace_method == 'best':
                selected = range(self.fitnesses.shape[0])
                
            # Concatenate populations
            self.fitnesses = np.concatenate((self.fitnesses[selected], new_fitnesses))
            self.raw_fitnesses = np.concatenate((self.raw_fitnesses[selected], raw_fitnesses))
            self.qpop_params.current_pop = np.concatenate(
                    (self.qpop_params.current_pop[selected], new_pop_params))
            self.qpop_net.current_pop = np.concatenate(
                    (self.qpop_net.current_pop[selected], new_pop_net))
        
        ## TODO: Here we have the the last and new population Multi objective operation
        
            
        # Order the population based on fitness
        num_classic = self.qpop_params.num_ind * self.qpop_params.repetition
        self.fitnesses, self.raw_fitnesses, self.qpop_params.current_pop, \
            self.qpop_net.current_pop = self.order_pop(self.fitnesses,
                                                        self.raw_fitnesses,
                                                        self.qpop_params.current_pop,
                                                        self.qpop_net.current_pop,
                                                        selection=range(num_classic))       
        
        self.best_so_far = self.fitnesses[0]

    @staticmethod
    def order_pop(fitnesses, raw_fitnesses, pop_params, pop_net, selection=None):
        """ Order the population based on *fitnesses*.

        Args:
            fitnesses: ndarray with fitnesses values.
            raw_fitnesses: float ndarray representing the fitness of each individual before the
                penalization method.
            pop_params: ndarray with population of parameters.
            pop_net: ndarray with population of networks.
            selection: range to select elements from the population.

        Returns:
            ordered population and fitnesses.
        """

        if selection is None:
            selection = range(fitnesses.shape[0])
        idx = np.argsort(fitnesses)[::-1]
        pop_params = pop_params[idx][selection]
        pop_net = pop_net[idx][selection]
        fitnesses = fitnesses[idx][selection]
        raw_fitnesses = raw_fitnesses[idx][selection]

        return fitnesses, raw_fitnesses, pop_params, pop_net

    def update_best_id(self, new_fitnesses):
        """ Checks if the new population contains the best individual so far and updates
            *self.best_so_far_id*.

        Args:
            new_fitnesses: float numpy array representing the fitness of each individual in
                *new_pop*.
        """

        idx = np.argsort(new_fitnesses)[::-1]
        self.current_best_id = [self.current_gen, int(idx[0])]
        if new_fitnesses[idx[0]] > self.best_so_far:
            self.best_so_far_id = self.current_best_id

    def generate_classical(self):
        """ Generate a specific number of classical individuals from the observation of quantum
            individuals. This number is equal to (*num_ind* x *repetition*). The new classic
            individuals will be evaluated and ordered according to their fitness values.
        """

        # Generate distance for crossover and quantum updates every generation
        self.random = np.random.rand()

        # Generate classical pop for hyperparameters
        new_pop_params = self.qpop_params.generate_classical()
        if self.current_gen > 0:
            new_pop_params = self.qpop_params.classic_crossover(new_pop=new_pop_params,
                                                                distance=self.random)

        new_pop_params = self.qpop_params.convert_params_to_int(new_pop=new_pop_params)

        # Generate classical pop for network structure
        new_pop_net = self.qpop_net.generate_classical()
        
        ###---Cross-over for network structure---###
        if self.current_gen > 0 and self.en_pop_crossover:
            if self.current_gen % self.crossover_frequency == 0:
                num_offspring = int(len(new_pop_net) * self.pop_crossover_rate)
                best_current_pop = self.qpop_net.current_pop[:num_offspring]
                new_pop_net[:num_offspring] = self.qpop_net.apply_crossover(best_current_pop, new_pop_net[:num_offspring])
        
        self.logger.info("new population created =%s", new_pop_net)
        # Evaluate population
        new_fitnesses, raw_fitnesses = self.eval_pop(new_pop_params, new_pop_net)

        self.replace_pop(new_pop_params, new_pop_net, new_fitnesses, raw_fitnesses)

    def decode_pop(self, pop_params, pop_net):
        """ Decode a population of parameters and networks.

        Args:
            pop_params: float numpy array with a classic population of hyperparameters.
            pop_net: int numpy array with a classic population of networks.

        Returns:
            list of decoded params and list of decoded networks.
        """

        num_individuals = pop_net.shape[0]

        decoded_params = [None] * num_individuals
        decoded_nets = [None] * num_individuals

        for i in range(num_individuals):
            decoded_params[i] = self.qpop_params.chromosome.decode(pop_params[i])
            decoded_nets[i] = self.qpop_net.chromosome.decode(pop_net[i, :])

        return decoded_params, decoded_nets

    def eval_pop(self, pop_params, pop_net):
        """ Decode and evaluate a population of networks and hyperparameters.

        Args:
            pop_params: float numpy array with a classic population of hyperparameters.
            pop_net: int numpy array with a classic population of networks.

        Returns:
            fitnesses with penalization and without penalization; note that they are equal if
            no penalization is applied.
        """
        pop_net_bkp = pop_net.copy()

        pop_params_bkp = pop_params.copy()

        if not self.allow_duplicate_architectures:
            # Remove duplicate networks from the population
            pop_net = self.__get_unique_pop(pop_net)

            
            if pop_params_bkp.size != 0:
                # Remove duplicate params from the population
                pop_params = self.__get_unique_pop_params(pop_params)

        decoded_params, decoded_nets = self.decode_pop(pop_params, pop_net)

        self.logger.info('Evaluating new population ...')
        fitnesses = self.eval_func(decoded_params, decoded_nets, generation=self.current_gen)
        penalized_fitnesses = np.copy(fitnesses)

        if self.penalize_number:
            penalties = self.get_penalties(pop_net)
            penalized_fitnesses -= penalties

        # Update the total evaluation counter
        self.total_eval = self.total_eval + np.size(pop_params, axis=0)

        if not self.allow_duplicate_architectures:
            self.__save_unique_fitnesses(pop_net, fitnesses, penalized_fitnesses)

            if pop_params_bkp.size != 0:
                self.__save_unique_fitnesses_params(pop_params, fitnesses, penalized_fitnesses)

            penalized_fitnesses, fitnesses= self.__get_eval_history(pop_net_bkp)

            if pop_params_bkp.size != 0:
                penalized_fitnesses, fitnesses= self.__get_eval_history_params(pop_params_bkp)

        return penalized_fitnesses, fitnesses

    def __save_unique_fitnesses(self,
                                pop_net: np.array,
                                fitnesses: np.array,
                                penalized_fitnesses: np.array):
        """ Save unique fitnesses of the population of networks.

        Args:
            pop_net: int numpy array with a classic population of networks.
        """
        for net, fitness, penalized_fitness in zip(pop_net, fitnesses, penalized_fitnesses):
            hash = self.get_stable_hash(net)
            if hash not in self.eval_history.keys() and np.all(net != -1):
                hash = self.get_stable_hash(net)
                self.eval_history[hash] = EvalEntry(fitness=fitness, penalty=penalized_fitness, net_pop=net)

    def __save_unique_fitnesses_params(self,
                                       pop_params: np.array,
                                       fitnesses: np.array,
                                       penalized_fitnesses: np.array):
        """ Save unique fitnesses of the population of networks.

        Args:
            pop_net: int numpy array with a classic population of networks.
        """
        for param, fitness, penalized_fitness in zip(pop_params, fitnesses, penalized_fitnesses):
            hash = self.get_stable_hash(param)
            if hash not in self.eval_history_params.keys() and np.all(param != -1):
                hash = self.get_stable_hash(param)
                self.eval_history_params[hash] = EvalEntryParams(fitness=fitness, penalty=penalized_fitness, params_pop=param)

    def __get_unique_pop(self, pop_net: np.array):
        """ Get unique individuals from the population.

        Args:
            pop_net: int numpy array with a classic population of networks.

        Returns:
            unique population of networks.
        """
        unique_pop_net = []
        MARKER = [-1] * len(pop_net[0])
        for net in pop_net:
            hash = self.get_stable_hash(net)
            if hash not in self.eval_history.keys() and np.all(net != -1):
                unique_pop_net.append(net)
            else:
                unique_pop_net.append(MARKER)

        return np.array(unique_pop_net, dtype=np.int32)

    def __get_unique_pop_params(self, pop_params: np.array):
        """ Get unique individuals from the population.

        Args:
            pop_params: int numpy array with a classic population of networks.

        Returns:
            unique population of networks.
        """
        unique_pop_params = []
        MARKER = [-1] * len(pop_params[0])
        for net in pop_params:
            hash = self.get_stable_hash(net)
            if hash not in self.eval_history_params.keys() and np.all(net != -1):
                unique_pop_params.append(net)
            else:
                unique_pop_params.append(MARKER)

        return np.array(unique_pop_params)

    def __get_eval_history(self, pop_net_bkp: np.array) -> Tuple[np.array, np.array]:
        """ Get the evaluation history of the population of networks.

        Returns:
            fitnesses, penalized_fitnesses and population of networks.
        """
        fitnesses_temp = []
        penalized_fitnesses_temp = []
        for net in pop_net_bkp:
            hash = self.get_stable_hash(net)
            fitnesses_temp.append(self.eval_history[hash].fitness)
            penalized_fitnesses_temp.append(self.eval_history[hash].penalty)

        fitnesses = np.array(fitnesses_temp, dtype=np.float64)
        penalized_fitnesses = np.array(penalized_fitnesses_temp, dtype=np.float64)

        return penalized_fitnesses, fitnesses

    def __get_eval_history_params(self, pop_params_bkp: np.array) -> Tuple[np.array, np.array]:
        """ Get the evaluation history of the population of networks.

        Returns:
            fitnesses, penalized_fitnesses and population of networks.
        """
        fitnesses_temp = []
        penalized_fitnesses_temp = []
        for net in pop_params_bkp:
            hash = self.get_stable_hash(net)
            fitnesses_temp.append(self.eval_history_params[hash].fitness)
            penalized_fitnesses_temp.append(self.eval_history_params[hash].penalty)

        fitnesses = np.array(fitnesses_temp, dtype=np.float64)
        penalized_fitnesses = np.array(penalized_fitnesses_temp, dtype=np.float64)

        return penalized_fitnesses, fitnesses 

    def get_id_fn_dict(self, function: str) -> float:
        """ 
        Searches for the ID of the function in the function list (`fn_list`).
        
        Notes
        -----
        The ID corresponds to the position of the function in the list.
        
        Parameters
        ----------
        function : str
            The name of the function whose ID you want to find.
        
        Returns
        -------
        int or None
            The ID (index) of the function in the list, or None if not found.
        """
        for i, fn in enumerate(self.qpop_net.chromosome.fn_list):
            if fn == function:
                return i
        return None
    
    def get_stable_hash(self, net_pop: np.array) -> str:
        """
        Generates a stable SHA-256 hash from an architecture represented as a list of integers.

        The function converts the list of integers (`net_pop`) into a comma-separated string,
        then applies the SHA-256 hashing function to produce a unique and deterministic identifier.
        This hash can be used to check whether a given architecture has already been evaluated.

        Parameters
        ----------
        net_pop : List[int]
            List of integers representing a neural architecture (e.g., operations per layer).

        Returns
        -------
        str
            A 64-character hexadecimal hash generated from the architecture representation.
        """
        filtered_net = [x for x in net_pop if x != self.code_no_op]
        net_str = ','.join(map(str, filtered_net))
        return hashlib.sha256(net_str.encode()).hexdigest()

    def get_penalties(self, pop_net, penalty_factor=0.01):
        """ Penalize individuals with more than *self.penalize_number* reducing layers. The
            penalty is proportional (default factor of 1%) to the number of exceeding layers.

        Args:
            pop_net: ndarray representing the encoded population of networks (just evaluated).
            penalty_factor: (float) the factor to multiply the penalties for all networks.

        Returns:
            penalties for each network in pop_net.
        """

        penalties = np.zeros(shape=pop_net.shape[0])

        for i, net in enumerate(pop_net):
            unique, counts = np.unique(net, return_counts=True)
            reducing_fns_count = np.sum([counts[i] for i in range(len(unique))
                                        if unique[i] in self.reducing_fns_list])
            # Penalize individual only if number of reducing layers exceed the maximum allowed
            if reducing_fns_count > self.penalize_number:
                penalties[i] = reducing_fns_count - self.penalize_number

        penalties = penalty_factor * penalties

        return penalties

    def log_data(self):
        """ Log QNAS evolution info into a log file. """

        np.set_printoptions(precision=4)

        self.logger.info(f'New generation finished running!\n\n'
                        f'- Generation: {self.current_gen}\n'
                        f'- Best so far: {self.best_so_far_id} --> {self.best_so_far:.5f}\n'
                        f'- Fitnesses: {self.fitnesses}\n'
                        f'- Fitnesses without penalties: {self.raw_fitnesses}\n'
                        f'- Net Population: {self.qpop_net.current_pop}\n'
                        f'- Parameter lower: {self.qpop_params.lower}\n'
                        f'- Parameter upper: {self.qpop_params.upper}\n'
                        f'- Parameter population: {self.qpop_params.current_pop}\n')

    def save_data(self):
        """ Save QNAS data in a pickle file for logging and reloading purposes, including
            chromosomes, generation number, evaluation score and number of evaluations. Note
            that the data in the file is loaded and updated with the current generation, so that
            we keep track of the entire evolutionary process.
        """

        if self.current_gen == 0:
            data = dict()
        else:
            data = load_pkl(self.data_file)

        data[self.current_gen] = {'time': str(datetime.datetime.now()),
                                'total_eval': self.total_eval,
                                'best_so_far': self.best_so_far,
                                'best_so_far_id': self.best_so_far_id,
                                'fitnesses': self.fitnesses,
                                'raw_fitnesses': self.raw_fitnesses,
                                'lower': self.qpop_params.lower,
                                'upper': self.qpop_params.upper,
                                'params_pop': self.qpop_params.current_pop,
                                'net_probs': self.qpop_net.probabilities,
                                'num_net_nodes': self.qpop_net.chromosome.num_genes,
                                'net_pop': self.qpop_net.current_pop}

        self.dump_pkl_data(data)

    def dump_pkl_data(self, new_data):
        """ Saves *new_data* into *self.data_file* pickle file.

        Args:
            new_data: dict containing data to save.
        """

        with open(self.data_file, 'wb') as f:
            dump(new_data, f, protocol=HIGHEST_PROTOCOL)

    def load_qnas_data(self, file_path):
        """ Read pkl data in *file_path* and load its information to current QNAS. It also saves
            its info into the new pkl data file *self.data_file*.

        Args:
            file_path: (str) path to the pkl data file.
        """

        log_data = load_pkl(file_path)

        if not os.path.exists(self.data_file):
            self.dump_pkl_data(log_data)

        generation = max(log_data.keys())
        log_data = log_data[generation]

        self.current_gen = generation
        self.total_eval = log_data['total_eval']
        self.best_so_far = log_data['best_so_far']
        self.best_so_far_id = log_data['best_so_far_id']
        self.qpop_net.chromosome.set_num_genes(log_data['num_net_nodes'])

        self.fitnesses = log_data['fitnesses']
        self.raw_fitnesses = log_data['raw_fitnesses']
        self.qpop_params.lower = log_data['lower']
        self.qpop_params.upper = log_data['upper']
        self.qpop_net.probabilities = log_data['net_probs']

        self.qpop_params.current_pop = log_data['params_pop']
        self.qpop_net.current_pop = log_data['net_pop']
        
    def check_early_stopping(self):
        """
        Compute the early stopping of the evolution. If the best fitness does not improve 
        by at least 0.005 (0.5%) for `patience` generations, the evolution stops.
        """
        if self.current_gen > 1:
            #TODO: Check whether to use the absolute value or multiply by the sign
            improvement = np.abs((self.best_so_far - self.last_best_so_far) / self.last_best_so_far)
            if improvement > 0.005:
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            self.logger.info(f"Early stopping counter: {self.early_stopping_counter}")
            if self.early_stopping_counter >= self.patience:
                self.logger.info(f"Early stopping at generation {self.current_gen}!")
                return True

        self.last_best_so_far = self.best_so_far
        return False

    def update_quantum(self):
        """ Update quantum populations of networks and hyperparameters. """

        if np.remainder(self.current_gen,
                        self.update_quantum_gen) == 0 and self.current_gen > 0:

            self.qpop_params.update_quantum(intensity=self.random)
            self.qpop_net.update_quantum(intensity=self.random)
    
    def go_next_gen(self):
        """ Go to the next generation --> update quantum genes, log data, delete unnecessary
            training files and update generation counter.
        """

        self.update_quantum()

        self.save_data()
        self.log_data()
        #self.save_train_data()

        # Remove Tensorflow models files
        delete_old_dirs(self.experiment_path, keep_best=True,
                        best_id=f'{self.best_so_far_id[0]}_{self.best_so_far_id[1]}')
        self.current_gen += 1

    def evolve(self):
        """ Run the evolution. """
        start_evolution = time.time()
        max_generations = self.generations
        #print(f"early stopping enable?: {self.early_stopping}")
        #print(f"population crossover enable?: {self.en_pop_crossover}")
        
        # Update maximum number of generations if continue previous evolution process
        if self.current_gen > 0:
            max_generations += self.current_gen + 1
            # Increment current generation, as in the log file we have the completed generations
            self.current_gen += 1

        while self.current_gen < max_generations:
            # Estimate time to finish the evolution
            if self.current_gen % 5 == 0 and self.current_gen > 0:
                int_time = time.time()
                int_hours, int_mins, est_hours, est_mins = calculate_time(start_evolution,int_time,self.current_gen, max_generations, end_evol=False)
                self.logger.info(f"Current evolution time at generation {self.current_gen}: {int_hours} hours and {int_mins} mins")
                self.logger.info(f"Estimated time to finish the evolution: {est_hours} hours and {est_mins} mins")

            self.generate_classical()
            self.go_next_gen()
            
            if self.early_stopping and self.check_early_stopping(): break
        
        end_evolution = time.time()
        evolution_hours, evolution_minutes = calculate_time(start_evolution,end_evolution)
        self.logger.info(f"Total evolution time: {evolution_hours} hours and {evolution_minutes} minutes")