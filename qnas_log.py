"""
O objetivo desse código é auxliar na leitura dos arquivos de log gerados pelo
algoritmo Q-NAS
"""
from typing import List, Any, Dict, Tuple
import numpy as np
import copy
import hashlib
import constants.default_names as names
import util
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hashlib
import pandas as pd
from datetime import datetime
from constants import const_time


class LogParamsEvolution:
    """
    Handle the YAML log containing the function list and parameter evolution
    metadata used by QNAS.

    Parameters
    ----------
    log_params_evolution_path : str
        Path to the YAML file that stores the log of parameter evolution.

    Attributes
    ----------
    log_params_evolution_path : str
        Path to the YAML file.
    log_params_evolution : dict
        Parsed YAML content with QNAS metadata.
    """
    def __init__(self,
                 log_params_evolution_path: str):
        self.log_params_evolution_path = log_params_evolution_path
        self.log_params_evolution = util.load_yaml(log_params_evolution_path)
    
    def get_fn_list(self) -> List[str]:
        """
        Retrieve the list of functions used in QNAS from the log.

        Returns
        -------
        List[str]
            List of function names registered in the QNAS log.
        """
        return self.log_params_evolution[names.QNAS][names.FN_LIST]
    
    def get_initial_probs(self) -> List[float]:
        return self.log_params_evolution[names.QNAS][names.INITIAL_PROBS]
    
    def get_fn_dict(self) -> Dict[str, Any]:
        return self.log_params_evolution[names.QNAS][names.FN_DICT]


class QNASPalette:
    """
    Handles color palette generation for QNAS visualizations.

    Uses a fixed base color palette (from QNAS original style). If the number of
    functions exceeds the base palette size, additional colors are generated
    automatically from a smooth colormap in the same visual style.
    """

    # 🎨 Paleta base (inspirada no estilo original)
    _BASE_COLORS = [
        '#4C72B0',  # azul
        '#C44E52',  # vermelho
        '#DD8452',  # laranja
        '#E15759',  # coral
        '#EDC948',  # amarelo
        '#4E5F56',  # verde-acinzentado
        '#55A868',  # verde
        '#4BACC6',  # azul-claro
        '#8172B2',  # roxo
        '#3E2F5B',  # violeta escuro
    ]

    def __init__(self,
                 log_params_evolution: LogParamsEvolution,
                 colormap_fallback: str = "Spectral"):
        """
        Parameters
        ----------
        log_params_evolution : LogParamsEvolution
            Object containing metadata and function list.
        colormap_fallback : str
            Matplotlib colormap name for generating extra colors beyond the base set.
        """
        self.fn_list = log_params_evolution.get_fn_list()
        self.colormap_fallback = colormap_fallback
        self.palette = self._build_palette()

    def _generate_extra_colors(self, n_extra: int) -> list:
        """Generate extra colors beyond the base palette using a smooth colormap."""
        cmap = plt.get_cmap(self.colormap_fallback)
        # gera tons distribuídos igualmente no colormap
        extra_colors = [mcolors.to_hex(cmap(i / (n_extra - 1))) for i in range(n_extra)]
        return extra_colors

    def _build_palette(self) -> Dict[str, str]:
        """Combine base colors with generated ones if necessary."""
        n_funcs = len(self.fn_list)
        n_base = len(self._BASE_COLORS)

        if n_funcs <= n_base:
            all_colors = self._BASE_COLORS[:n_funcs]
        else:
            n_extra = n_funcs - n_base
            extra_colors = self._generate_extra_colors(n_extra)
            all_colors = self._BASE_COLORS + extra_colors

        palette = {fn: color for fn, color in zip(self.fn_list, all_colors)}
        return palette

    def get_palette(self) -> Dict[str, str]:
        """Return the full color palette mapping."""
        return self.palette


class DataQNASPKL:
    """
    Handle the QNAS population data loaded from a pickle file.

    Parameters
    ----------
    file_path : str
        Path to the pickle file containing QNAS data.

    Attributes
    ----------
    file_path : str
        Path to the pickle file.
    data_qnas : dict
        Parsed pickle content with QNAS generations and populations.
    """
    def __init__(self,
                 file_path: str):
        self.file_path = file_path
        self.data_qnas = util.load_pkl(file_path)
    
    def get_data_qnas(self) -> Dict[int, Dict[str, Any]]:
        """
        Get the population data across all generations.

        Returns
        -------
        Dict[int, Dict[str, Any]]
            Dictionary indexed by generation number, where each value contains:
            - names.CANDIDATE_PARAMS_POP : population of parameters
            - names.CANDIDATE_NET_POP : population of networks
        """
        return self.data_qnas

    def get_generations(self) -> List[int]:
        """Return a list of all generation indices."""
        return list(self.data_qnas.keys())

    def get_runtime(self) -> float:
        """
        Get the total runtime of the QNAS execution.

        Returns
        -------
        float
            Total runtime in seconds.
        """
        start_time = self.data_qnas[0][names.TIME]
        last_key = list(self.data_qnas.keys())[-1]
        end_time = self.data_qnas[last_key][names.TIME]

        start_time = datetime.strptime(start_time, const_time.DATE_FORMAT)
        end_time = datetime.strptime(end_time, const_time.DATE_FORMAT)

        time_difference = end_time - start_time

        hours = time_difference.total_seconds() / const_time.HOUR_TO_SECOND
        return hours

    def get_generation_fitness_metrics(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Return best per generation, best so far, average, and worst fitness values."""

        gen_best_values = []
        best_fitness_values = []
        avg_fitness_values = []
        worst_fitness_values = []

        for generation, data in self.data_qnas.items():
            fitness_values = [abs(f) for f in data[names.CANDIDATE_RAW_FITNESSES]]

            gen_best_values.append(min(fitness_values))
            best_fitness_values.append(abs(data[names.BEST_SO_FAR]))
            avg_fitness_values.append(sum(fitness_values) / len(fitness_values))
            worst_fitness_values.append(max(fitness_values))

        return gen_best_values, best_fitness_values, avg_fitness_values, worst_fitness_values
    
    def get_top_fitness_metrics(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Compute fitness statistics based on the elite (best) individuals across all generations.

        This function differs from `get_generation_fitness_metrics`, which analyzes 
        all candidate fitnesses within each generation. Here, only the best individuals 
        from each generation (stored in `names.FITNESSES`) are considered.
        """

        gen_best_values = []
        best_fitness_values = []
        avg_fitness_values = []
        worst_fitness_values = []

        for generation, data in self.data_qnas.items():
            fitness_values = [abs(f) for f in data[names.FITNESSES]]

            gen_best_values.append(min(fitness_values))
            best_fitness_values.append(abs(data[names.BEST_SO_FAR]))
            avg_fitness_values.append(sum(fitness_values) / len(fitness_values))
            worst_fitness_values.append(max(fitness_values))

        return gen_best_values, best_fitness_values, avg_fitness_values, worst_fitness_values

    def get_best_so_far_id(self):
        return self.data_qnas[next(reversed(self.data_qnas))][names.BEST_SO_FAR_ID]

class QNASLog:
    """
    Provide logging and analysis utilities for QNAS populations.

    Parameters
    ----------
    data_qnas : DataQNASPKL
        Object containing the QNAS population data.
    log_params_evolution : LogParamsEvolution
        Object containing metadata and function list.

    Attributes
    ----------
    data_qnas : DataQNASPKL
        Population data handler.
    log_params_evolution : LogParamsEvolution
        Log handler for function list and metadata.
    """
    def __init__(self,
                 data_qnas: DataQNASPKL,
                 log_params_evolution: LogParamsEvolution):
        self.data_qnas = data_qnas
        self.log_params_evolution = log_params_evolution

    @staticmethod
    def identify_search_type(params_pop: np.array,
                             net_pop: np.array) -> Tuple[bool, bool]:
        """
        Identify the type of search performed.

        Determines whether the search involves only parameters, only networks, 
        or both, based on the population arrays provided.

        Parameters
        ----------
        params_pop : np.ndarray
            Array containing the parameter population.
        net_pop : np.ndarray
            Array containing the network population.

        Returns
        -------
        Tuple[bool, bool]
            A tuple of two boolean values:
            - First element (bool): True if network search is present, False otherwise.
            - Second element (bool): True if parameter search is present, False otherwise.

        Raises
        ------
        ValueError
            If both populations are empty or if their lengths are inconsistent.
        """
        search_params = False
        search_net = False
        if len(params_pop) != 0 and len(net_pop) != 0:
            search_params = True
            search_net = True
            if len(params_pop) != len(net_pop):
                raise ValueError("params_pop and net_pop must have the same length.")
        elif len(params_pop) != 0 and len(net_pop) == 0:
            search_params = True
            search_net = False
        elif len(params_pop) == 0 and len(net_pop) != 0:
            search_params = False
            search_net = True
        else:
            search_params = False
            search_net = False
            raise ValueError("Both params_pop and net_pop are empty.")
        return search_net, search_params

    @staticmethod
    def get_individual_by_search_type(params_pop: np.array,
                                      net_pop: np.array,
                                      code_no_op: int) -> np.array:
        """
        Build individuals from the populations according to the search type.

        Depending on the search type, individuals can be:
        - Parameters only
        - Networks only
        - Tuples of (network, parameters)

        Parameters
        ----------
        params_pop : np.ndarray
            Array containing the parameter population.
        net_pop : np.ndarray
            Array containing the network population.
        code_no_op : int
            Identifier of the NO_OP operation, used to filter networks.

        Returns
        -------
        list
            List of individuals (parameters, networks, or tuples).

        Raises
        ------
        ValueError
            If no valid search type is identified.
        """
        search_net, search_params = QNASLog.identify_search_type(params_pop, net_pop)
        params_pop = params_pop.tolist()
        net_pop = net_pop.tolist()

        if search_net:
            cleaned_net_pop = []
            for i in range(len(net_pop)):
                # remove elementos iguais a code_no_op
                filtered_net = [x for x in net_pop[i] if x != code_no_op]
                cleaned_net_pop.append(filtered_net)
            net_pop = copy.deepcopy(cleaned_net_pop)

        if search_params and search_net:
            individuals = [(net, param) for net, param in zip(net_pop, params_pop)]
            return individuals
        elif search_params:
            return params_pop
        elif search_net:
            return net_pop
        else:
            raise ValueError("No search type identified.")

    @staticmethod
    def get_stable_hash(individual: np.array) -> str:
        """
        Generate a stable SHA-256 hash for an individual.

        The individual is converted into a comma-separated string,
        then hashed with SHA-256 to produce a deterministic identifier.

        Parameters
        ----------
        individual : list or tuple
            Representation of an individual (e.g., architecture as list of integers,
            or (network, parameters)).

        Returns
        -------
        str
            A 64-character hexadecimal hash.
        """
        net_str = ','.join(map(str, individual))
        return hashlib.sha256(net_str.encode()).hexdigest()

    @staticmethod
    def get_id_fn_dict(function: str, fn_list: List[str]) -> float:
        """
        Find the index of a function name inside the function list.

        Parameters
        ----------
        function : str
            Function name to search for.
        fn_list : List[str]
            List of available function names.

        Returns
        -------
        int or None
            The index of the function in the list, or None if not found.
        """
        for i, fn in enumerate(fn_list):
            if fn == function:
                return i
        return None

    @staticmethod
    def count_unique_individuals(params_pop: np.array,
                                 net_pop: np.array,
                                 fn_list: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Count unique individuals in a single generation.

        Parameters
        ----------
        params_pop : np.ndarray
            Array containing the parameter population.
        net_pop : np.ndarray
            Array containing the network population.
        fn_list : List[str]
            List of function names, used to identify NO_OP.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary where:
            - Keys (str): individual hashes
            - Values (dict):
                - names.AMOUNT_REPETITIONS: number of times the individual appears
                - names.INDIVIDUAL: the individual itself
        """
        code_no_op = QNASLog.get_id_fn_dict(names.NO_OP, fn_list)
        individuals = QNASLog.get_individual_by_search_type(params_pop, net_pop, code_no_op)
        repetition_dict = {}
        for individual in individuals:

            individual_hash = QNASLog.get_stable_hash(individual)

            if individual_hash not in repetition_dict:
                repetition_dict[individual_hash] = {
                    names.AMOUNT_REPETITIONS: 1,
                    names.INDIVIDUAL: individual
                }
            else:
                repetition_dict[individual_hash][names.AMOUNT_REPETITIONS] += 1
        return repetition_dict
    
    def count_unique_individuals_all_gens(self) -> Dict[str, Dict[str, Any]]:
        """
        Count unique individuals across all generations.

        Aggregates the counts from `count_unique_individuals` for each generation,
        summing repetitions of the same individual across generations.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary where:
            - Keys (str): individual hashes
            - Values (dict):
                - names.AMOUNT_REPETITIONS: total repetitions across all generations
                - names.INDIVIDUAL: the individual itself
        """
        repetition_dict = {}

        for gen_id, gen_data in self.data_qnas.get_data_qnas().items():
            params_pop = gen_data[names.CANDIDATE_PARAMS_POP]
            net_pop = gen_data[names.CANDIDATE_NET_POP]

            gen_counts = QNASLog.count_unique_individuals(params_pop,
                                                          net_pop,
                                                          self.log_params_evolution.get_fn_list())

            for hash, data in gen_counts.items():
                if hash not in repetition_dict:
                    repetition_dict[hash] = data
                else:
                    repetition_dict[hash][names.AMOUNT_REPETITIONS] += data[names.AMOUNT_REPETITIONS]

        return repetition_dict
    
    def count_unique_individuals_by_gens(self) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """
        Count unique individuals for each generation separately.

        Returns
        -------
        Dict[int, Dict[str, Dict[str, Any]]]
            Dictionary where:
            - Keys (int): generation indices
            - Values (dict): result of `count_unique_individuals` for that generation
                - Keys (str): individual hashes
                - Values (dict):
                    - names.AMOUNT_REPETITIONS: repetitions in that generation
                    - names.INDIVIDUAL: the individual itself
        """
        gens_dict = {}

        for gen_id, gen_data in self.data_qnas.get_data_qnas().items():
            params_pop = gen_data[names.CANDIDATE_PARAMS_POP]
            net_pop = gen_data[names.CANDIDATE_NET_POP]

            gen_counts = QNASLog.count_unique_individuals(params_pop,
                                                          net_pop,
                                                          self.log_params_evolution.get_fn_list())

            gens_dict[gen_id] = gen_counts

        return gens_dict

class TrainingParams:
    def __init__(self,
                 file_path: str):
        self.file_path = file_path
        self.training_params = util.load_yaml(file_path)
    
    def get_decoded_params(self) -> Dict[str, Any]:
        return self.training_params[names.DECODED_PARAMS]