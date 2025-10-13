"""
Sobre este experimento:
Esse experimento é para executar um arquivo config passado pelo argumento --config via terminal
A ideia é ter um código externo que chame esse código da evolução via terminal
O run_evolution é semelhante ao run_evolution_turbofan_FD001_v0.py!!!!
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import auxiliar
import qnas
import qnas_config as cfg
import evaluation
from util import check_files, init_log, download_dataset, load_yaml


def main(**args):
    
    logger = init_log(args['log_level'], name=__name__)

    if not os.path.exists(args['experiment_path']):
        logger.info(f"Creating {args['experiment_path']} ...")
        os.makedirs(args['experiment_path'])

    # Evolution or continue previous evolution
    if not args['continue_path']:
        phase = 'evolution'
    else:
        phase = 'continue_evolution'
        logger.info(f"Continue evolution from: {args['continue_path']}. Checking files ...")
        check_files(args['continue_path'])

    logger.info(f"Getting parameters from {args['config_file']} ...")
    config = cfg.ConfigParameters(args, phase=phase)
    config.get_parameters()
    logger.info(f"Saving parameters for {config.phase} phase ...")
    config.save_params_logfile()
    
    if config.train_spec['mixed_precision']:
        logger.info(f"Using mixed precision training ...")
        
    # Download dataset
    dataset_status = download_dataset(params=config.train_spec)
    status_message = "Dataset is already downloaded." if dataset_status else "Dataset downloaded successfully."
    logger.info(status_message)
    
    eval_pop = evaluation.EvalPopulation(params=config.train_spec,
                                                fn_dict=config.fn_dict,
                                                log_level=config.train_spec['log_level'])
    
    qnas_cnn = qnas.QNAS(eval_pop, config.train_spec['experiment_path'],
                        log_file=config.files_spec['log_file'],
                        log_level=config.train_spec['log_level'],
                        data_file=config.files_spec['data_file'])

    qnas_cnn.initialize_qnas(**config.QNAS_spec)
    
    # Start evolution
    logger.info(f"Starting evolution ...")
    qnas_cnn.evolve()
    logger.info(f"Evolution finished.")

if __name__ == '__main__':

	# How to run: python run_evolution.py --config FD002/config_files/config_turbofan_FD001_v100.txt
    import argparse
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file name (inside config_files/)")
    args = parser.parse_args()

    # ["FD002", "config_files", "config_turbofan_FD001_v100.txt"]
    root_folder, config_file_folder, config_file_name = args.config.split("/")

    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), root_folder)

    config_path = os.path.join(base_path, config_file_folder, config_file_name)
    config_file = load_yaml(config_path)

    dataset = config_file['train']['dataset']
    exp_path_base = config_file['train']['exp_path_base']
    exp = config_file['train']['exp']
    file_extension = config_file['train']['file_extension']
    dataset_path = os.path.join(base_path, 'data')
    repeat = config_file['train']['repeat']

    exp_path = os.path.join(base_path, exp_path_base, f"{exp}_repeat_{repeat}")

    arguments = {
        "experiment_path": exp_path,
        "data_path": dataset_path,
        "dataset": dataset,
        "config_file": config_path,
        "continue_path": '',
        "fitness_metric": config_file['train']['fitness_metric'],
        "log_level": config_file['train']['log_level'],
        "network_config": config_file['train']['network_config'],
        "optimizer": config_file['train']['optimizer'],
        "data_augmentation": config_file['train']['data_augmentation'],
        "early_stopping": config_file['train']['early_stopping'],
        "en_pop_crossover": config_file['train']['en_pop_crossover'],
        "save_checkpoints_epochs": config_file['train']['save_checkpoints_epochs'],
        "limit_data_value": config_file['train']['limit_data_value'],
        "network_gap": config_file['train']['network_gap'],
        "log_file_path": '',
        "model_path": ''
    }

    main(**arguments)