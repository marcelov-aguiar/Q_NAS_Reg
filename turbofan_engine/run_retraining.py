"""
Sobre este experimento:
Esse experimento é para executar um arquivo config passado pelo argumento --config via terminal
A ideia é ter um código externo que chame esse código do retreinamento da arquitetura
encontrada na evolućão via terminal
O run_retraining é semelhante ao run_retrain_turbofan_FD002_v0.py!!!!
"""

import argparse
import os
import qnas_config as cfg
from util import check_files, init_log, save_results_file, load_yaml
from cnn import input
from cnn import train_detailed as train
import time

DEBUG = True

def main(**args):

    logger = init_log(args['log_level'], name=__name__)
    # Check if *experiment_path* contains all the necessary files to retrain an evolved model
    experiment_path = args['experiment_path']
    check_files(args['experiment_path'])
    
    config_code = args['config_code']

    # Get all parameters
    logger.info(f"Retraining evolved model {experiment_path} ...")
    config = cfg.ConfigParameters(args, phase='retrain')
    config.get_parameters()
    config.load_evolved_data(experiment_path=experiment_path)

    # Load data
    logger.info(f"Loading data ...")
    data_loader = input.GenericDataLoader(params=config.train_spec)
    train_loader, val_loader = data_loader.get_loader(individual=config.evolved_params['net'],
                                                      pin_memory_device=args['device'])
    test_loader = data_loader.get_loader(individual=config.evolved_params['net'],
                                         for_train=False, pin_memory_device=args['device'])
    
    output_dict = {}
    # Retrain model for the number of repetitions
    for i in range(1, args['num_repetitions']+1):

        config.train_spec['experiment_path'] = os.path.join(experiment_path, f"retrain_{config_code}_{i}")

        logger.info(f"Retraining {experiment_path} repetition {i} ...")
        start_time = time.perf_counter()
        results_dict = train.train_and_eval(params=config.train_spec,
                                            fn_dict=config.fn_dict,
                                            net_list=config.evolved_params['net'],
                                            best_individual_info=config.best_individual_info,
                                            train_loader=train_loader,
                                            val_loader=val_loader,
                                            test_loader=test_loader)    
        config.train_spec['experiment_path'] = os.path.join(experiment_path, f"retrain_{config_code}_{i+1}")
        
        end_time = time.perf_counter()
        hours, rem = divmod(end_time - start_time, 3600)
        mins, _ = divmod(rem, 60)
        logger.info(f"Retraining {experiment_path} repetition {i} finished in {int(hours):02d}h:{int(mins):02d}m.")
        
        results_dict["time"] = f"{int(hours):02d}h:{int(mins):02d}m"
        output_dict[f"{config.train_spec['lr_scheduler']}_{config_code}_retrain_{i}"] = results_dict

        # Save results
        logger.info(f"Saving results ...")
        if config.train_spec['lr_scheduler']  != "None":
            file_name = f"retrain_results_{config_code}_{i}_{config.train_spec['lr_scheduler']}.txt"
            save_results_file(out_path=experiment_path, results_dict=output_dict, file_name=file_name)
        else:
            save_results_file(out_path=experiment_path, results_dict=output_dict, file_name=f"retrain_results_{config_code}_{i}.txt")

    logger.info(f"Retraining finished.")


if __name__ == '__main__':

	# How to run: python run_retraining.py --config FD001/config_files/config_turbofan_FD001_v18.txt --repeat 3
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True, help="Config file name (inside config_files/)")
	
	parser.add_argument(
		"--repeat",
		type=int,
		default=1,
		help="Number of repetitions (default: 1)"
    )
    
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
	repeat = int(args.repeat)  # config_file['train']['repeat']

	exp_path = os.path.join(base_path,
	                        exp_path_base,
	                        f"{exp}_repeat_{repeat}")

	arguments = {
        "retrain_folder": "retrain",
        "config_code": "F12",
        "max_epochs": config_file['train']['max_epochs_retrain'], #30,
        "batch_size": config_file['train']['batch_size_retrain'], #400,
        "eval_batch_size": config_file['train']['eval_batch_size_retrain'], #32,
        "lr_scheduler": config_file['train']['lr_scheduler'], #LambdaLR",
        "num_workers": config_file['train']['num_workers'], #4,
        "limit_data": False,
        "experiment_path": exp_path,
        "data_path": dataset_path,
        "dataset": dataset,
        "config_file": config_path,
        "continue_path": '',
        "extra_params": config_file['train']['extra_params'],
        "fitness_metric": config_file['train']['fitness_metric'],
        "log_level": config_file['train']['log_level'],
        "network_config": config_file['train']['network_config'],
        "optimizer": config_file['train']['optimizer'],
        "data_augmentation": config_file['train']['data_augmentation'],
        "save_checkpoints_epochs": config_file['train']['save_checkpoints_epochs'],
        "limit_data_value": config_file['train']['limit_data_value'],
        "network_gap": config_file['train']['network_gap'],
        "log_file_path": '', # For QIEA multihead
        "model_path": '', # For QIEA multihead
        "device": 'cuda',
        "num_repetitions": config_file['train']['num_repetitions_retrain'], #1
    }

	main(**arguments)