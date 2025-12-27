"""
Código responsável por criar log onde o config file tem a chave shared_head_architecture
"""
import os
import logging
from pathlib import Path
from util import load_yaml
from mlflow_utils import mlflow_tracking


def setup_logger(log_file: Path):
    logger = logging.getLogger("shared_head_checker")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler (opcional)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
	# id = mlflow_tracking.get_run_id_by_experiment_and_name("FD004", "exp_v12")
	# print(id)
	base_path = Path(__file__).resolve().parent
	log_path = base_path / "shared_head_architecture_check.log"

	logger = setup_logger(log_path)

	for dataset in ["FD001", "FD002", "FD003", "FD004"]:
		config_dir = base_path / dataset / "config_files"
		config_files = [f for f in os.listdir(config_dir) if f.endswith(".txt")]

		# Sort by numeric version
		config_files = sorted(config_files, key=mlflow_tracking.extract_version)

		logger.info(f"======== {dataset} ========")

		for config_name in config_files:
			dataset_name = config_name.split("_")[2]
			config_path = base_path / dataset_name / "config_files" / config_name

			try:
				config_data = load_yaml(config_path)
				shared_head_architecture = (
					config_data["train"]["extra_params"]["shared_head_architecture"]
				)

				if not shared_head_architecture:
					logger.info(
						f"{config_name} -> shared_head_architecture = False"
					)
				else:
					logger.info(
						f"{config_name} -> shared_head_architecture = True"
					)

			except KeyError:
				logger.warning(f"Pulando {config_name}: chave não encontrada.")
			except Exception as e:
				logger.error(f"Erro ao processar {config_name}: {e}")
