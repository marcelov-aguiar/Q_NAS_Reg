import os
import logging
from pathlib import Path
from util import load_yaml

from mlflow_utils import mlflow_tracking


LOG_FILE = "log_run_mlflow.log"

logging.basicConfig(
    level=logging.INFO,  # controla o nível dos logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),            # terminal
        logging.FileHandler(LOG_FILE, 'a')  # arquivo
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    base_path = os.path.dirname(os.path.abspath(__file__))

    config_dir = os.path.join(base_path, "config_files")
    config_files = [f for f in os.listdir(config_dir) if f.endswith(".txt")]
    # Sort by numeric version
    config_files = sorted(config_files, key=mlflow_tracking.extract_version)
    config_files = [
	    "config_femto_v5.txt",
		"config_femto_v6.txt",
		"config_femto_v7.txt",
		"config_femto_v8.txt",
        "config_femto_v9.txt"
    ]
    for config_name in config_files:
        dataset = config_name.split("_")[1]
        try:
            base_exp_path = Path(__file__).resolve().parent / f"exp_{dataset}"
            config_path = Path(__file__).resolve().parent / "config_files" / config_name
            config_data = load_yaml(config_path)

            exp_name = config_data['train']['exp']
            lr_scheduler = config_data['train']['lr_scheduler']
            num_repeats = int(config_data["train"]["repeat"])
            logger.info(f"🔁 Número de rodadas de buscas definido no config {exp_name}: {num_repeats}")

            repeat_data = {}
            for repeat_id in range(1, num_repeats + 1):
                repeat_dir = base_exp_path / f"{exp_name}_repeat_{repeat_id}"
                if not repeat_dir.exists():
                    logger.warning(f"⚠️ Warning: repeat folder not found -> {repeat_dir}")
                    continue
                
                retrain_data = mlflow_tracking.load_retrain_jsons(repeat_dir, lr_scheduler)
                repeat_data[repeat_id] = retrain_data
                logger.info(f"✅ Busca {repeat_id} carregado com sucesso ({len(retrain_data)} retrains).")

            if not repeat_data:
                logger.warning("❌ Nenhum dado válido de repeat encontrado.")
            else:

                logger.info(f"⏳ Carregando '{exp_name}' no MLflow. Aguarde...")
                mlflow_tracking.log_experiment_run(dataset, exp_name, repeat_data, base_exp_path, config_data)
                logger.info(f"🏁 Experimento '{exp_name}' registrado com sucesso no MLflow.")
        except FileNotFoundError as e:
            logger.error(e)
            logger.warning("Arquivo não encontrado. A execução continuara")
            logger.warning(f"Arquivo {config_path} não processado")
            continue
