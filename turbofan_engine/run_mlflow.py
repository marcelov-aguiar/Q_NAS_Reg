import os
import mlflow
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from qnas_log import LogParamsEvolution
from qnas_log import DataQNASPKL, QNASLog, QNASPalette
from qnas_visualizer import QNASVisualizer
import constants.default_names as names
from util import load_yaml
# ==========================================================
# ================ PLOT FUNCTIONS (SRP) ====================
# ==========================================================

def generate_training_curves(data: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """Generates training and validation curves (loss and RMSE) and returns them as matplotlib figures."""
    plots = {}

    # --- Loss ---
    fig_loss, ax = plt.subplots()
    ax.plot(data["training_losses"], label="Training Loss")
    ax.plot(data["validation_losses"], label="Validation Loss")
    ax.set_title("Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    fig_loss.tight_layout()
    plots["loss"] = fig_loss

    # --- RMSE ---
    fig_rmse, ax = plt.subplots()
    ax.plot(data["training_rmse"], label="Training RMSE")
    ax.plot(data["validation_rmse"], label="Validation RMSE")
    ax.set_title("RMSE Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True)
    fig_rmse.tight_layout()
    plots["rmse"] = fig_rmse

    plt.close('all')
    return plots


# ==========================================================
# ============== MLflow LOG FUNCTIONS ======================
# ==========================================================

def log_retrain_run(retrain_id: str, metrics: Dict[str, Any], plots: Dict[str, plt.Figure], retrain_data: Dict[str, Any], parent_id: str):
    """Logs a single retraining (retrain_X) as a child MLflow run, with test, validation, and training metrics."""
    with mlflow.start_run(run_name=retrain_id, nested=True, parent_run_id=parent_id):
        # --- Test metrics ---
        mlflow.log_metric("test_rmse", metrics.get("test_rmse"))
        mlflow.log_metric("test_loss", metrics.get("test_loss"))

        # --- Validation metric ---
        best_val_rmse = metrics.get("best_rmse")
        mlflow.log_metric("validation_rmse_best", best_val_rmse)

        # --- Training metric corresponding to best validation ---
        val_list = retrain_data.get("validation_rmse", [])
        train_list = retrain_data.get("training_rmse", [])
        if best_val_rmse in val_list:
            idx_best = val_list.index(best_val_rmse)
            corresponding_train = train_list[idx_best]
            mlflow.log_metric("training_rmse_best", corresponding_train)

        # --- Log plots ---
        for name, fig in plots.items():
            mlflow.log_figure(fig, f"plots/{retrain_id}_{name}.png")
            plt.close(fig)

        return metrics.get("test_rmse")


def log_repeat_run(repeat_id: int, retrain_data_list: List[Dict[str, Any]], parent_id: str, exp_name: str,
                   log_params_evolution: Any, config_data: Dict[str, Any]):
    """Logs one search repeat (search_repeat_X) and all its retrain runs."""
    with mlflow.start_run(run_name=f"search_repeat_{repeat_id}", nested=True, parent_run_id=parent_id):
        test_rmses = []
        best_retrain = None

        for retrain_data in retrain_data_list:
            retrain_id = retrain_data["id"]
            metrics = retrain_data["metrics"]
            plots = retrain_data["plots"]

            test_rmse = log_retrain_run(retrain_id, metrics, plots, retrain_data, mlflow.active_run().info.run_id)
            if test_rmse is not None:
                test_rmses.append(test_rmse)
                if best_retrain is None or test_rmse < best_retrain["rmse"]:
                    best_retrain = {"rmse": test_rmse, "plots": plots}

        # --- Log best test RMSE of this repeat ---
        if test_rmses:
            mlflow.log_metric("min_test_rmse", float(np.min(test_rmses)))

        # --- QNAS evolution info (sem alteração) ---
        try:
            base_path = Path(__file__).resolve().parent
            dataset_dir = base_path / "FD001" / "exp_FD001" / f"{exp_name}_repeat_{repeat_id}"
            data_qnas_path = dataset_dir / names.DATA_QNAS_PKL

            if data_qnas_path.exists():
                data_qnas = DataQNASPKL(data_qnas_path)
                qnas_log = QNASLog(data_qnas, log_params_evolution)
                qnas_palette = QNASPalette(log_params_evolution)

                mlflow.log_metric("runtime_hours", data_qnas.get_runtime())
                repetition_dict = qnas_log.count_unique_individuals_all_gens()
                mlflow.log_metric("unique_architectures", len(repetition_dict))

                num_individuals = int(config_data["QNAS"]["num_quantum_ind"])
                gen_best, best_fitness, avg_fitness, worst_fitness = data_qnas.get_top_fitness_metrics()

                fig_fit = QNASVisualizer.plot_fitness_evolution(
                    generations=data_qnas.get_generations(),
                    gen_best=gen_best,
                    best_fitness=best_fitness,
                    avg_fitness=avg_fitness,
                    worst_fitness=worst_fitness,
                    show=False
                )
                mlflow.log_figure(fig_fit, "plots/qnas_fitness_evolution.png")
                plt.close(fig_fit)

                for ind_idx in range(num_individuals):
                    fig_prob = QNASVisualizer.plot_individual(
                        data_qnas.data_qnas,
                        generations=[0, max(data_qnas.data_qnas.keys())],
                        individual_idx=ind_idx,
                        labels=log_params_evolution.log_params_evolution["QNAS"].get("fn_list", []),
                        palette=qnas_palette.get_palette(),
                        figsize=(15, 6),
                        bar_width=0.75,
                        show=False
                    )
                    mlflow.log_figure(fig_prob, f"plots/qnas_probabilities_ind_{ind_idx}.png")
                    plt.close(fig_prob)

                    fig_hp = QNASVisualizer.plot_hyperparameter_evolution_grid(
                        dados=data_qnas.data_qnas,
                        geracoes=[0, max(data_qnas.data_qnas.keys())],
                        individual_index=ind_idx,
                        param_names=['LSTM1', 'LSTM2'],
                        show=False
                    )
                    mlflow.log_figure(fig_hp, f"plots/qnas_hyperparams_ind_{ind_idx}.png")
                    plt.close(fig_hp)

        except Exception as e:
            print(f"⚠️ Failed to log QNAS evolution for repeat {repeat_id}: {e}")

        return best_retrain


def log_experiment_run(data_set: str,
                       exp_name: str,
                       repeat_data: Dict[int, List[Dict[str, Any]]],
                       log_params_evolution: Any,
                       config_data: Dict[str, Any]):
    """Logs the full experiment (exp_X) as main MLflow run."""
    mlflow.set_experiment(data_set)

    with mlflow.start_run(run_name=exp_name) as main_run:
        qnas_dict = getattr(log_params_evolution, "log_params_evolution", log_params_evolution)
        qnas_params = qnas_dict.get("QNAS", {})
        for key, value in qnas_params.items():
            mlflow.log_param(key, json.dumps(value, ensure_ascii=False) if not isinstance(value, (int, float, str, bool)) else value)

        # --- Nested repeats ---
        best_test_rmses = []
        for repeat_id, retrain_data_list in repeat_data.items():
            best_retrain = log_repeat_run(repeat_id, retrain_data_list, main_run.info.run_id,
                                          exp_name, log_params_evolution, config_data)
            if best_retrain:
                best_test_rmses.append(best_retrain["rmse"])

        # --- Main experiment metrics (test metrics aggregated) ---
        if best_test_rmses:
            mlflow.log_metric("exp_best_test_rmse", float(np.min(best_test_rmses)))
            mlflow.log_metric("exp_mean_test_rmse", float(np.mean(best_test_rmses)))
            mlflow.log_metric("exp_std_test_rmse", float(np.std(best_test_rmses)))


# ==========================================================
# =============== DATA LOADING & PIPELINE ==================
# ==========================================================

def load_retrain_jsons(exp_repeat_path: Path) -> List[Dict[str, Any]]:
    """Loads retrain result file and prepares data for MLflow logging."""
    results_file = exp_repeat_path / "retrain_results_F12_5_LambdaLR.txt"
    if not results_file.exists():
        raise FileNotFoundError(f"Missing file: {results_file}")

    with open(results_file) as f:
        data = json.load(f)

    retrain_data = []
    for k, v in data.items():
        retrain_id = k
        metrics = {
            "test_rmse": v["test_rmse"],
            "test_loss": v["test_loss"],
            "best_rmse": v["best_rmse"],
        }
        plots = generate_training_curves(v)
        retrain_data.append({"id": retrain_id, "metrics": metrics, "plots": plots})
    return retrain_data


# ==========================================================
# ====================== MAIN SCRIPT =======================
# ==========================================================

if __name__ == "__main__":

    dataset = "FD001"
    base_path = os.path.dirname(os.path.abspath(__file__))

    config_dir = os.path.join(base_path, dataset, "config_files")
    config_files = [f for f in os.listdir(config_dir) if f.endswith(".txt")]
    config_files = ["config_turbofan_FD001_v10.txt"]
    for config_name in config_files:
        
        try:
            base_path = Path(__file__).resolve().parent / dataset / f"exp_{dataset}"
            config_path = Path(__file__).resolve().parent / dataset / "config_files" / config_name
            config_data = load_yaml(config_path)

            exp_name = config_data['train']['exp']

            num_repeats = int(config_data["train"]["repeat"])
            print(f"🔁 Número de repetições definido no config: {num_repeats}")

            log_params_evolution_path = base_path / f"{exp_name}_repeat_1" / names.LOG_PARAMS_EVOLUTION_TXT
            log_params_evolution = LogParamsEvolution(log_params_evolution_path)

            repeat_data = {}
            for repeat_id in range(1, num_repeats + 1):
                repeat_dir = base_path / f"{exp_name}_repeat_{repeat_id}"
                if not repeat_dir.exists():
                    print(f"⚠️ Warning: repeat folder not found -> {repeat_dir}")
                    continue

                retrain_data = load_retrain_jsons(repeat_dir)
                repeat_data[repeat_id] = retrain_data
                print(f"✅ Repeat {repeat_id} carregado com sucesso ({len(retrain_data)} retrains).")

            if not repeat_data:
                print("❌ Nenhum dado válido de repeat encontrado.")
            else:
                log_experiment_run(dataset, exp_name, repeat_data, log_params_evolution, config_data)
                print(f"🏁 Experimento '{exp_name}' registrado com sucesso no MLflow.")
        except FileNotFoundError as e:
            print(e)
            print("Arquivo não encontrado. A execução continuara")
            print(f"Arquivo {config_path} não processado")
            continue
