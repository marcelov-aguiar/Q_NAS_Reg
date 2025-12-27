import os
import mlflow
import json
import logging
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
from qnas_log import LogParamsEvolution
from qnas_log import DataQNASPKL, QNASLog, QNASPalette, TrainingParams
from qnas_visualizer import QNASVisualizer
import constants.default_names as names
from util import load_yaml
from mlflow.tracking import MlflowClient
import shutil

# =============== LOGGER CONFIGURATION =====================
# ==========================================================

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



# ==========================================================
# ================ PLOT FUNCTIONS (SRP) ====================
# ==========================================================

# Salvar todos os parametros do QNAS no config

# Salvar todos os parametros do train no config

# Salvar por busca e pegar essa infos no arquivo training_params.txt
model_params_to_save = [
    names.DECODED_PARAMS,
    names.NET_LIST
]

model_metrics_to_save = [
    names.TOTAL_FLOPS,
    names.TOTAL_TRAINABLE_PARAMS,
    names.TRAINING_TIME,
    names.MODEL_MEMORY_USAGE,
]


def calculate_convergence_metrics(data_qnas_dict: Dict[int, Any],
                                  update_interval: int = 1) -> Dict[str, float]:
    """
    Calcula métricas de convergência e dinâmica evolutiva baseadas na variação
    das probabilidades (SAD - Sum of Absolute Differences).

    Parameters
    ----------
    data_qnas_dict : Dict[int, Any]
        Dicionário contendo os dados de log do Q-NAS. As chaves devem ser
        o número da geração (int) e os valores devem conter a chave 'net_probs'
        com as matrizes de probabilidade (shape: [n_ind, n_nodes, n_funcs]).
    
    update_interval : int, optional
        O intervalo de gerações em que o algoritmo efetivamente aplica a
        atualização quântica (parâmetro `update_quantum_gen` do config).
        O padrão é 1.
        
        Importante: Usar o valor correto evita o cálculo de "falsa inatividade"
        nos intervalos onde o algoritmo está programado para não fazer nada.

    Returns
    -------
    Dict[str, float]
        Um dicionário contendo as seguintes métricas:
        
        * 'sad_mean': Média do SAD de toda a população (Início vs. Fim).
        * 'sad_std': Desvio padrão do SAD da população.
        * 'sad_elite': SAD total do primeiro indivíduo (Rank 0 / Melhor).
        * 'sad_bottom': SAD total do último indivíduo (Rank N / Pior).
        * 'inactivity_rate': Proporção de atualizações onde a mudança foi nula (zero).

    Notes
    -----
    **Interpretação das Métricas:**

    1. **SAD (Sum of Absolute Differences):**
       Mede a "quantidade de aprendizado" ou movimentação da massa de probabilidade.
       Calculado como ``sum(|Prob_Final - Prob_Inicial|)``.
       
       * **SAD Alto:** O indivíduo convergiu ou mudou drasticamente de opinião. 
         Indica uma evolução forte e decisiva.
       * **SAD Baixo (~0):** O indivíduo terminou a evolução na mesma dúvida 
         que começou (estagnação). Indica falha no aprendizado.

    2. **Elite vs. Bottom (`sad_elite` vs `sad_bottom`):**
       Compara a qualidade do aprendizado entre o melhor e o pior indivíduo.
       
       * **Gap Alto (Elite >> Bottom):** Confirma o problema do "Professor Ruim".
         O melhor indivíduo aprende com o melhor clássico e evolui bem. O último
         aprende com um clássico medíocre e fica confuso/estagnado.
       * **Gap Baixo:** A população evolui de forma homogênea (sinal saudável).

    3. **Taxa de Inatividade (`inactivity_rate`):**
       Mede a frequência com que os indivíduos "dormem" durante uma rodada de atualização.
       
       * **Valor Alto (> 0.5):** Indica que a máscara aleatória está bloqueando
         atualizações com muita frequência. Isso ocorre geralmente quando se tem
         muitos indivíduos e poucos blocos (genes), tornando estatisticamente
         raro um indivíduo ser selecionado para atualização.
       * **Valor Baixo (< 0.2):** O fluxo de atualização é constante e saudável.
    """
    generations = sorted(data_qnas_dict.keys())
    if not generations:
        return None

    start_gen = generations[0]
    end_gen = generations[-1]

    # Matrizes de Probabilidade: Shape (Num_Individuos, Num_Nos, Num_Funcoes)
    probs_start = data_qnas_dict[start_gen][names.NET_PROBS]
    probs_end = data_qnas_dict[end_gen][names.NET_PROBS]

    # --- 1. SAD Total (Inicio vs Fim) ---
    # Diferença absoluta entre a última e a primeira geração
    diff_total = np.abs(probs_end - probs_start)
    sad_per_ind = np.sum(diff_total, axis=(1, 2))

    mean_sad = float(np.mean(sad_per_ind))
    std_sad = float(np.std(sad_per_ind))
    first_ind_sad = float(sad_per_ind[0])   # Elite (assumindo que ind 0 é o melhor)
    last_ind_sad = float(sad_per_ind[-1])   # Bottom

    # --- 2. Inatividade (Geração a Geração) ---
    inactive_count = 0
    total_comparisons = 0
    threshold = 1e-6 # Consideramos 0 se a mudança for muito ínfima

    # Percorre geração por geração para ver quem "dormiu"
    for i in range(0, len(generations) - update_interval, update_interval):
        g_curr = generations[i]
        g_next = generations[i + update_interval]

        if g_next not in data_qnas_dict:
            break

        p_curr = data_qnas_dict[g_curr]['net_probs']
        p_next = data_qnas_dict[g_next]['net_probs']
        
        # SAD de um passo (step)
        step_sad = np.sum(np.abs(p_next - p_curr), axis=(1, 2))
        
        # Conta quantos indivíduos tiveram SAD quase zero nessa transição
        inactive_count += np.sum(step_sad < threshold)
        total_comparisons += len(step_sad)

    inactivity_rate = inactive_count / total_comparisons if total_comparisons > 0 else 0.0

    return {
        "sad_mean": mean_sad,
        "sad_std": std_sad,
        "sad_elite": first_ind_sad,
        "sad_bottom": last_ind_sad,
        "inactivity_rate": inactivity_rate
    }


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


def log_repeat_run(base_exp_path: str, repeat_id: int, retrain_data_list: List[Dict[str, Any]],
                   parent_id: str, exp_name: str,
                   log_params_evolution: LogParamsEvolution, config_data: Dict[str, Any]):
    """Logs one search repeat (search_repeat_X) and all its retrain runs."""

    qnas_metrics_export = None
    shared_head_architecture = config_data[names.TRAIN][names.EXTRA_PARAMS][names.SHARED_HEAD_ARCHITECTURE]
    num_heads = config_data[names.TRAIN][names.EXTRA_PARAMS][names.NUM_SENSORS]
    nodes_per_head = config_data[names.QNAS][names.MAX_NUM_NODES]

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
            dataset_dir = base_exp_path / f"{exp_name}_repeat_{repeat_id}"
            data_qnas_path = dataset_dir / names.DATA_QNAS_PKL

            if data_qnas_path.exists():
                data_qnas = DataQNASPKL(data_qnas_path)
                qnas_log = QNASLog(data_qnas, log_params_evolution)
                qnas_palette = QNASPalette(log_params_evolution)

                mlflow.log_metric("runtime_hours", data_qnas.get_runtime())
                repetition_dict = qnas_log.count_unique_individuals_all_gens()
                mlflow.log_metric("unique_architectures", len(repetition_dict))

                best_so_far_id = data_qnas.get_best_so_far_id()
                best_so_far_id = "_".join(map(str, best_so_far_id))
                training_params_path = dataset_dir / best_so_far_id/ names.TRAINING_PARAMS_TXT
                training_params = TrainingParams(training_params_path)
                for model_params in model_params_to_save:
                    mlflow.log_param(f'model.{model_params}',
                                      training_params.training_params[model_params])
                mlflow.log_param('best_so_far_id', best_so_far_id)
                for model_metrics in model_metrics_to_save:
                    value = training_params.training_params[model_metrics]
                    mlflow.log_metric(model_metrics, float(value))

                num_individuals = int(config_data[names.QNAS][names.NUM_QUANTUM_IND])
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
                    if shared_head_architecture: # caso experimento tiver mesma CNN para cada head
                        fig_prob = QNASVisualizer.plot_individual(
                            data_qnas.data_qnas,
                            generations=[0, max(data_qnas.data_qnas.keys())],
                            individual_idx=ind_idx,
                            labels=log_params_evolution.log_params_evolution[names.QNAS].get(names.FN_LIST, []),
                            palette=qnas_palette.get_palette(),
                            figsize=(15, 6),
                            bar_width=0.75,
                            show=False
                        )
                        mlflow.log_figure(fig_prob, f"plots/qnas_probabilities_ind_{ind_idx}.png")
                        plt.close(fig_prob)
                    else: # caso experimento tiver CNN diferente para cada head
                        fig_prob = QNASVisualizer.plot_individual(
                            data_qnas.data_qnas,
                            generations=[0, max(data_qnas.data_qnas.keys())],
                            individual_idx=ind_idx,
                            labels=log_params_evolution.log_params_evolution[names.QNAS].get(names.FN_LIST, []),
                            palette=qnas_palette.get_palette(),
                            figsize=(35, 12),
                            bar_width=0.75,
                            show=False
                        )
                        mlflow.log_figure(fig_prob, f"plots/qnas_probabilities_ind_{ind_idx}.png")
                        plt.close(fig_prob)

                        fig_prob_head = QNASVisualizer.plot_individual_head_aggregation(
                            data_qnas.data_qnas,
                            generations=[0, max(data_qnas.data_qnas.keys())],
                            individual_idx=ind_idx,
                            labels=log_params_evolution.get_fn_list(),
                            palette=qnas_palette.get_palette(),
                            figsize=(18, 6),
                            bar_width=0.75,
                            show=False,
                            num_heads=num_heads,
                            nodes_per_head=nodes_per_head,
                            aggregate_by_head=True
                        )
                        mlflow.log_figure(fig_prob_head, f"plots/qnas_probabilities_ind_{ind_idx}_head.png")
                        plt.close(fig_prob_head)

                    fig_hp = QNASVisualizer.plot_hyperparameter_evolution_grid(
                        dados=data_qnas.data_qnas,
                        geracoes=[0, max(data_qnas.data_qnas.keys())],
                        individual_index=ind_idx,
                        param_names=['LSTM1', 'LSTM2'],
                        show=False
                    )
                    mlflow.log_figure(fig_hp, f"plots/qnas_hyperparams_ind_{ind_idx}.png")
                    plt.close(fig_hp)
                
                # Metrics Sum of Absolute Differences and inactivity
                update_interval = int(config_data.get(names.QNAS, {}).get("update_quantum_gen", 1))
                mlflow.log_param("actual_update_interval", update_interval)

                qnas_metrics = calculate_convergence_metrics(data_qnas.data_qnas, update_interval)
                
                if qnas_metrics:
                    # Loga métricas locais desta busca no MLFlow
                    mlflow.log_metric("qnas_sad_mean", qnas_metrics["sad_mean"])
                    mlflow.log_metric("qnas_sad_std", qnas_metrics["sad_std"])
                    mlflow.log_metric("qnas_inactivity_rate", qnas_metrics["inactivity_rate"])
                    mlflow.log_metric("qnas_sad_elite", qnas_metrics["sad_elite"])
                    mlflow.log_metric("qnas_sad_bottom", qnas_metrics["sad_bottom"])
                    
                    # Prepara dados para retornar ao experimento pai (para cálculo de médias globais)
                    qnas_metrics_export = {
                        "elite": qnas_metrics["sad_elite"],
                        "bottom": qnas_metrics["sad_bottom"],
                        "mean_sad": qnas_metrics["sad_mean"],
                        "inactivity": qnas_metrics["inactivity_rate"]
                    }

        except Exception as e:
            logger.warning(f"⚠️ Failed to log QNAS evolution for repeat {repeat_id}: {e}")

        return best_retrain, qnas_metrics_export


def log_experiment_run(data_set: str,
                       exp_name: str,
                       repeat_data: Dict[int, List[Dict[str, Any]]],
                       base_exp_path: str,
                       config_data: Dict[str, Any]):
    """Logs the full experiment (exp_X) as main MLflow run."""
    mlflow.set_experiment(data_set)

    with mlflow.start_run(run_name=exp_name) as main_run:
        # Save QNAS params
        #qnas_params = log_params_evolution.log_params_evolution[names.QNAS]
        qnas_params = config_data[names.QNAS]
        for key, value in qnas_params.items():
            mlflow.log_param(f'qnas.{key}', json.dumps(value, ensure_ascii=False) if not isinstance(value, (int, float, str, bool)) else value)

        # Save train params
        train_params = config_data[names.TRAIN]
        for key, value in train_params.items():
            mlflow.log_param(f'train.{key}', json.dumps(value, ensure_ascii=False) if not isinstance(value, (int, float, str, bool)) else value)

        # --- Nested repeats ---
        best_test_rmses = []
        best_retrain_info = None
        best_repeat_id = None

        all_elite_sads = []
        all_bottom_sads = []
        all_mean_sads = []       
        all_inactivity_rates = []

        for repeat_id, retrain_data_list in repeat_data.items():
            log_params_evolution_path = base_exp_path / f"{exp_name}_repeat_{repeat_id}" / names.LOG_PARAMS_EVOLUTION_TXT
            log_params_evolution = LogParamsEvolution(log_params_evolution_path)
            best_retrain, qnas_metrics_export = log_repeat_run(base_exp_path, repeat_id, retrain_data_list, main_run.info.run_id,
                                          exp_name, log_params_evolution, config_data)
            if best_retrain:
                best_test_rmses.append(best_retrain["rmse"])

                # Salva info do melhor retrain global
                if best_retrain_info is None or best_retrain["rmse"] < best_retrain_info["rmse"]:
                    best_retrain_info = best_retrain
                    best_repeat_id = repeat_id

            if qnas_metrics_export:
                all_elite_sads.append(qnas_metrics_export["elite"])
                all_bottom_sads.append(qnas_metrics_export["bottom"])
                all_mean_sads.append(qnas_metrics_export["mean_sad"])          
                all_inactivity_rates.append(qnas_metrics_export["inactivity"])
        
        # Metrics Sum of Absolute Differences and inactivity
        if all_elite_sads: # Se coletou de pelo menos uma rodada
            # 1. Média da Inatividade Global (Sua dúvida principal sobre o 'mask')
            mlflow.log_metric("exp_avg_inactivity_rate", float(np.mean(all_inactivity_rates)))
            
            # 2. Média do SAD Global das rodadas de busca
            mlflow.log_metric("exp_avg_search_sad", float(np.mean(all_mean_sads)))

            # 3. Análise Elite vs Bottom (Média das 3 rodadas)
            mean_elite = np.mean(all_elite_sads)
            mean_bottom = np.mean(all_bottom_sads)
            
            mlflow.log_metric("exp_avg_sad_elite", float(mean_elite))
            mlflow.log_metric("exp_avg_sad_bottom", float(mean_bottom))
        
        # --- Main experiment metrics (test metrics aggregated) ---
        if best_test_rmses:
            mlflow.log_metric("exp_best_test_rmse", float(np.min(best_test_rmses)))
            mlflow.log_metric("exp_mean_test_rmse", float(np.mean(best_test_rmses)))
            mlflow.log_metric("exp_std_test_rmse", float(np.std(best_test_rmses)))

        # --- Log best repeat parameters/metrics to parent run ---
        if best_retrain_info is not None:
            try:
                # base_exp_path = Path(__file__).resolve().parent
                dataset_dir = base_exp_path / f"{exp_name}_repeat_{best_repeat_id}"
                data_qnas_path = dataset_dir / names.DATA_QNAS_PKL
                if data_qnas_path.exists():
                    data_qnas = DataQNASPKL(data_qnas_path)

                    log_params_evolution_path = dataset_dir / names.LOG_PARAMS_EVOLUTION_TXT
                    log_params_evolution = LogParamsEvolution(log_params_evolution_path)
                    qnas_log = QNASLog(data_qnas, log_params_evolution)
                    mlflow.log_metric("best_runtime_hours", data_qnas.get_runtime())
                    repetition_dict = qnas_log.count_unique_individuals_all_gens()
                    mlflow.log_metric("best_unique_architectures", len(repetition_dict))

                    best_so_far_id = data_qnas.get_best_so_far_id()
                    best_so_far_id = "_".join(map(str, best_so_far_id))
                    training_params_path = dataset_dir / best_so_far_id / names.TRAINING_PARAMS_TXT
                    training_params = TrainingParams(training_params_path)
                    mlflow.log_param('best_so_far_id', best_so_far_id)
                    # Log parameters (architecture)
                    for model_param in model_params_to_save:
                        mlflow.log_param(f"best_{model_param}", training_params.training_params[model_param])

                    # Log metrics (efficiency)
                    for model_metric in model_metrics_to_save:
                        value = training_params.training_params[model_metric]
                        mlflow.log_metric(f"best_{model_metric}", float(value))

            except Exception as e:
                logger.warning(f"⚠️ Failed to log best repeat parameters/metrics: {e}")
# ==========================================================
# =============== DATA LOADING & PIPELINE ==================
# ==========================================================

def load_retrain_jsons(exp_repeat_path: Path, lr_scheduler: str) -> List[Dict[str, Any]]:
    """Loads retrain result file and prepares data for MLflow logging."""
    results_file = exp_repeat_path / f"retrain_results_F12_5_{lr_scheduler}.txt"
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


def extract_version(filename):
    """
    Extracts the numeric version X from 'config_turbofan_FD001_vX.txt'.
    Returns an integer for proper numeric sorting.
    """
    match = re.search(r"_v(\d+)\.txt$", filename)
    return int(match.group(1)) if match else float('inf')  # inf → files without version go to the end


def get_all_descendant_runs(client: MlflowClient, experiment_id: str, parent_run_id: str):
    """
    Busca recursivamente todos os nested runs (em qualquer nível) de um run pai.
    """
    all_descendants = []

    def recurse(current_parent_id):
        nested_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{current_parent_id}'",
            run_view_type=mlflow.tracking.client.ViewType.ALL
        )
        for run in nested_runs:
            all_descendants.append(run)
            recurse(run.info.run_id)

    recurse(parent_run_id)
    return all_descendants


def delete_run_physical_and_mlflow(experiment_name: str, parent_run_id: str, mlruns_path: str = "./mlruns"):
    """
    Apaga um run principal e todos os nested runs (em todos os níveis),
    tanto via MLflow quanto fisicamente no disco.
    """
    client = MlflowClient()
    
    # 1️⃣ Pega o ID do experimento
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"❌ Experimento '{experiment_name}' não encontrado")
        return
    experiment_id = experiment.experiment_id
    
    # 2️⃣ Busca todos os runs descendentes (níveis 1, 2, ...)
    all_nested = get_all_descendant_runs(client, experiment_id, parent_run_id)
    print(f"🔍 Encontrados {len(all_nested)} nested runs (todos os níveis)")

    # 3️⃣ Apaga todos os nested runs (de baixo para cima)
    for run in reversed(all_nested):  # remove filhos antes dos pais
        run_id = run.info.run_id
        # print(f"🗑 Apagando nested run {run_id} via MLflow")
        client.delete_run(run_id)
        
        run_path = Path(mlruns_path) / experiment_id / run_id
        if run_path.exists():
            # print(f"🗑 Apagando pasta física {run_path}")
            shutil.rmtree(run_path)

    # 4️⃣ Apaga o run principal
    print(f"🗑 Apagando run principal {parent_run_id} via MLflow")
    client.delete_run(parent_run_id)
    
    run_path = Path(mlruns_path) / experiment_id / parent_run_id
    if run_path.exists():
        print(f"🗑 Apagando pasta física {run_path}")
        shutil.rmtree(run_path)
    
    print("✅ Run principal e todos os nested runs apagados com sucesso (API + disco)!")

def get_run_id_by_experiment_and_name(
    experiment_name: str,
    run_name: str
) -> Optional[str]:
    """
    Retorna o run_id dado o nome do experimento e o nome do run.
    Se houver mais de um run com o mesmo nome, retorna o mais recente.
    """
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        return None

    return runs[0].info.run_id