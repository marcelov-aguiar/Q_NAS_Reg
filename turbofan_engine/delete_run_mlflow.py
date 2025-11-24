import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import shutil


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


# === Exemplo de uso ===
if __name__ == "__main__":
    experiment_name = "FD003"
    parent_run_id = "283324d62e0943e48e8ca2e5d8fc7024"
    mlruns_path = "./mlruns"
    delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
