from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "femto"
    parent_run_id = "a8b4d1c5db0b4a8a8a6a84e28e7d50bc"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
