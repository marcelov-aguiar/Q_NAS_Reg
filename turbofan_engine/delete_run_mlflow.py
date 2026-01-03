from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "FD004"
    parent_run_id = "8a57932090304f5992658a47cb4efddb"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
