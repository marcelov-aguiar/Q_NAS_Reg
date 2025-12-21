from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "FD004"
    parent_run_id = "12699438dbce4ef3843705ba6b337759"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
