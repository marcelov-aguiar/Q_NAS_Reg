"""
Responsável por executar o retreinamento de todos os arquivos .txt dos diretorios passados
Semelhante ao runner_evolution.py, mas chama o run_retreining.py
"""
import subprocess
import os

if __name__ == "__main__":
	base_path = os.path.dirname(os.path.abspath(__file__))
	
	run_script = os.path.join(base_path, "run_retraining.py")

	# config_dir = os.path.join(base_path, "config_files")
	# config_files = [f for f in os.listdir(config_dir) if f.endswith(".txt")]
	config_files = [
		"FD001/config_files/config_turbofan_FD001_v7.txt",
		"FD002/config_files/config_turbofan_FD002_v4.txt",
		"FD003/config_files/config_turbofan_FD003_v4.txt",
		"FD004/config_files/config_turbofan_FD004_v4.txt"
	]
	for cfg in config_files:
		print(f"\n=== Executando experimento com {cfg} ===\n")
		subprocess.run(
			["python", run_script, "--config", cfg],
			cwd=base_path,
			check=True
        )