"""
Responsável por executar o retreinamento de todos os arquivos .txt dos diretorios passados
Semelhante ao runner_evolution.py, mas chama o run_retreining.py
"""
import subprocess
import os
from util import load_yaml


if __name__ == "__main__":
	base_path = os.path.dirname(os.path.abspath(__file__))
	
	run_script = os.path.join(base_path, "run_retraining.py")

	# config_dir = os.path.join(base_path, "config_files")
	# config_files = [f for f in os.listdir(config_dir) if f.endswith(".txt")]
	config_files = [
		#"FD001/config_files/config_turbofan_FD001_v15.txt",
		#"FD001/config_files/config_turbofan_FD001_v16.txt",
		#"FD001/config_files/config_turbofan_FD001_v17.txt",
		#"FD001/config_files/config_turbofan_FD001_v18.txt",
		"FD001/config_files/config_turbofan_FD001_v25.txt",
		"FD002/config_files/config_turbofan_FD002_v8.txt",
		"FD003/config_files/config_turbofan_FD003_v8.txt",
		"FD003/config_files/config_turbofan_FD003_v9.txt",
		"FD004/config_files/config_turbofan_FD004_v9.txt"
		#"FD004/config_files/config_turbofan_FD004_v8.txt"
		#"FD001/config_files/config_turbofan_FD001_v20.txt",
		#"FD001/config_files/config_turbofan_FD001_v21.txt",
		#"FD004/config_files/config_turbofan_FD004_v3.txt",
		#"FD004/config_files/config_turbofan_FD004_v4.txt",
		#"FD004/config_files/config_turbofan_FD004_v5.txt",
		#"FD004/config_files/config_turbofan_FD004_v6.txt"
	]
	for cfg in config_files:
		config_path = os.path.join(base_path, cfg)
		config_data = load_yaml(config_path)
		repeat_count = config_data['train']['repeat']
		for repetition in range(1, repeat_count + 1):
			print(f"\n=== Executando retreinamento com {cfg} {repetition}/{repeat_count} ===\n")
			subprocess.run(
				["python", run_script, "--config", cfg, "--repeat", str(repetition)],
				cwd=base_path,
				check=True
       		)