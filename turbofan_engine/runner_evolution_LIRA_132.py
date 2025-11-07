"""
Responsável por executar o retreinamento de todos os arquivos .txt dos diretorios passados
Chama o run_evolution.py
"""
import subprocess
import os
from util import load_yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
	base_path = os.path.dirname(os.path.abspath(__file__))
	
	run_script = os.path.join(base_path, "run_evolution.py")
	# How to execute: LD_LIBRARY_PATH= python nome_do_arquivo.py
	# config_dir = os.path.join(base_path, "config_files")
	# config_files = [f for f in os.listdir(config_dir) if f.endswith(".txt")]
	config_files = [
		"FD003/config_files/config_turbofan_FD003_v7.txt",
		"FD003/config_files/config_turbofan_FD003_v8.txt",
		"FD003/config_files/config_turbofan_FD003_v9.txt",
		"FD003/config_files/config_turbofan_FD003_v10.txt",
		"FD001/config_files/config_turbofan_FD001_v27.txt"
	]
	for cfg in config_files:
		config_path = os.path.join(base_path, cfg)
		config_data = load_yaml(config_path)
		repeat_count = config_data['train']['repeat']
		for repetition in range(1, repeat_count + 1):
			print(f"\n=== Executando experimento com {cfg} {repetition}/{repeat_count} ===\n")
			subprocess.run(
				["python", run_script, "--config", cfg, "--repeat", str(repetition)],
				cwd=base_path,
				check=True
        	)