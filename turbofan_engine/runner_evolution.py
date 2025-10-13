"""
Responsável por executar o retreinamento de todos os arquivos .txt dos diretorios passados
Chama o run_evolution.py
"""
import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
	base_path = os.path.dirname(os.path.abspath(__file__))
	
	run_script = os.path.join(base_path, "run_evolution.py")
	# How to execute: LD_LIBRARY_PATH= python nome_do_arquivo.py
	# config_dir = os.path.join(base_path, "config_files")
	# config_files = [f for f in os.listdir(config_dir) if f.endswith(".txt")]
	config_files = [
		#"FD001/config_files/config_turbofan_FD001_v8.txt",
		#"FD001/config_files/config_turbofan_FD001_v9.txt",
		#"FD002/config_files/config_turbofan_FD002_v5.txt",
		#"FD002/config_files/config_turbofan_FD002_v6.txt",
		#"FD002/config_files/config_turbofan_FD002_v7.txt",
		"FD003/config_files/config_turbofan_FD003_v5.txt",
		"FD003/config_files/config_turbofan_FD003_v6.txt",
		"FD004/config_files/config_turbofan_FD004_v5.txt",
		"FD004/config_files/config_turbofan_FD004_v6.txt"
	]
	for cfg in config_files:
		print(f"\n=== Executando experimento com {cfg} ===\n")
		subprocess.run(
			["python", run_script, "--config", cfg],
			cwd=base_path,
			check=True
        )