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
		"FD001/config_files/config_turbofan_FD001_v10.txt",
		"FD001/config_files/config_turbofan_FD001_v11.txt",
		"FD001/config_files/config_turbofan_FD001_v12.txt",
		"FD001/config_files/config_turbofan_FD001_v13.txt"
		#"FD001/config_files/config_turbofan_FD001_v9.txt",
		#"FD002/config_files/config_turbofan_FD002_v5.txt",
		#"FD002/config_files/config_turbofan_FD002_v6.txt",
		#"FD002/config_files/config_turbofan_FD002_v7.txt",
		#"FD003/config_files/config_turbofan_FD003_v5.txt",
		#"FD003/config_files/config_turbofan_FD003_v6.txt",
		#"FD004/config_files/config_turbofan_FD004_v5.txt",
		#"FD004/config_files/config_turbofan_FD004_v6.txt"
		#"FD003/config_files/config_turbofan_FD003_v3.txt"
	]
	for cfg in config_files:
		print(f"\n=== Executando experimento com {cfg} ===\n")
		subprocess.run(
			["python", run_script, "--config", cfg],
			cwd=base_path,
			check=True
        )