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
		#"FD001/config_files/config_turbofan_FD001_v29.txt",
		#"FD001/config_files/config_turbofan_FD001_v35.txt",
		# "FD001/config_files/config_turbofan_FD001_v36.txt",
		# "FD002/config_files/config_turbofan_FD002_v11.txt", # apos retreinamento alterar repeat para 3
		# "FD002/config_files/config_turbofan_FD002_v10.txt",
		# "FD002/config_files/config_turbofan_FD002_v20.txt", # apos retreinamento alterar repeat para 3
		# "FD003/config_files/config_turbofan_FD003_v12.txt",
		# "FD003/config_files/config_turbofan_FD003_v13.txt",
		# "FD003/config_files/config_turbofan_FD003_v14.txt",
		# "FD003/config_files/config_turbofan_FD003_v15.txt",
		# "FD003/config_files/config_turbofan_FD003_v16.txt",
		# "FD003/config_files/config_turbofan_FD003_v17.txt",
		# "FD003/config_files/config_turbofan_FD003_v18.txt",
		# "FD003/config_files/config_turbofan_FD003_v19.txt",
		# "FD003/config_files/config_turbofan_FD003_v20.txt",
		# "FD003/config_files/config_turbofan_FD003_v21.txt",
		# "FD004/config_files/config_turbofan_FD004_v11.txt", # apos retreinamento alterar repeat para 3
		# "FD004/config_files/config_turbofan_FD004_v12.txt",
		# "FD004/config_files/config_turbofan_FD004_v13.txt",
		# "FD004/config_files/config_turbofan_FD004_v14.txt",
		"FD004/config_files/config_turbofan_FD004_v15.txt",
		"FD004/config_files/config_turbofan_FD004_v16.txt",
		"FD004/config_files/config_turbofan_FD004_v17.txt",
		"FD004/config_files/config_turbofan_FD004_v18.txt",
		"FD004/config_files/config_turbofan_FD004_v19.txt",
		"FD004/config_files/config_turbofan_FD004_v20.txt",
		"FD004/config_files/config_turbofan_FD004_v21.txt",
		"FD004/config_files/config_turbofan_FD004_v22.txt",
		"FD004/config_files/config_turbofan_FD004_v23.txt"
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