from air_quality.preprocessing.air_quality_window import AirQualityWindow

class AirQualityNetworkLauncher:
	def __init__(self):
		pass

	def opt_network_input_generator(
			self,
			df,
			cols_sensors,
			campaign_name,
			target_name,
			sequence_length=30,
			stride=1,
			window_length=3,
			horizon=1):
		"""
		Gera o input para o modelo Multi-Head.
		Args:
			cols_sensors: Colunas que são features (usado para identificar o X).
		"""
		n_window = int((sequence_length - window_length) / stride + 1)

		# 1. Gera X (Features janeladas)
		# O FemtoWindow usa cols_sensors para saber o que é Feature
		seq_array = AirQualityWindow().seq_generation(df, cols_sensors, sequence_length, horizon, campaign_name)

		# 2. Formata para Multi-Head (List of Arrays para cada sensor)
		# Shape final: Lista de [Samples, 1, Window, 1] ou similar dependendo da arquitetura
		network_input = AirQualityWindow().networkinput_generation(seq_array, stride, n_window, window_length)

		# 3. Gera Y (Labels janelados)
		network_label = AirQualityWindow().label_generation(df, sequence_length, horizon, campaign_name, target_name)

		return network_input, network_label