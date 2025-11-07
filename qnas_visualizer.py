import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.ticker as ticker


class QNASVisualizer:

	@staticmethod
	def plot_individual(
		data,
		generations,
		individual_idx,
		labels,
		palette=None,
		figsize=(14,6),
		bar_width=0.8,
		show=True
	):
		"""
		Plots stacked bar charts with small gaps, custom colors, and adjustable figure size.

		Args:
		    data (dict): your data dict where data[gen]['net_probs'] is shape (5,20,11)
		    generations (list of int): e.g. [0, last_gen]
		    individual_idx (int): which individual to plot (0–4)
		    labels (list of str]): length-11 list of operator names
		    palette (dict): optional mapping label → hex color
		    figsize (tuple): (width, height) in inches
		    bar_width (float): width of each bar (0.5 → big gaps, 0.9 → small gaps)
		"""
		n_gens = len(generations)
		fig, axes = plt.subplots(n_gens, 1, figsize=figsize, sharex=True, sharey=True)
		if n_gens == 1:
			axes = [axes]

	    # Build a list of colors in the same order as `labels`
		if palette:
			colors = [palette[label] for label in labels]
		else:
			colors = None  # let pandas choose

		for ax, gen in zip(axes, generations):
			probs = data[gen]['net_probs'][individual_idx]  # (20,11)
			df = pd.DataFrame(probs, columns=labels)

			df.plot(
				kind='bar',
				stacked=True,
				ax=ax,
				width=bar_width,
				edgecolor='white',
				linewidth=0.5,
				color=colors,
				legend=False
			)

			ax.set_ylabel('Probability', fontsize=12)
			ax.set_title(f'Individual {individual_idx} – Generation {gen}', fontsize=12, fontweight='bold')
			ax.set_xticks(np.arange(probs.shape[0]))
			ax.set_xticklabels(np.arange(probs.shape[0]), rotation=0)

		axes[-1].set_xlabel('Network nodes', fontsize=12)

		# One legend on top plot
		axes[0].legend(
			labels,
			bbox_to_anchor=(1.02, 1),
			loc='upper left',
			title='Functions'
		)

		plt.tight_layout()
		if show:
			plt.show()
		return fig

	@staticmethod
	def plot_hyperparameter_evolution_grid(
		dados: Dict[int, Dict[str, np.ndarray]],
		geracoes: List[int],
		individual_index: int,
		param_names: List[str],
		final_values: Optional[List[float]] = None,
		show=True
	):
		"""
		Plota uma grade de gráficos comparando a evolução da PDF de múltiplos
		hiperparâmetros para um único indivíduo quântico.

		Args:
		    dados (Dict):
		        Dicionário com os dados de evolução, onde a chave é a geração (int).
		        Cada valor é um dict com as chaves 'lower' e 'upper'.

		    geracoes (List[int]):
		        Lista com duas gerações para comparar: [geração_inicial, geração_final].

		    individual_index (int):
		        O índice do indivíduo quântico a ser plotado.

		    param_names (List[str]):
		        Lista com os nomes dos hiperparâmetros, que serão usados
		        como títulos dos subplots.

		    final_values (Optional[List[float]], optional):
		        Lista opcional com os valores finais (melhores) para cada
		        hiperparâmetro, que serão plotados como uma linha vertical tracejada.
				[0.35, 0.48, 4.5e-4] # Valores para a linha tracejada do hiperparâmetro.
		"""
		# --- Validação das Entradas ---
		if len(geracoes) != 2:
			raise ValueError("A lista 'geracoes' deve conter exatamente dois números.")
	
		gen_inicial_num, gen_final_num = geracoes[0], geracoes[1]
	
		try:
			dados_inicial = dados[gen_inicial_num]
			dados_final = dados[gen_final_num]
		except KeyError as e:
			raise KeyError(f"A geração {e} não foi encontrada no dicionário de dados.")

		num_params = dados_inicial['lower'].shape[1]
		if len(param_names) != num_params:
			raise ValueError(f"O número de nomes de parâmetros ({len(param_names)}) não "
							 f"corresponde ao número de hiperparâmetros nos dados ({num_params}).")

	    # --- Configuração do Layout do Grid ---
		if num_params <= 3:
			num_rows, num_cols = 1, num_params
		else:
			num_cols = 3
			num_rows = int(np.ceil(num_params / num_cols))

		plt.style.use('seaborn-notebook')
		fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False)
	
		# Cores fixas
		cor_inicial = '#c0392b'  # Vermelho
		cor_final = '#34495e'    # Azul/Cinza escuro

		# --- Itera e plota cada hiperparâmetro ---
		for i, ax in enumerate(axes.flat):
			if i >= num_params:
				ax.set_visible(False)
				continue
			
			# Extração e Cálculo dos Dados
			lower_inicial = dados_inicial['lower'][individual_index, i]
			upper_inicial = dados_inicial['upper'][individual_index, i]
			lower_final = dados_final['lower'][individual_index, i]
			upper_final = dados_final['upper'][individual_index, i]

			width_inicial = upper_inicial - lower_inicial
			width_final = upper_final - lower_final
	
			altura_inicial = 1.0 / width_inicial if width_inicial > 0 else 0
			altura_final = 1.0 / width_final if width_final > 0 else 0

			# Eixo Principal (Esquerda) - INICIAL (Vermelho)
			ax_twin = ax.twinx()
			x_inicial = [lower_inicial, lower_inicial, upper_inicial, upper_inicial]
			y_inicial = [0, altura_inicial, altura_inicial, 0]
			p1, = ax.plot(x_inicial, y_inicial, color=cor_inicial, linewidth=2, label='initial') # Captura o handle p1
			ax.tick_params(axis='y', labelcolor=cor_inicial)
			ax.set_ylim(0, altura_inicial * 1.15)
	
			# Eixo Gêmeo (Direita) - FINAL (Azul/Cinza)
			x_final = [lower_final, lower_final, upper_final, upper_final]
			y_final = [0, altura_final, altura_final, 0]
			p2, = ax_twin.plot(x_final, y_final, color=cor_final, linewidth=2, label='final') # Captura o handle p2
			ax_twin.tick_params(axis='y', labelcolor=cor_final)
			ax_twin.set_ylim(0, altura_final * 1.15)

			# Linha Tracejada
			p3 = None
			if final_values:
				p3 = ax_twin.axvline(x=final_values[i], color=cor_final, linestyle='--', linewidth=1.2, label='final values') # Captura o handle p3

			# Anotações e Rótulos
			ax.set_title(param_names[i], fontsize=14)
			width_ratio = width_final / width_inicial if width_inicial > 0 else 0
			ax.text(0.05, 0.9, f'width ratio = {width_ratio:.2f}', transform=ax.transAxes, fontsize=10)
	
			if i % num_cols == 0:
				ax.set_ylabel(f'Individual {individual_index}', fontsize=12)

		# Monta a lista de handles (objetos das linhas) para a legenda manualmente
		legend_handles = [p1, p2]
		if p3: # Adiciona o handle da linha tracejada se ele foi criado
			legend_handles.append(p3)
	
		# Cria a legenda para a figura inteira usando a lista de handles montada
		fig.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=True)

		plt.tight_layout(rect=[0, 0, 1, 0.96])
		if show:
			plt.show()
		return fig

	@staticmethod
	def plot_fitness_evolution(
		generations: List[int],
		gen_best: List[float],
		best_fitness: List[float],
		avg_fitness: List[float],
		worst_fitness: List[float],
		show_best_fitness: bool = True,
		show: bool = True,
	):
		"""
		Plot fitness (loss) evolution over generations for the training data.

		Args:
		    generations (List[int]): List of generation indices.
		    gen_best (List[float]): Best fitness values for each generation.
		    best_fitness (List[float]): Best fitness values observed so far.
		    avg_fitness (List[float]): Average fitness per generation.
		    worst_fitness (List[float]): Worst fitness per generation.
		    show_best_fitness (bool): Whether to plot the global best fitness curve.
		    show (bool): Whether to display the figure interactively (default: True).

		Returns:
			matplotlib.figure.Figure: The generated matplotlib figure.
		"""
		fig, ax = plt.subplots(figsize=(10, 6))

		ax.plot(generations, gen_best, color='lime', alpha=0.4, marker='o', linewidth=0.8, label='Best in Generation')
		if show_best_fitness:
			ax.plot(generations, best_fitness, color='green', alpha=0.4, marker='o', linewidth=0.8, label='Best Fitness')
		ax.plot(generations, avg_fitness, color='blue', alpha=0.4, marker='o', linewidth=0.8, label='Average Fitness')
		ax.plot(generations, worst_fitness, color='red', alpha=0.4, marker='o', linewidth=0.8, label='Worst Fitness')

		ax.set_xlabel('Generation')
		ax.set_ylabel('Fitness (Loss)')
		ax.set_title('QNAS Fitness Evolution')
		ax.legend()
		ax.grid(True)
		ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
		fig.tight_layout()

		if show:
			plt.show()

		return fig
