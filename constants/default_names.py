PARAMS_POP = 'params_pop'
NET_POP = 'net_pop'

CROSSOVER_RATE = "crossover_rate"
MAX_GENERATIONS = "max_generations"
MAX_NUM_NODES = "max_num_nodes"
NUM_QUANTUM_IND = "num_quantum_ind"
PENALIZE_NUMBER = "penalize_number"
REPETITION = "repetition"
REPLACE_METHOD = "replace_method"
UPDATE_QUANTUM_GEN = "update_quantum_gen"
UPDATE_QUANTUM_RATE = "update_quantum_rate"
SAVE_DATA_FREQ = "save_data_freq"
CROSSOVER_FREQUENCY = "crossover_frequency"
POP_CROSSOVER_RATE = "pop_crossover_rate"
POP_CROSSOVER_METHOD = "pop_crossover_method"
PATIENCE = "patience"
ALLOW_DUPLICATE_ARCHITECTURES = "allow_duplicate_architectures"
BATCH_SIZE = "batch_size"
BATCH_SIZE_RETRAIN = "batch_size_retrain"
CRITERION = "criterion"
DATA_AUGMENTATION = "data_augmentation"
NUM_SENSORS = "num_sensors"
SHARED_HEAD_ARCHITECTURE = "shared_head_architecture" #se True a mesma CNN é usada em todos os head, se False CNN diferente em cada head

DATALOADER_CLASS = "dataloader_class"
DATASET = "dataset"
DATASET_TYPE = "dataset_type"
DECAY = "decay"
EPOCHS_TO_EVAL = "epochs_to_eval"
EVAL_BATCH_SIZE = "eval_batch_size"
EVAL_BATCH_SIZE_RETRAIN = "eval_batch_size_retrain"
EXP = "exp"
EXP_PATH_BASE = "exp_path_base"
EXTRA_PARAMS = "extra_params"
FILE_EXTENSION = "file_extension"
FITNESS_METRIC = "fitness_metric"
LEARNING_RATE = "learning_rate"
LIMIT_DATA = "limit_data"
LIMIT_DATA_VALUE = "limit_data_value"
MAX_EPOCHS = "max_epochs"
MAX_EPOCHS_RETRAIN = "max_epochs_retrain"
MAX_INFERENCE_TIME = "max_inference_time"
MAX_PARAMS = "max_params"
MIXED_PRECISION = "mixed_precision"
MO_METRIC_BASE = "mo_metric_base"
NETWORK_CONFIG = "network_config"
NETWORK_GAP = "network_gap"
NUM_CLASSES = "num_classes"
NUM_REPETITIONS_RETRAIN = "num_repetitions_retrain"
NUM_WORKERS = "num_workers"
OPTIMIZER = "optimizer"
PHASE = "phase"
REPEAT = "repeat"
SAVE_CHECKPOINTS_EPOCHS = "save_checkpoints_epochs"
SAVE_SUMMARY_EPOCHS = "save_summary_epochs"
SUBTRACT_MEAN = "subtract_mean"
TASK = "task"
THREADS = "threads"
WEIGHT_DECAY = "weight_decay"


INITIAL_PROBS = "initial_probs"

DECODED_PARAMS = "decoded_params"
NET_LIST = "net_list"
NET_PROBS = "net_probs"
TOTAL_FLOPS = "total_flops"
TOTAL_TRAINABLE_PARAMS = "total_trainable_params"
TRAINING_TIME = "training_time"
MODEL_MEMORY_USAGE = "model_memory_usage"
GENERATION = "generation"

FUNCTION = "function"
KERNEL = "kernel"
STRIDE = "stride"
FILTERS = "filters"

CANDIDATE_NET_POP = 'candidate_net_pop'
CANDIDATE_PARAMS_POP = 'candidate_params_pop'
CANDIDATE_FITNESSES = 'candidate_fitnesses'
CANDIDATE_RAW_FITNESSES = 'candidate_raw_fitnesses'

NO_OP = 'no_op'

QNAS = 'QNAS'
TRAIN = "train"
FN_LIST = 'fn_list'
FN_DICT = "fn_dict"
PARAMS = "params"
BEST_SO_FAR_ID = "best_so_far_id"

AMOUNT_REPETITIONS = 'amount_repetitions'

INDIVIDUAL = 'individual'
TIME = 'time'
FITNESSES = 'fitnesses'
BEST_SO_FAR = 'best_so_far'

# Files name
LOG_PARAMS_EVOLUTION_TXT = "log_params_evolution.txt"
DATA_QNAS_PKL = "data_QNAS.pkl"
LOG_QNAS_TXT = "log_QNAS.txt"
TRAINING_PARAMS_TXT = "training_params.txt"

UNDERLINE = "_"

PARAMS_TO_SAVE = [
	CROSSOVER_RATE,
	MAX_GENERATIONS,
	MAX_NUM_NODES,
	NUM_QUANTUM_IND,
	PENALIZE_NUMBER,
	REPETITION,
	REPLACE_METHOD,
	UPDATE_QUANTUM_GEN,
	UPDATE_QUANTUM_RATE,
	SAVE_DATA_FREQ,
	CROSSOVER_FREQUENCY,
	POP_CROSSOVER_RATE,
	POP_CROSSOVER_METHOD,
	PATIENCE,
	ALLOW_DUPLICATE_ARCHITECTURES
]