HOUR = "hour"
MINUTE = "min"
SECOND = "sec"
MICROSECOND = "microsecond"
VIB_H = "vib_h"
VIB_V = "vib_v"

COLUMNS = [
	HOUR, MINUTE, SECOND, MICROSECOND, VIB_H, VIB_V
]

# Bearings
TYPES_BEARING_TRAIN = [
    "Bearing1_1",
    "Bearing1_2",
    "Bearing2_1",
    "Bearing2_2",
    "Bearing3_1",
    "Bearing3_2",
]

TYPES_BEARING_TEST = [
    "Bearing1_3",
    "Bearing1_4",
    "Bearing1_5",
    "Bearing1_6",
    "Bearing1_7",
    "Bearing2_3",
    "Bearing2_4",
    "Bearing2_5",
    "Bearing2_6",
    "Bearing2_7",
    "Bearing3_3",
]


RMS = "rms"
KURTOSIS = "kurtosis"
PEAK_TO_PEAK = "peak_to_peak"
MEAN_FREQ = "mean_freq"
RMS_FREQ = "rms_freq"
WAVELET_ENERGY = "wavelet_energy"

CONCATENATED_FINAL_TEST_DATASET_NAME = "final_test_last_dataset.parquet"
CONCATENATED_TEST_DATASET_NAME = "femto_multihead_test.parquet"
CONCATENATED_TRAIN_DATASET_NAME = "femto_multihead_train.parquet"

COLS_NON_SENSOR = ['bearing_id', 'RUL']

COLS_TO_DROP = ['file_name',
				'sample_idx',
				'rul_seconds',
				'rul_files',
				'elapsed_time',
				'delta_seconds',
				'dataset_type']

SCALER_NAME = "femto_scaler.save"


# Estes são os índices (0-based) que correspondem aos tempos de corte oficiais.
# Ex: 1_4 -> idx 1138 -> Tempo 11380s -> RUL 339s.
RPT_INDICES = {
    "1_3": 1801,
    "1_4": 1138,
    "1_5": 2301,
    "1_6": 2301,
    "1_7": 1501,
    "2_3": 1201,
    "2_4": 611,
    "2_5": 2001,
    "2_6": 571,
    "2_7": 171,
    "3_3": 351
}