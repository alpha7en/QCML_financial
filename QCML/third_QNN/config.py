# config.py

# --- Data Configuration ---
DATA_PATH = "0moex_qcml_final_dataset_with_embeddings.parquet"  # Укажите путь к вашему файлу Parquet
DATETIME_COLUMN = "TRADEDATE"         # Название колонки с датой/временем
TARGET_COLUMN = "FinalTarget" # Название вашей целевой колонки
# Список колонок, которые не являются ни признаками, ни целевой переменной, ни датой
# Например, если есть какие-то ID или другие служебные колонки, которые не должны попасть в признаки.
# Если таких колонок нет, оставьте список пустым: OTHER_EXCLUDE_COLS = []
OTHER_EXCLUDE_COLS = []


# --- Data Splitting Configuration ---
TEST_SIZE = 0.2  # Доля данных для тестовой выборки (20%)

# --- QNN Model Configuration ---
# N_QUBITS будет определено автоматически по количеству признаков после загрузки данных
N_LAYERS = 5  # Количество слоев в StronglyEntanglingLayers
ROTATION_GATE_EMBEDDING = 'X' # Тип вращения для AngleEmbedding ('X', 'Y', или 'Z')

# --- Training Configuration ---
LEARNING_RATE = 0.5
BATCH_SIZE = 1
EPOCHS = 4 # Начните с небольшого количества и увеличивайте по мере необходимости

# --- Reproducibility ---
RANDOM_SEED = 42

RESULTS_DIR = "results" # Директория для сохранения результатов
MODEL_NAME = "qnn_stock_predictor"

# --- Device Configuration ---
# Использовать "default.qubit" для симуляции на CPU.
# Если есть GPU и установлен pennylane-lightning[gpu], можно попробовать "lightning.gpu"
# Для начала рекомендуется "default.qubit"
QUANTUM_DEVICE = "lightning.qubit"

EMBEDDING_TYPE = "Amplitude" # или "Angle"
N_QUBITS_AMPLITUDE = 5

NUM_MP_WORKERS = 24