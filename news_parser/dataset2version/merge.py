import pandas as pd
import numpy as np
import time
import os

# --- Конфигурация ---
# INPUT 1: Финальный датасет (очищенный, с новостями)
INPUT_FINAL_DATASET = 'moex_qcml_final_dataset_cleaned_with_news.parquet'
# INPUT_FINAL_DATASET = 'moex_qcml_final_dataset_cleaned_with_news.csv'

# INPUT 2: Файл с эмбеддингами
EMBEDDINGS_FILE = 'embeding/umap_embeddings.csv' # Укажите имя вашего файла эмбеддингов

# OUTPUT: Финальный датасет с эмбеддингами вместо тикеров
OUTPUT_DATASET_WITH_EMBEDDINGS = '0moex_qcml_final_dataset_with_embeddings.parquet'
# OUTPUT_DATASET_WITH_EMBEDDINGS = 'moex_qcml_final_dataset_with_embeddings.csv'

# Ожидаемые имена колонок эмбеддингов
EMBEDDING_COLS = ['umap_1', 'umap_2', 'umap_3']

# ==============================================================================
# --- Загрузка Данных ---
# ==============================================================================
print(f"\n--- Добавление Эмбеддингов Компаний ---")
print(f"Загрузка основного датасета: {INPUT_FINAL_DATASET}...")
df_final = None
try:
    if INPUT_FINAL_DATASET.endswith('.parquet'):
        df_final = pd.read_parquet(INPUT_FINAL_DATASET)
    elif INPUT_FINAL_DATASET.endswith('.csv'):
        df_final = pd.read_csv(INPUT_FINAL_DATASET, parse_dates=['TRADEDATE'])
    else: raise ValueError("Неподдерживаемый формат основного файла.")

    # Проверка необходимых колонок в основном датасете
    required_final_cols = ['TRADEDATE', 'SECID']
    missing_final_cols = [col for col in required_final_cols if col not in df_final.columns]
    if missing_final_cols:
        raise ValueError(f"В основном датасете отсутствуют колонки: {missing_final_cols}")

    if not pd.api.types.is_datetime64_any_dtype(df_final['TRADEDATE']):
        df_final['TRADEDATE'] = pd.to_datetime(df_final['TRADEDATE'])

    print(f"Основной датасет загружен. Строк: {len(df_final)}.")

except FileNotFoundError: print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл {INPUT_FINAL_DATASET} не найден."); exit()
except Exception as e: print(f"КРИТИЧЕСКАЯ ОШИБКА загрузки основного датасета: {e}"); exit()


print(f"\nЗагрузка эмбеддингов из файла: {EMBEDDINGS_FILE}...")
df_embeddings = None
try:
    # Читаем CSV, используя первую колонку как индекс (где ожидаются тикеры)
    df_embeddings = pd.read_csv(EMBEDDINGS_FILE, index_col=0)

    # Проверяем наличие колонок эмбеддингов
    missing_embedding_cols = [col for col in EMBEDDING_COLS if col not in df_embeddings.columns]
    if missing_embedding_cols:
         raise ValueError(f"В файле эмбеддингов отсутствуют колонки: {missing_embedding_cols}")

    # Присваиваем имя индексу для слияния
    df_embeddings.index.name = 'SECID'

    print(f"Эмбеддинги загружены. Тикеров: {len(df_embeddings)}.")
    print("Пример эмбеддингов:")
    print(df_embeddings.head(3))

except FileNotFoundError: print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл {EMBEDDINGS_FILE} не найден."); exit()
except Exception as e: print(f"КРИТИЧЕСКАЯ ОШИБКА загрузки эмбеддингов: {e}"); exit()


# ==============================================================================
# --- Слияние Датасетов ---
# ==============================================================================
print("\nВыполнение слияния основного датасета с эмбеддингами...")
start_time = time.time()

# Используем left merge по колонке SECID в df_final и индексу SECID в df_embeddings
df_merged = pd.merge(
    df_final,
    df_embeddings[EMBEDDING_COLS], # Выбираем только нужные колонки эмбеддингов
    left_on='SECID',
    right_index=True, # Используем индекс df_embeddings для слияния
    how='left',       # Сохраняем все строки из df_final
    validate="many_to_one" # Каждому SECID в df_final должен соответствовать макс 1 эмбеддинг
)

print(f"Слияние завершено ({time.time() - start_time:.2f} сек).")
print(f"Размер DataFrame после слияния: {df_merged.shape}") # Размер строк должен остаться как у df_final

# ==============================================================================
# --- Обработка Отсутствующих Эмбеддингов ---
# ==============================================================================
print("\nПроверка и обработка строк без соответствующих эмбеддингов...")

# Находим строки, где НЕ удалось присоединить эмбеддинги (есть NaN хотя бы в одной umap колонке)
missing_embeddings_mask = df_merged[EMBEDDING_COLS].isnull().any(axis=1)
num_missing = missing_embeddings_mask.sum()

if num_missing > 0:
    print(f"WARNING: Найдено {num_missing} строк, для которых не найдены эмбеддинги.")
    # Выведем примеры тикеров без эмбеддингов
    tickers_missing = df_merged.loc[missing_embeddings_mask, 'SECID'].unique()
    print(f"  Примеры тикеров без эмбеддингов: {tickers_missing[:10]}...") # Показываем первые 10

    # Удаляем строки без эмбеддингов
    print("Удаление строк без эмбеддингов...")
    df_merged.dropna(subset=EMBEDDING_COLS, inplace=True)
    print(f"Осталось строк после удаления: {len(df_merged)}")
else:
    print("Для всех строк найдены соответствующие эмбеддинги.")

# ==============================================================================
# --- Удаление Колонки SECID ---
# ==============================================================================
print("\nУдаление исходной колонки SECID...")

if 'SECID' in df_merged.columns:
    df_merged.drop(columns=['SECID'], inplace=True)
    print("Колонка SECID успешно удалена.")
else:
    print("WARNING: Колонка SECID уже отсутствует.")

# ==============================================================================
# --- Финальная Проверка ---
# ==============================================================================
print("\n--- Финальный Датасет с Эмбеддингами (Проверка) ---")
print("Колонки финального датасета:", df_merged.columns.tolist()) # SECID должен отсутствовать
print("\nИнформация о финальном датасете (info):")
df_merged.info(verbose=True, show_counts=True) # Все колонки должны иметь одинаковый non-null count

print("\nПроверка на NaN в финальном датасете:")
nan_check = df_merged.isnull().sum()
nan_cols = nan_check[nan_check > 0]
if not nan_cols.empty:
     print(f"!!! WARNING: Найдены NaN после очистки: \n{nan_cols}")
else:
     print("NaN значения в финальном датасете отсутствуют.")

# ==============================================================================
# --- Сохранение Финального Датасета с Эмбеддингами ---
# ==============================================================================
print(f"\n--- Сохранение финального датасета с эмбеддингами в {OUTPUT_DATASET_WITH_EMBEDDINGS} ---")
try:
    # Сохраняем без индекса
    if OUTPUT_DATASET_WITH_EMBEDDINGS.endswith('.parquet'):
        df_merged.to_parquet(OUTPUT_DATASET_WITH_EMBEDDINGS, index=False)
        print(f"Файл {OUTPUT_DATASET_WITH_EMBEDDINGS} сохранен.")
    elif OUTPUT_DATASET_WITH_EMBEDDINGS.endswith('.csv'):
        df_merged.to_csv(OUTPUT_DATASET_WITH_EMBEDDINGS, index=False)
        print(f"Файл {OUTPUT_DATASET_WITH_EMBEDDINGS} сохранен.")
    else:
        print("Формат файла для сохранения не определен.")

except Exception as e:
    print(f"\nОШИБКА при сохранении финального файла: {e}")

print("\n--- Завершение Добавления Эмбеддингов ---")
print(f"Финальный датасет с эмбеддингами сохранен в {OUTPUT_DATASET_WITH_EMBEDDINGS} и готов для обучения.")