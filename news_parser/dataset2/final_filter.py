import pandas as pd
import numpy as np
import time
import os

# --- Конфигурация для Финальной Очистки ---
# INPUT: Файл, сохраненный после Шага 6, содержащий ВСЕ колонки
INPUT_FILE_STEP6 = '6intermediate_market_data_step6_target_imputed_reg.parquet'
# INPUT_FILE_STEP6 = 'intermediate_market_data_step6_target_imputed_reg.csv'

# OUTPUT: Очищенный финальный датасет (с новостями)
FINAL_CLEANED_DATASET_FILE = 'moex_qcml_final_dataset_cleaned_with_news.parquet' # Новое имя
# FINAL_CLEANED_DATASET_FILE = 'moex_qcml_final_dataset_cleaned_with_news.csv'

# Список РАССЧИТАННЫХ нормализованных признаков, которые мы ОСТАВЛЯЕМ
CALCULATED_NORM_FEATURES = [
    'norm_Momentum', 'norm_Size', 'norm_Beta'
]
# Целевая переменная
FINAL_TARGET = 'FinalTarget'

# ==============================================================================
# --- Загрузка Данных из Шага 6 ---
# ==============================================================================
print(f"\n--- Финальная Очистка Датасета (с Новостями) ---")
print(f"Загрузка данных из файла Шага 6: {INPUT_FILE_STEP6}...")
df_full = None
NEWS_FEATURES = [] # Определим при загрузке
try:
    if INPUT_FILE_STEP6.endswith('.parquet'): df_full = pd.read_parquet(INPUT_FILE_STEP6)
    elif INPUT_FILE_STEP6.endswith('.csv'): df_full = pd.read_csv(INPUT_FILE_STEP6, parse_dates=['TRADEDATE'])
    else: raise ValueError("Неподдерживаемый формат.")

    print("Проверка базовых колонок ПОСЛЕ ЗАГРУЗКИ...")
    # Определяем новостные колонки
    NEWS_FEATURES = [col for col in df_full.columns if col.startswith('news_')]
    if not NEWS_FEATURES: print("WARNING: Новостные колонки не обнаружены в загруженном файле!")

    # Проверяем наличие всех необходимых колонок
    required_cols = ['TRADEDATE', 'SECID', FINAL_TARGET] + CALCULATED_NORM_FEATURES + NEWS_FEATURES
    missing_required = [col for col in required_cols if col not in df_full.columns]
    if missing_required:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing_required}")

    if not pd.api.types.is_datetime64_any_dtype(df_full['TRADEDATE']):
        df_full['TRADEDATE'] = pd.to_datetime(df_full['TRADEDATE'])

    print(f"Данные загружены. Строк: {len(df_full)}.")
    # print("Колонки в загруженном файле:", df_full.columns.tolist())

except FileNotFoundError: print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл {INPUT_FILE_STEP6} не найден."); exit()
except Exception as e: print(f"КРИТИЧЕСКАЯ ОШИБКА загрузки: {e}"); exit()

# ==============================================================================
# --- Очистка Данных ---
# ==============================================================================

# 1. Выбор нужных колонок (идентификаторы, РАССЧИТАННЫЕ norm_ признаки, **НОВОСТИ**, таргет)
cols_to_keep_cleaned = ['TRADEDATE', 'SECID'] + CALCULATED_NORM_FEATURES + NEWS_FEATURES + [FINAL_TARGET]
print(f"\n1. Выбор колонок ({len(cols_to_keep_cleaned)}): {cols_to_keep_cleaned[:5]}...{cols_to_keep_cleaned[-3:]}") # Показываем начало и конец списка
try:
    df_cleaned = df_full[cols_to_keep_cleaned].copy()
    print("  Колонки выбраны.")
except Exception as e:
     print(f"ОШИБКА при выборе колонок: {e}"); exit()


# 2. Удаление строк с NaN в ЛЮБОЙ из выбранных колонок
# (Рассчитанные norm_ признаки, Новости ИЛИ FinalTarget)
print("\n2. Удаление строк с NaN в выбранных признаках, новостях или таргете...")
rows_before_drop = len(df_cleaned)
subset_to_check_nan = CALCULATED_NORM_FEATURES + NEWS_FEATURES + [FINAL_TARGET]
df_cleaned.dropna(subset=subset_to_check_nan, inplace=True)
rows_after_drop = len(df_cleaned)
print(f"  Количество строк до удаления NaN: {rows_before_drop}")
print(f"  Количество строк после удаления NaN: {rows_after_drop}")
print(f"  Удалено строк: {rows_before_drop - rows_after_drop}")

# ==============================================================================
# --- Финальная Проверка Очищенного Датасета ---
# ==============================================================================
print("\n--- Очищенный Финальный Датасет (Проверка) ---")
print("Колонки очищенного датасета:", df_cleaned.columns.tolist())
print("\nИнформация об очищенном датасете (info):")
df_cleaned.info(verbose=True, show_counts=True) # Все колонки должны иметь rows_after_drop non-null count

print("\nПроверка на NaN в очищенном датасете:")
nan_check = df_cleaned.isnull().sum()
nan_cols = nan_check[nan_check > 0]
if not nan_cols.empty:
     print(f"!!! WARNING: Найдены NaN после очистки: \n{nan_cols}")
else:
     print("NaN значения в выбранных колонках отсутствуют.")

print("\nСтатистика для FinalTarget в очищенном датасете:")
if FINAL_TARGET in df_cleaned.columns and df_cleaned[FINAL_TARGET].notna().any():
     print(df_cleaned[FINAL_TARGET].describe())
else:
     print("Колонка FinalTarget отсутствует или пуста.")


# ==============================================================================
# --- Сохранение Очищенного Финального Датасета ---
# ==============================================================================
print(f"\n--- Сохранение очищенного финального датасета в {FINAL_CLEANED_DATASET_FILE} ---")
try:
    # Сохраняем без индекса
    if FINAL_CLEANED_DATASET_FILE.endswith('.parquet'):
        df_cleaned.to_parquet(FINAL_CLEANED_DATASET_FILE, index=False)
        print(f"Файл {FINAL_CLEANED_DATASET_FILE} сохранен.")
    elif FINAL_CLEANED_DATASET_FILE.endswith('.csv'):
        df_cleaned.to_csv(FINAL_CLEANED_DATASET_FILE, index=False)
        print(f"Файл {FINAL_CLEANED_DATASET_FILE} сохранен.")
    else:
        print("Формат файла для сохранения не определен.")

except Exception as e:
    print(f"\nОШИБКА при сохранении финального файла: {e}")

print("\n--- Завершение Очистки ---")
print(f"Очищенный датасет (включая новости) сохранен в {FINAL_CLEANED_DATASET_FILE} и готов для обучения.")