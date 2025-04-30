import pandas as pd
import numpy as np
import time
import os

# --- Конфигурация для Шага 7 ---
# INPUT: Файл, сохраненный после Шага 6
INPUT_FILE_STEP6 = '6intermediate_market_data_step6_target_imputed_reg.parquet'
# INPUT_FILE_STEP6 = 'intermediate_market_data_step6_target_imputed_reg.csv'

# OUTPUT: Финальный датасет для обучения модели
FINAL_DATASET_FILE = 'moex_qcml_final_dataset.parquet'
# FINAL_DATASET_FILE = 'moex_qcml_final_dataset.csv'

# Список нормализованных признаков, которые должны войти в финальный датасет
# (согласно методологии, даже если они NaN)
FINAL_FEATURES_NORM = [
    'norm_Accruals', 'norm_EBITDA_to_TEV', 'norm_Momentum',
    'norm_Operating_Efficiency', 'norm_Profit_Margin', 'norm_Size',
    'norm_Value', 'norm_Beta'
]
FINAL_TARGET = 'FinalTarget'

# ==============================================================================
# --- Загрузка Данных из Шага 6 ---
# ==============================================================================
print(f"\n--- Шаг 7: Формирование Финального Датасета ---")
print(f"Загрузка данных из файла Шага 6: {INPUT_FILE_STEP6}...")
df_full = None
try:
    if INPUT_FILE_STEP6.endswith('.parquet'): df_full = pd.read_parquet(INPUT_FILE_STEP6)
    elif INPUT_FILE_STEP6.endswith('.csv'): df_full = pd.read_csv(INPUT_FILE_STEP6, parse_dates=['TRADEDATE'])
    else: raise ValueError("Неподдерживаемый формат.")

    print("Проверка базовых колонок ПОСЛЕ ЗАГРУЗКИ...")
    required_cols = ['TRADEDATE', 'SECID', FINAL_TARGET] + FINAL_FEATURES_NORM
    missing_required = [col for col in required_cols if col not in df_full.columns]
    if missing_required:
        # Если не хватает norm_ фин. признаков, это ожидаемо, но проверим остальные
        expected_but_missing = [col for col in missing_required if col not in ['norm_Accruals', 'norm_EBITDA_to_TEV', 'norm_Operating_Efficiency', 'norm_Profit_Margin', 'norm_Value']]
        if expected_but_missing:
             raise ValueError(f"Отсутствуют необходимые колонки: {expected_but_missing}")
        else:
             print(f"ПРИМЕЧАНИЕ: Ожидаемо отсутствуют NaN-колонки фин. признаков: {missing_required}")
             # Убираем их из списка для выбора
             FINAL_FEATURES_NORM = [col for col in FINAL_FEATURES_NORM if col in df_full.columns]

    if not pd.api.types.is_datetime64_any_dtype(df_full['TRADEDATE']):
        df_full['TRADEDATE'] = pd.to_datetime(df_full['TRADEDATE'])

    print(f"Данные загружены. Строк: {len(df_full)}.")

except FileNotFoundError: print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл {INPUT_FILE_STEP6} не найден."); exit()
except Exception as e: print(f"КРИТИЧЕСКАЯ ОШИБКА загрузки: {e}"); exit()

# ==============================================================================
# --- Шаг 7.1: Выбор Необходимых Колонок ---
# ==============================================================================
print("\nВыбор колонок для финального датасета...")

# Колонки для выбора: идентификаторы + нормированные признаки + таргет
cols_to_keep = ['TRADEDATE', 'SECID'] + FINAL_FEATURES_NORM + [FINAL_TARGET]
print(f"Выбираемые колонки ({len(cols_to_keep)}): {cols_to_keep}")

try:
    df_final = df_full[cols_to_keep].copy()
    print("Колонки успешно выбраны.")
except KeyError as e:
    print(f"ОШИБКА: Не удалось выбрать колонки: {e}")
    print("Возможная причина: одна из колонок в cols_to_keep отсутствует в df_full.")
    print("Колонки в df_full:", df_full.columns.tolist())
    exit()
except Exception as e:
     print(f"Неожиданная ошибка при выборе колонок: {e}")
     exit()


# ==============================================================================
# --- Шаг 7.2: Удаление Строк с NaN в FinalTarget ---
# ==============================================================================
print(f"\nУдаление строк с NaN в колонке '{FINAL_TARGET}'...")
rows_before_drop = len(df_final)
print(f"Количество строк до удаления NaN: {rows_before_drop}")

# Используем dropna только по целевой колонке
df_final.dropna(subset=[FINAL_TARGET], inplace=True)

rows_after_drop = len(df_final)
print(f"Количество строк после удаления NaN: {rows_after_drop}")
print(f"Удалено строк: {rows_before_drop - rows_after_drop}")

# ==============================================================================
# --- Шаг 7.3: Финальная Проверка ---
# ==============================================================================
print("\n--- Финальный Датасет (Проверка) ---")
print("Колонки финального датасета:", df_final.columns.tolist())
print("\nИнформация о финальном датасете (info):")
# Устанавливаем MultiIndex для более наглядного info
if not isinstance(df_final.index, pd.MultiIndex):
     try:
          df_final.set_index(['TRADEDATE', 'SECID'], inplace=True)
          df_final.sort_index(inplace=True)
     except KeyError:
          print("Не удалось установить MultiIndex, TRADEDATE/SECID отсутствуют?")

df_final.info(verbose=True, show_counts=True)

print("\nПример финального датасета (head):")
print(df_final.head())
print("\nПример финального датасета (tail):")
print(df_final.tail())
print("\nСтатистика для FinalTarget в финальном датасете:")
if FINAL_TARGET in df_final.columns and df_final[FINAL_TARGET].notna().any():
     print(df_final[FINAL_TARGET].describe())
else:
     print("Колонка FinalTarget отсутствует или пуста.")

# ==============================================================================
# --- Шаг 7.4: Сохранение Финального Датасета ---
# ==============================================================================
print(f"\n--- Сохранение финального датасета в {FINAL_DATASET_FILE} ---")
try:
    # Сбрасываем индекс перед сохранением для стандартного формата файла
    df_to_save = df_final.reset_index()
    if FINAL_DATASET_FILE.endswith('.parquet'):
        df_to_save.to_parquet(FINAL_DATASET_FILE)
        print(f"Файл {FINAL_DATASET_FILE} сохранен.")
    elif FINAL_DATASET_FILE.endswith('.csv'):
        df_to_save.to_csv(FINAL_DATASET_FILE, index=False)
        print(f"Файл {FINAL_DATASET_FILE} сохранен.")
    else:
        print("Формат файла для сохранения не определен (нужен .parquet или .csv)")

except Exception as e:
    print(f"\nОШИБКА при сохранении финального файла: {e}")

print("\n--- Завершение Шага 7 ---")
print("Финальный датасет готов для обучения модели.")