import pandas as pd
import numpy as np
import time
import os

# --- Конфигурация для Шага 5 ---
# INPUT: Файл, сохраненный после Шага 4
INPUT_FILE_STEP4 = 'intermediate_market_data_step4_final_v5_debug.parquet' # Имя файла с рассчитанной Beta
# INPUT_FILE_STEP4 = 'intermediate_market_data_step4_iterated.csv'

# OUTPUT: Новый файл для результатов Шага 5
OUTPUT_FILE_STEP5 = '5intermediate_market_data_step5_normalized.parquet'
# OUTPUT_FILE_STEP5 = 'intermediate_market_data_step5_normalized.csv'

# Список признаков для нормализации (из Главного Сообщения, минус ShortUtil)
FEATURES_TO_NORMALIZE = [
    'Accruals', 'EBITDA_to_TEV', 'Momentum', 'Operating_Efficiency',
    'Profit_Margin', 'Size', 'Value', 'Beta'
]

# ==============================================================================
# --- Загрузка Данных из Шага 4 ---
# ==============================================================================
print(f"\n--- Шаг 5: Нормализация Признаков (Z-scoring) ---")
print(f"Загрузка данных из файла Шага 4: {INPUT_FILE_STEP4}...")
NEWS_FEATURES = [] # Определим при загрузке
df = None
try:
    if INPUT_FILE_STEP4.endswith('.parquet'): df = pd.read_parquet(INPUT_FILE_STEP4)
    elif INPUT_FILE_STEP4.endswith('.csv'): df = pd.read_csv(INPUT_FILE_STEP4, parse_dates=['TRADEDATE'])
    else: raise ValueError("Неподдерживаемый формат.")

    # --- Проверки после загрузки ---
    print("Проверка базовых колонок ПОСЛЕ ЗАГРУЗКИ...")
    required_cols = ['TRADEDATE', 'SECID'] # Минимально необходимые
    for col in required_cols:
        if col not in df.columns: raise ValueError(f"Отсутствует необходимая колонка: {col}")
    if not pd.api.types.is_datetime64_any_dtype(df['TRADEDATE']):
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])

    # Определяем новостные колонки
    NEWS_FEATURES = [col for col in df.columns if col.startswith('news_')]
    if NEWS_FEATURES: print(f"Обнаружены новостные колонки ({len(NEWS_FEATURES)}).")

    # --- Установка Индекса ---
    print("Установка индекса...")
    df.set_index(['TRADEDATE', 'SECID'], inplace=True)
    df.sort_index(inplace=True)
    if not df.index.is_unique: print("WARNING: Индекс не уникален!")
    print(f"Данные загружены. Строк: {len(df)}.")
    print(f"Колонки DataFrame при старте Шага 5: {df.columns.tolist()}")

except FileNotFoundError: print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл {INPUT_FILE_STEP4} не найден."); exit()
except Exception as e: print(f"КРИТИЧЕСКАЯ ОШИБКА загрузки: {e}"); exit()

# ==============================================================================
# --- Шаг 5: Кросс-секционная Нормализация (Z-scoring) ---
# ==============================================================================
print("\n--- Выполнение кросс-секционного Z-scoring ---")
start_time = time.time()

# Определяем, какие из FEATURES_TO_NORMALIZE реально присутствуют в df
features_to_process = [f for f in FEATURES_TO_NORMALIZE if f in df.columns]
if not features_to_process:
     print("ОШИБКА: Не найдены колонки для нормализации!")
     exit()

print(f"Признаки для нормализации ({len(features_to_process)}): {features_to_process}")
print("ПРИМЕЧАНИЕ: Шаг 'Вычитание среднего по индустрии' пропущен из-за отсутствия данных о секторах.")

# Группируем по дате
grouped_by_date = df.groupby(level='TRADEDATE')

# Применяем нормализацию к каждой колонке
for feature in features_to_process:
    print(f"  Нормализация признака: {feature}...")
    new_col_name = f"norm_{feature}" # Новое имя колонки

    # Рассчитываем Z-score с помощью transform
    # .std(ddof=0) для стандартного отклонения популяции, но ddof=1 (по умолчанию) тоже допустимо
    z_score = grouped_by_date[feature].transform(lambda x: (x - x.mean()) / x.std())

    # Заменяем inf/-inf на NaN (если std было 0 для какой-то даты)
    df[new_col_name] = z_score.replace([np.inf, -np.inf], np.nan)

    # --- Отладка: Проверим результат для одного признака ---
    if feature == 'Size': # Например, для Size
         print(f"    ОТЛАДКА ({feature}):")
         print(f"      Кол-во не-NaN в исходном '{feature}': {df[feature].notna().sum()}")
         print(f"      Кол-во не-NaN в '{new_col_name}': {df[new_col_name].notna().sum()}")
         if df[new_col_name].notna().any():
              print(f"      Статистика '{new_col_name}' (describe):")
              # Среднее по всем данным должно быть близко к 0, std к 1 (но не идеально из-за структуры)
              print(df[new_col_name].describe())
         else:
              print(f"      Все значения '{new_col_name}' являются NaN.")

print(f"\nНормализация завершена ({time.time() - start_time:.2f} сек).")

# --- Проверка результата Шага 5 ---
print("\n--- Проверка DataFrame после Шага 5 ---")
print("Новые нормализованные колонки:")
norm_cols = [col for col in df.columns if col.startswith('norm_')]
print(norm_cols)
print("\nИнформация о DataFrame (включая нормализованные колонки):")
df.info(verbose=True, show_counts=True)

# Проверим средние и стандартные отклонения нормализованных колонок
# Они не будут идеально 0 и 1 из-за NaN и структуры данных, но должны быть близки
print("\nСтатистика для нормализованных колонок:")
if norm_cols:
     # Не считаем статистику для полностью NaN колонок (фин. признаки)
     valid_norm_cols = df[norm_cols].dropna(axis=1, how='all').columns.tolist()
     if valid_norm_cols:
          print(df[valid_norm_cols].describe())
     else:
          print("Нет нормализованных колонок с не-NaN значениями для статистики.")
else:
     print("Нормализованные колонки не найдены.")


# ==============================================================================
# --- Сохранение Результата Шага 5 ---
# ==============================================================================
print(f"\n--- Сохранение данных после Шага 5 в {OUTPUT_FILE_STEP5} ---")
try:
    df_to_save = df.reset_index()
    print("Колонки для сохранения:", df_to_save.columns.tolist()) # Проверяем наличие norm_ колонок
    if OUTPUT_FILE_STEP5.endswith('.parquet'):
        df_to_save.to_parquet(OUTPUT_FILE_STEP5)
        print(f"Файл {OUTPUT_FILE_STEP5} сохранен.")
    elif OUTPUT_FILE_STEP5.endswith('.csv'):
        df_to_save.to_csv(OUTPUT_FILE_STEP5, index=False)
        print(f"Файл {OUTPUT_FILE_STEP5} сохранен.")
except Exception as e:
    print(f"\nОШИБКА при сохранении файла: {e}")

# ==============================================================================
# --- Верификация Сохраненного Файла (Опционально) ---
# ==============================================================================
print(f"\n--- Верификация сохраненного файла {OUTPUT_FILE_STEP5} ---")
try:
    if OUTPUT_FILE_STEP5.endswith('.parquet'): df_check = pd.read_parquet(OUTPUT_FILE_STEP5)
    elif OUTPUT_FILE_STEP5.endswith('.csv'): df_check = pd.read_csv(OUTPUT_FILE_STEP5, parse_dates=['TRADEDATE'])
    print("Файл успешно прочитан.")
    norm_cols_check = [col for col in df_check.columns if col.startswith('norm_')]
    print(f"Найдены нормализованные колонки в файле: {norm_cols_check}")
    if 'norm_Beta' in df_check.columns:
         print("\nСтатистика norm_Beta из файла:")
         if df_check['norm_Beta'].notna().any(): print(df_check['norm_Beta'].describe())
         else: print("  Все значения norm_Beta в файле NaN.")
    else: print("Колонка norm_Beta отсутствует в сохраненном файле!")
except Exception as e: print(f"ОШИБКА при верификации: {e}")


print("\n--- Завершение Шага 5 ---")