import pandas as pd
import numpy as np
import time
import os

# --- Configuration for Step 3 ---
# INPUT: File saved by the successful Steps 1 & 2 run
INPUT_FILE_STEP2 = 'intermediate_market_data_step3_final.parquet'
# INPUT_FILE_STEP2 = 'intermediate_market_data_step2_unique_check.csv' # If you used CSV

# OUTPUT: New file for the results of this isolated Step 3 run
OUTPUT_FILE_STEP3 = 'intermediate_market_data_step3_isolated.parquet'
# OUTPUT_FILE_STEP3 = 'intermediate_market_data_step3_isolated.csv'

# Constants for calculation
MOMENTUM_SHORT_LAG = 21
MOMENTUM_LONG_LAG = 252
FINANCIAL_FEATURES = [
    'Accruals', 'EBITDA_to_TEV', 'Operating_Efficiency',
    'Profit_Margin', 'Value'
]

# ==============================================================================
# --- Load Data from Step 2 ---
# ==============================================================================
print(f"\n--- Изолированный Шаг 3 ---")
print(f"Загрузка данных из файла Шага 2: {INPUT_FILE_STEP2}...")

try:
    if INPUT_FILE_STEP2.endswith('.parquet'):
        df = pd.read_parquet(INPUT_FILE_STEP2)
    elif INPUT_FILE_STEP2.endswith('.csv'):
         df = pd.read_csv(INPUT_FILE_STEP2, parse_dates=['TRADEDATE'])
    else:
         raise ValueError("Неподдерживаемый формат входного файла.")

    # --- Проверки после загрузки ---
    print("Проверка колонок ПОСЛЕ ЗАГРУЗКИ:", df.columns.tolist())
    if 'TRADEDATE' not in df.columns: raise ValueError("Нет колонки TRADEDATE")
    if not pd.api.types.is_datetime64_any_dtype(df['TRADEDATE']):
        print("Конвертация TRADEDATE...")
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    if 'SECID' not in df.columns: raise ValueError("Нет колонки SECID")
    if 'CLOSE' not in df.columns: raise ValueError("Нет колонки CLOSE")
    if 'MarketCap' not in df.columns: print("WARNING: Нет колонки MarketCap") # Size будет NaN

    print("Установка индекса...")
    df.set_index(['TRADEDATE', 'SECID'], inplace=True)
    df.sort_index(inplace=True)
    if not df.index.is_unique: print("WARNING: Индекс загруженных данных не уникален!")
    print(f"Данные загружены. Строк: {len(df)}.")

except FileNotFoundError:
    print(f"\nКРИТИЧЕСКАЯ ОШИБКА: Файл {INPUT_FILE_STEP2} не найден. Убедитесь, что он существует.")
    exit()
except Exception as e:
    print(f"\nКРИТИЧЕСКАЯ ОШИБКА при загрузке файла {INPUT_FILE_STEP2}: {e}")
    exit()

# ==============================================================================
# --- Шаг 3: Расчет Входных Признаков (с Детальной Отладкой) ---
# ==============================================================================
print("\n--- Расчет Признаков Шага 3 ---")

# --- Расчет Momentum ---
print("\nРасчет Momentum...")
start_time = time.time()
try:
    if 'CLOSE' in df.columns:
        df_grouped = df.groupby(level='SECID')['CLOSE']
        close_t_minus_21 = df_grouped.transform(lambda x: x.shift(MOMENTUM_SHORT_LAG))
        close_t_minus_252 = df_grouped.transform(lambda x: x.shift(MOMENTUM_LONG_LAG))
        # ОТЛАДКА 1:
        print(f"ОТЛАДКА (Momentum): Не-NaN в close_t_minus_21: {close_t_minus_21.notna().sum()}")
        print(f"ОТЛАДКА (Momentum): Не-NaN в close_t_minus_252: {close_t_minus_252.notna().sum()}")
        ratio = close_t_minus_21 / close_t_minus_252
        # ОТЛАДКА 2:
        print(f"ОТЛАДКА (Momentum): Не-NaN/inf/0 в ratio (до log): {ratio.notna().sum()}/{np.isinf(ratio).sum()}/{(ratio == 0).sum()}")
        momentum_calculated = np.log(ratio)
         # ОТЛАДКА 3:
        print(f"ОТЛАДКА (Momentum): Не-NaN/inf в momentum_calculated (после log): {momentum_calculated.notna().sum()}/{np.isinf(momentum_calculated).sum()}")
        df['Momentum'] = momentum_calculated.replace([np.inf, -np.inf], np.nan)
        # ОТЛАДКА 4 + 5:
        print(f"Momentum рассчитан и присвоен ({time.time() - start_time:.2f} сек).")
        print("ОТЛАДКА: Проверка колонки 'Momentum' СРАЗУ ПОСЛЕ РАСЧЕТА:")
        if 'Momentum' in df.columns:
            print(f"  Тип данных Momentum: {df['Momentum'].dtype}") # ВАЖНО: Проверяем тип
            print(f"  Количество НЕ-NaN Momentum: {df['Momentum'].notna().sum()}")
            if df['Momentum'].notna().any():
                print("  Статистика Momentum (describe):")
                print(df['Momentum'].describe())
                print("  Примеры НЕ-NaN значений Momentum:")
                print(df['Momentum'].dropna().head())
            else: print("  Все значения Momentum являются NaN.")
        else: print("  ОШИБКА: Колонка 'Momentum' не добавилась!")
    else:
        print("WARNING: Колонка CLOSE отсутствует, Momentum не рассчитан.")
        df['Momentum'] = np.nan
except Exception as e:
     print(f"ОШИБКА при расчете Momentum: {e}")
     df['Momentum'] = np.nan

# --- Расчет Size ---
print("\nРасчет Size (ln(MarketCap))...")
start_time = time.time()
try:
    if 'MarketCap' in df.columns and df['MarketCap'].notna().any():
         # ОТЛАДКА 6:
        print(f"ОТЛАДКА (Size): Не-NaN/<=0 в MarketCap: {df['MarketCap'].notna().sum()}/{(df['MarketCap'] <= 0).sum()}")
        market_cap_positive = df['MarketCap'].where(df['MarketCap'] > 0, np.nan)
        # ОТЛАДКА 7:
        print(f"ОТЛАДКА (Size): Не-NaN в market_cap_positive (до log): {market_cap_positive.notna().sum()}")
        size_calculated = np.log(market_cap_positive)
        # ОТЛАДКА 8:
        print(f"ОТЛАДКА (Size): Не-NaN/inf в size_calculated (после log): {size_calculated.notna().sum()}/{np.isinf(size_calculated).sum()}")
        df['Size'] = size_calculated
        # ОТЛАДКА 9 + 10:
        print(f"Size рассчитан и присвоен ({time.time() - start_time:.2f} сек).")
        print("ОТЛАДКА: Проверка колонки 'Size' СРАЗУ ПОСЛЕ РАСЧЕТА:")
        if 'Size' in df.columns:
            print(f"  Тип данных Size: {df['Size'].dtype}") # ВАЖНО: Проверяем тип
            print(f"  Количество НЕ-NaN Size: {df['Size'].notna().sum()}")
            if df['Size'].notna().any():
                print("  Статистика Size (describe):")
                print(df['Size'].describe())
                print("  Примеры НЕ-NaN значений Size:")
                print(df['Size'].dropna().head())
            else: print("  Все значения Size являются NaN.")
        else: print("  ОШИБКА: Колонка 'Size' не добавилась!")
    else:
        print("WARNING: MarketCap отсутствует или пуст. 'Size' будет NaN.")
        df['Size'] = np.nan
except Exception as e:
     print(f"ОШИБКА при расчете Size: {e}")
     df['Size'] = np.nan
if 'Size' not in df.columns: df['Size'] = np.nan

# --- Создание Placeholder'ов ---
print("\nСоздание placeholder'ов для финансовых признаков (NaN)...")
for feature in FINANCIAL_FEATURES: df[feature] = np.nan
print(f"Колонки {FINANCIAL_FEATURES} установлены в NaN.")


# --- Проверка DataFrame В ПАМЯТИ перед сохранением ---
print("\n--- Проверка DataFrame В ПАМЯТИ (Перед Сохранением) ---")
print("Колонки:", df.columns.tolist())
print("\nИнформация (info):")
df.info(verbose=True, show_counts=True)
print("\nСтатистика Momentum (в памяти):")
if 'Momentum' in df.columns and df['Momentum'].notna().any(): print(df['Momentum'].describe())
else: print("Momentum NaN или отсутствует.")
print("\nСтатистика Size (в памяти):")
if 'Size' in df.columns and df['Size'].notna().any(): print(df['Size'].describe())
else: print("Size NaN или отсутствует.")
print("\nПроверка NaN в финансовых признаках (в памяти):")
fin_nan_counts_mem = df[FINANCIAL_FEATURES].notna().sum()
if fin_nan_counts_mem.sum() == 0: print("Финансовые колонки корректно содержат только NaN.")
else: print(f"ОШИБКА: Финансовые колонки в памяти НЕ пусты! {fin_nan_counts_mem}")


# ==============================================================================
# --- Сохранение Результата Шага 3 ---
# ==============================================================================
print(f"\n--- Сохранение данных после Шага 3 в {OUTPUT_FILE_STEP3} ---")
try:
    df_to_save = df.reset_index()
    print("Колонки для сохранения:", df_to_save.columns.tolist())
    if OUTPUT_FILE_STEP3.endswith('.parquet'):
        df_to_save.to_parquet(OUTPUT_FILE_STEP3)
        print(f"Файл {OUTPUT_FILE_STEP3} сохранен.")
    elif OUTPUT_FILE_STEP3.endswith('.csv'):
        df_to_save.to_csv(OUTPUT_FILE_STEP3, index=False)
        print(f"Файл {OUTPUT_FILE_STEP3} сохранен.")
except Exception as e:
    print(f"\nОШИБКА при сохранении файла: {e}")


# ==============================================================================
# --- Верификация Сохраненного Файла ---
# ==============================================================================
print(f"\n--- Верификация сохраненного файла {OUTPUT_FILE_STEP3} ---")
try:
    if OUTPUT_FILE_STEP3.endswith('.parquet'):
        df_check = pd.read_parquet(OUTPUT_FILE_STEP3)
    elif OUTPUT_FILE_STEP3.endswith('.csv'):
        df_check = pd.read_csv(OUTPUT_FILE_STEP3, parse_dates=['TRADEDATE'])

    print("Файл успешно прочитан.")
    print("Колонки ПОСЛЕ ЗАГРУЗКИ:", df_check.columns.tolist())

    # Проверка типов и установка индекса для удобства сравнения
    if 'TRADEDATE' not in df_check.columns: raise ValueError("Нет TRADEDATE в загруженном файле")
    if not pd.api.types.is_datetime64_any_dtype(df_check['TRADEDATE']):
        df_check['TRADEDATE'] = pd.to_datetime(df_check['TRADEDATE'])
    if 'SECID' not in df_check.columns: raise ValueError("Нет SECID в загруженном файле")
    df_check.set_index(['TRADEDATE', 'SECID'], inplace=True)
    df_check.sort_index(inplace=True)

    # Сравнение с DataFrame в памяти (df)
    print("\n--- Сравнение Данных В Памяти vs Загруженные из Файла ---")

    # Сравнение Momentum
    print("\nСравнение Momentum:")
    momentum_mem_nan = df['Momentum'].notna().sum()
    momentum_file_nan = df_check['Momentum'].notna().sum()
    print(f"  Не-NaN в памяти: {momentum_mem_nan}")
    print(f"  Не-NaN в файле: {momentum_file_nan}")
    if momentum_mem_nan != momentum_file_nan:
         print("  !!! РАСХОЖДЕНИЕ В КОЛИЧЕСТВЕ NaN !!!")
    if momentum_mem_nan > 0:
         print("  Статистика в памяти:")
         print(df['Momentum'].describe().to_string()) # Выводим полностью
         print("  Статистика в файле:")
         print(df_check['Momentum'].describe().to_string()) # Выводим полностью
         # Сравним первые несколько не-NaN значений
         mem_head = df['Momentum'].dropna().head().reset_index(drop=True)
         file_head = df_check['Momentum'].dropna().head().reset_index(drop=True)
         if not mem_head.equals(file_head):
              print("  !!! РАСХОЖДЕНИЕ В ПЕРВЫХ ЗНАЧЕНИЯХ !!!")
              print("  В памяти:\n", mem_head)
              print("  В файле:\n", file_head)

    # Сравнение Size
    print("\nСравнение Size:")
    size_mem_nan = df['Size'].notna().sum()
    size_file_nan = df_check['Size'].notna().sum()
    print(f"  Не-NaN в памяти: {size_mem_nan}")
    print(f"  Не-NaN в файле: {size_file_nan}")
    if size_mem_nan != size_file_nan:
         print("  !!! РАСХОЖДЕНИЕ В КОЛИЧЕСТВЕ NaN !!!")
    if size_mem_nan > 0:
         print("  Статистика в памяти:")
         print(df['Size'].describe().to_string())
         print("  Статистика в файле:")
         print(df_check['Size'].describe().to_string())
         mem_head_size = df['Size'].dropna().head().reset_index(drop=True)
         file_head_size = df_check['Size'].dropna().head().reset_index(drop=True)
         if not mem_head_size.equals(file_head_size):
              print("  !!! РАСХОЖДЕНИЕ В ПЕРВЫХ ЗНАЧЕНИЯХ !!!")
              print("  В памяти:\n", mem_head_size)
              print("  В файле:\n", file_head_size)

    # Сравнение финансовых плейсхолдеров (должны быть все NaN)
    print("\nСравнение Финансовых:")
    fin_mem_nan = df[FINANCIAL_FEATURES].notna().sum().sum()
    fin_file_nan = df_check[FINANCIAL_FEATURES].notna().sum().sum()
    print(f"  Не-NaN в памяти: {fin_mem_nan}")
    print(f"  Не-NaN в файле: {fin_file_nan}")
    if fin_mem_nan != 0 or fin_file_nan != 0:
         print("  !!! ОШИБКА: Финансовые колонки содержат НЕ-NaN значения !!!")

except FileNotFoundError:
    print(f"ОШИБКА: Файл {OUTPUT_FILE_STEP3} не найден для верификации.")
except Exception as e:
    print(f"ОШИБКА при верификации сохраненного файла: {e}")

print("\n--- Завершение Изолированного Шага 3 ---")