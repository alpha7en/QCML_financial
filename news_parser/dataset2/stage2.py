import pandas as pd
import numpy as np
import time

# --- Константы для расчета ---
MOMENTUM_SHORT_LAG = 21 # t-21 торговый день
MOMENTUM_LONG_LAG = 252 # t-252 торговый день
FINANCIAL_FEATURES = [
    'Accruals', 'EBITDA_to_TEV', 'Operating_Efficiency',
    'Profit_Margin', 'Value'
]

# --- Загрузка данных из предыдущего шага ---
INPUT_FILE = 'intermediate_market_data_step2_unique_check.parquet'
# INPUT_FILE = 'intermediate_market_data_step2_apimoex_no_sectors.csv' # Если сохраняли в CSV
OUTPUT_FILE_STEP3 = 'intermediate_market_data_step3.parquet'
# OUTPUT_FILE_STEP3 = 'intermediate_market_data_step3.csv'

print(f"\n--- Шаг 3: Расчет Входных Признаков Модели ---")
print(f"Загрузка данных из {INPUT_FILE}...")

try:
    if INPUT_FILE.endswith('.parquet'):
        df = pd.read_parquet(INPUT_FILE)
    elif INPUT_FILE.endswith('.csv'):
         df = pd.read_csv(INPUT_FILE, parse_dates=['TRADEDATE'])

    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    df.set_index(['TRADEDATE', 'SECID'], inplace=True)
    df.sort_index(inplace=True)
    print(f"Данные загружены. Строк: {len(df)}. Колонки: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"\nКРИТИЧЕСКАЯ ОШИБКА: Файл {INPUT_FILE} не найден. Невозможно продолжить.")
    exit()
except Exception as e:
    print(f"\nКРИТИЧЕСКАЯ ОШИБКА при загрузке файла {INPUT_FILE}: {e}")
    exit()

# --- Расчет Momentum ---
print("\nРасчет Momentum...")
start_time = time.time()

# Группируем по тикеру и сдвигаем цены закрытия
# shift() в pandas работает по индексированным строкам (т.е. по торговым дням в нашем случае)
df['Close_t_minus_21'] = df.groupby(level='SECID')['CLOSE'].shift(MOMENTUM_SHORT_LAG)
df['Close_t_minus_252'] = df.groupby(level='SECID')['CLOSE'].shift(MOMENTUM_LONG_LAG)

# Рассчитываем логарифмическую доходность
# Используем np.log для натурального логарифма
df['Momentum'] = np.log(df['Close_t_minus_21'] / df['Close_t_minus_252'])

# Обработка бесконечностей и NaN, которые могут возникнуть при делении на ноль или логарифмировании нуля/отрицательного числа
df['Momentum'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Удаляем вспомогательные колонки
df.drop(columns=['Close_t_minus_21', 'Close_t_minus_252'], inplace=True)

momentum_non_nan = df['Momentum'].notna().sum()
print(f"Momentum рассчитан. non-NaN: {momentum_non_nan} ({time.time() - start_time:.2f} сек).")
if momentum_non_nan == 0 and len(df) > MOMENTUM_LONG_LAG:
     print("WARNING: Ни одного значения Momentum не рассчитано. Проверьте лаги и наличие данных CLOSE.")

# --- Расчет Size ---
print("\nРасчет Size (ln(MarketCap))...")
start_time = time.time()

if 'MarketCap' in df.columns and df['MarketCap'].notna().any():
    # Используем np.log для натурального логарифма
    # Добавляем small epsilon для предотвращения log(0), если MarketCap может быть 0
    epsilon = 1e-10
    df['Size'] = np.log(df['MarketCap'].replace(0, np.nan) + epsilon) # Заменяем 0 на NaN перед логарифмом
    # Обработка -inf (если MarketCap был NaN или отрицательным)
    df['Size'].replace([np.inf, -np.inf], np.nan, inplace=True)
    size_non_nan = df['Size'].notna().sum()
    print(f"Size рассчитан. non-NaN: {size_non_nan} ({time.time() - start_time:.2f} сек).")
else:
    print("WARNING: Колонка 'MarketCap' отсутствует или пуста. Признак 'Size' будет NaN.")
    df['Size'] = np.nan

# --- Создание Placeholder'ов для признаков, зависящих от фин. данных ---
print("\nСоздание placeholder'ов для признаков, зависящих от фин. данных (будут NaN)...")
for feature in FINANCIAL_FEATURES:
    df[feature] = np.nan
    print(f"  Создана колонка '{feature}' (NaN)")

# --- Комментарии по расчету финансовых признаков (если бы данные были) ---
# df['Accruals'] = (-1) * df.groupby(level='SECID')[['TOTALASSETS', 'WORKINGCAPITAL', 'TOTALLIABILITIES', 'LONGTERMINVESTMENTS', 'LONGTERMDEBT']].apply(
#     lambda x: (x['TOTALASSETS'] - x['WORKINGCAPITAL'] - x['TOTALLIABILITIES'] - x['LONGTERMINVESTMENTS'] + x['LONGTERMDEBT']).diff(4) / x['TOTALASSETS'].shift(4) # Примерная логика, требует квартальных данных
# )
# df['TotalDebt'] = df['LONGTERMDEBT'] + df['SHORTTERMDEBT'] # Нужны STD
# df['TEV'] = df['MarketCap'] + df['TotalDebt'] - df['CASHEQUIVALENTS']
# df['EBITDA_L4Q'] = df.groupby(level='SECID')['EBITDA'].rolling(4).sum().reset_index(level=0, drop=True) # Требует квартальных данных
# df['EBITDA_to_TEV'] = df['EBITDA_L4Q'] / df['TEV']
# df['Revenues_L4Q'] = df.groupby(level='SECID')['REVENUES'].rolling(4).sum().reset_index(level=0, drop=True) # Требует квартальных данных
# df['Operating_Efficiency'] = df['Revenues_L4Q'] / df['TOTALASSETS'] # TA - последний доступный? Или средний за период? Уточнить по статье.
# df['NetIncome_L4Q'] = df.groupby(level='SECID')['NETINCOME'].rolling(4).sum().reset_index(level=0, drop=True) # Требует квартальных данных
# df['Profit_Margin'] = df['NetIncome_L4Q'] / df['Revenues_L4Q']
# df['Value'] = df['Revenues_L4Q'] / df['MarketCap']

print("\nПРИМЕЧАНИЕ: Признаки Accruals, EBITDA_to_TEV, Operating_Efficiency, Profit_Margin, Value установлены в NaN из-за отсутствия интегрированных финансовых данных.")

# --- Промежуточный результат Шага 3 ---
print("\nПромежуточный результат после Шага 3:")
# Показываем информацию, включая новые колонки
print(df.reset_index().info(verbose=True, show_counts=True))
print("\nПример данных (head) с новыми признаками:")
print(df[['CLOSE', 'MarketCap', 'Momentum', 'Size', 'Value']].head()) # Value будет NaN
print("\nПример данных (sample) с новыми признаками:")
# Выбираем строки, где Momentum и Size не NaN (если такие есть)
sample_df = df.dropna(subset=['Momentum', 'Size'])
if not sample_df.empty:
    print(sample_df[['CLOSE', 'MarketCap', 'Momentum', 'Size', 'Value']].sample(min(5, len(sample_df))))
else:
    print("Не найдено строк с рассчитанными Momentum и Size для sample.")
    print(df[['CLOSE', 'MarketCap', 'Momentum', 'Size', 'Value']].sample(min(5, len(df))))


# --- Сохранение промежуточного результата ---
print(f"\nСохранение промежуточных данных после Шага 3 в {OUTPUT_FILE_STEP3}...")
try:
    if not df.index.is_unique:
         print("WARNING: Обнаружен неуникальный индекс перед сохранением. Удаляем дубликаты...")
         df = df[~df.index.duplicated(keep='last')]

    if OUTPUT_FILE_STEP3.endswith('.parquet'):
        df.reset_index().to_parquet(OUTPUT_FILE_STEP3)
    elif OUTPUT_FILE_STEP3.endswith('.csv'):
         df.reset_index().to_csv(OUTPUT_FILE_STEP3, index=False)
    print("Сохранение завершено.")
except ImportError:
     print(f"ОШИБКА: Для сохранения в Parquet установите 'pyarrow'. (`pip install pyarrow`)")
except Exception as e:
    print(f"Ошибка при сохранении промежуточного файла: {e}")

# Загрузим для следующего шага
try:
    print(f"\nЗагрузка сохраненных данных из {OUTPUT_FILE_STEP3} для следующего шага...")
    if OUTPUT_FILE_STEP3.endswith('.parquet'):
        df_step3 = pd.read_parquet(OUTPUT_FILE_STEP3)
    elif OUTPUT_FILE_STEP3.endswith('.csv'):
         df_step3 = pd.read_csv(OUTPUT_FILE_STEP3, parse_dates=['TRADEDATE'])

    df_step3['TRADEDATE'] = pd.to_datetime(df_step3['TRADEDATE'])
    df_step3.set_index(['TRADEDATE', 'SECID'], inplace=True)
    df_step3.sort_index(inplace=True)
    print("Данные после Шага 3 успешно загружены.")
except FileNotFoundError:
    print(f"ОШИБКА: Файл {OUTPUT_FILE_STEP3} не найден. Следующие шаги будут работать с данными в памяти (df).")
    df_step3 = df # Используем данные из памяти
except Exception as e:
    print(f"ОШИБКА при загрузке файла {OUTPUT_FILE_STEP3}: {e}. Следующие шаги будут работать с данными в памяти (df).")
    df_step3 = df # Используем данные из памяти