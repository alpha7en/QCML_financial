import pandas as pd
import numpy as np
import requests
import apimoex
import statsmodels.api as sm
from datetime import date
import time # Для пауз между запросами
import os # Для работы с файлами

# --- Конфигурация ---
START_DATE = '2020-04-17'
END_DATE = '2025-04-17'
BOARD = 'TQBR'
IMOEX_TICKER = 'IMOEX'
NEWS_DATA_PATH = 'final_dataset.csv' # Ваш файл с новостями
OUTPUT_FILE_STEP3 = 'intermediate_market_data_step3_final.parquet' # Итоговый файл после Шага 3
# OUTPUT_FILE_STEP3 = 'intermediate_market_data_step3_final.csv'

# --- Списки Признаков ---
FEATURES_TO_NORMALIZE = [
    'Accruals', 'EBITDA_to_TEV', 'Momentum', 'Operating_Efficiency',
    'Profit_Margin', 'Size', 'Value', 'Beta'
]
ALL_REGRESSION_FACTORS = FEATURES_TO_NORMALIZE[:]
NEWS_FEATURES = []
FINANCIAL_FEATURES = [ # Признаки, зависящие от фин. данных
    'Accruals', 'EBITDA_to_TEV', 'Operating_Efficiency',
    'Profit_Margin', 'Value'
]
MOMENTUM_SHORT_LAG = 21
MOMENTUM_LONG_LAG = 252

# --- Инициализация ---
session = requests.Session()
print(f"Старт обработки данных для периода: {START_DATE} - {END_DATE}")

# ==============================================================================
# --- Шаг 1: Определение Вселенной Акций MOEX (apimoex only) ---
# ==============================================================================
print("\n--- Шаг 1: Получение тикеров и ISSUESIZE через apimoex ---")
initial_tickers = []
ticker_info_df = pd.DataFrame()
try:
    print(f"Запрос списка бумаг для {BOARD}...")
    board_securities_data = apimoex.get_board_securities(
        session=session, board=BOARD,
        columns=('SECID', 'SHORTNAME', 'SECTYPE', 'LOTSIZE', 'ISSUESIZE')
    )
    board_securities_df = pd.DataFrame(board_securities_data)
    target_sectypes = ['1', '2'] # Обыкновенные и привилегированные акции
    stock_tickers_df = board_securities_df[board_securities_df['SECTYPE'].isin(target_sectypes)].copy()
    if not stock_tickers_df.empty:
        initial_tickers = stock_tickers_df['SECID'].unique().tolist()
        cols_to_extract = ['SECID']
        if 'ISSUESIZE' in stock_tickers_df.columns:
            cols_to_extract.append('ISSUESIZE')
            stock_tickers_df['ISSUESIZE'] = pd.to_numeric(stock_tickers_df['ISSUESIZE'], errors='coerce')
        ticker_info_df = stock_tickers_df[cols_to_extract].drop_duplicates(subset=['SECID']).set_index('SECID')
        ticker_info_df.rename(columns={'ISSUESIZE': 'issuesize'}, inplace=True)
        print(f"Найдено {len(initial_tickers)} тикеров.")
    else:
         print("WARNING: Не найдено подходящих акций.")
except Exception as e:
    print(f"Критическая ошибка при получении списка тикеров: {e}")
    exit()
if not initial_tickers: print("Критическая ошибка: Список тикеров пуст."); exit()

# ==============================================================================
# --- Шаг 2: Сбор Исходных Данных ---
# ==============================================================================
print("\n--- Шаг 2: Сбор исходных рыночных данных ---")

# --- Загрузка IMOEX ---
imoex_data = None
try:
    print(f"Загрузка данных для {IMOEX_TICKER}...")
    imoex_hist = apimoex.get_market_history(session=session, security=IMOEX_TICKER, start=START_DATE, end=END_DATE, engine='stock', market='index', columns=('TRADEDATE', 'CLOSE'))
    imoex_data = pd.DataFrame(imoex_hist)
    if not imoex_data.empty:
        imoex_data['TRADEDATE'] = pd.to_datetime(imoex_data['TRADEDATE'])
        imoex_data = imoex_data.drop_duplicates(subset=['TRADEDATE'], keep='last').set_index('TRADEDATE')
        imoex_data = imoex_data[['CLOSE']].rename(columns={'CLOSE': f'{IMOEX_TICKER}_CLOSE'})
        print(f"Данные {IMOEX_TICKER} загружены.")
    else: imoex_data = None
except Exception as e: imoex_data = None; print(f"Ошибка {IMOEX_TICKER}: {e}")

# --- Загрузка данных по акциям ---
all_market_data = []
print(f"\nНачинаем загрузку истории для {len(initial_tickers)} тикеров...")
for ticker in initial_tickers:
    try:
        stock_hist = apimoex.get_board_history(session=session, security=ticker, start=START_DATE, end=END_DATE, board=BOARD, columns=('TRADEDATE', 'CLOSE', 'VOLUME', 'VALUE'))
        stock_df = pd.DataFrame(stock_hist)
        if not stock_df.empty:
            stock_df['TRADEDATE'] = pd.to_datetime(stock_df['TRADEDATE'])
            stock_df['SECID'] = ticker
            all_market_data.append(stock_df[['TRADEDATE', 'SECID', 'CLOSE', 'VOLUME', 'VALUE']])
    except Exception as e: print(f"Ошибка для {ticker}: {e}")
    time.sleep(0.05) # Небольшая пауза
print("Загрузка истории завершена.")
if not all_market_data: print("\nКРИТИЧЕСКАЯ ОШИБКА: Нет данных по акциям."); exit()

# --- Объединение и УНИКАЛЬНОСТЬ ---
print("\nОбъединение и проверка уникальности...")
market_data_raw_df = pd.concat(all_market_data, ignore_index=True)
duplicates_count = market_data_raw_df.duplicated(subset=['TRADEDATE', 'SECID']).sum()
if duplicates_count > 0:
    print(f"WARNING: Удалено {duplicates_count} дубликатов (TRADEDATE, SECID)...")
    market_data_df = market_data_raw_df.drop_duplicates(subset=['TRADEDATE', 'SECID'], keep='last')
else:
    market_data_df = market_data_raw_df
print(f"Объединенный DataFrame содержит {len(market_data_df)} строк.")

# --- Установка Индекса ---
print("\nУстановка индекса...")
market_data_df.set_index(['TRADEDATE', 'SECID'], inplace=True)
market_data_df.sort_index(inplace=True)
if not market_data_df.index.is_unique: print("КРИТИЧЕСКАЯ ОШИБКА: Индекс НЕ уникален!"); exit()
print("Индекс установлен и уникален.")

# --- Присоединение issuesize и расчет MarketCap ---
print("\nПрисоединение issuesize и расчет MarketCap...")
if not ticker_info_df.empty and 'issuesize' in ticker_info_df.columns:
    market_data_df = market_data_df.join(ticker_info_df[['issuesize']], on='SECID')
    if 'issuesize' in market_data_df.columns and market_data_df['issuesize'].notna().any():
        market_data_df['MarketCap'] = market_data_df['CLOSE'] * market_data_df['issuesize']
        print(f"MarketCap рассчитан.")
    else: market_data_df['MarketCap'] = np.nan
else: market_data_df['MarketCap'] = np.nan
if 'MarketCap' not in market_data_df.columns: market_data_df['MarketCap'] = np.nan

# --- Объединение с IMOEX ---
if imoex_data is not None:
    market_data_df = market_data_df.join(imoex_data, on='TRADEDATE')
    print(f"IMOEX данные присоединены.")
    if f'{IMOEX_TICKER}_CLOSE' not in market_data_df.columns or market_data_df[f'{IMOEX_TICKER}_CLOSE'].isnull().all():
         print(f"WARNING: {IMOEX_TICKER}_CLOSE отсутствует или пуст.")
         if 'Beta' in FEATURES_TO_NORMALIZE: FEATURES_TO_NORMALIZE.remove('Beta')
         if 'Beta' in ALL_REGRESSION_FACTORS: ALL_REGRESSION_FACTORS.remove('Beta')
         print("Признак 'Beta' будет исключен.")
else:
    print("\nIMOEX не загружен. Beta будет исключена.")
    if 'Beta' in FEATURES_TO_NORMALIZE: FEATURES_TO_NORMALIZE.remove('Beta')
    if 'Beta' in ALL_REGRESSION_FACTORS: ALL_REGRESSION_FACTORS.remove('Beta')

# --- PLACEHOLDER: Финансовые данные ---
print("\nPLACEHOLDER: Финансовые данные...")
# На данном этапе финансовые данные не загружаются.
# Соответствующие колонки будут созданы как NaN в Шаге 3.

# --- Загрузка Новостных Признаков (с ДОПОЛНИТЕЛЬНОЙ ОТЛАДКОЙ) ---
print(f"\nЗагрузка новостных признаков из {NEWS_DATA_PATH}...")
NEWS_FEATURES = [] # Сброс перед загрузкой
try:
    # --- ОТЛАДКА 1: Проверка существования файла ---
    if not os.path.exists(NEWS_DATA_PATH):
        print(f"ОШИБКА ОТЛАДКИ: Файл {NEWS_DATA_PATH} не найден!")
        raise FileNotFoundError # Вызываем ошибку, чтобы попасть в except

    news_df = pd.read_csv(NEWS_DATA_PATH)
    print(f"Файл {NEWS_DATA_PATH} прочитан. Строк: {len(news_df)}, Колонки: {news_df.columns.tolist()}")

    # --- ОТЛАДКА 2: Проверка данных СРАЗУ после чтения ---
    print("\nОТЛАДКА: Первые 5 строк news_df ПОСЛЕ ЗАГРУЗКИ:")
    print(news_df.head())
    print("\nОТЛАДКА: Типы данных news_df ПОСЛЕ ЗАГРУЗКИ:")
    print(news_df.dtypes)
    # --- ВАЖНО: Проверяем статистику - есть ли ненулевые значения? ---
    print("\nОТЛАДКА: Статистика news_df ПОСЛЕ ЗАГРУЗКИ (describe):")
    # Ограничим вывод для читаемости, если колонок много
    news_numeric_cols = news_df.select_dtypes(include=np.number).columns
    if len(news_numeric_cols) > 0:
         print(news_df[news_numeric_cols].describe())
         # Проверим, есть ли вообще ненулевые значения
         non_zero_check = (news_df[news_numeric_cols] != 0).any().any()
         print(f"ОТЛАДКА: Есть ли ненулевые числовые значения в news_df? -> {non_zero_check}")
    else:
         print("Числовых колонок для статистики не найдено.")


    # --- Обработка даты ---
    date_col_name = 'date' # Ожидаемое имя колонки с датой
    if date_col_name not in news_df.columns:
         print(f"ОШИБКА ОТЛАДКИ: Ожидаемая колонка '{date_col_name}' не найдена в {NEWS_DATA_PATH}!")
         # Попробуем найти колонку, похожую на дату, если ее имя другое?
         # Это усложнение, пока остановимся на ошибке.
         raise KeyError(f"Колонка '{date_col_name}' не найдена")

    print(f"\nКонвертация колонки '{date_col_name}' в datetime...")
    news_df[date_col_name] = pd.to_datetime(news_df[date_col_name])
    news_df.rename(columns={date_col_name: 'TRADEDATE'}, inplace=True)
    print("Конвертация даты завершена.")

    # --- Проверка наличия колонок ДАННЫХ ---
    news_cols_found = [col for col in news_df.columns if col != 'TRADEDATE']
    if not news_cols_found:
         print(f"WARNING: Файл {NEWS_DATA_PATH} не содержит колонок с данными, кроме TRADEDATE.")
         # Продолжаем, но NEWS_FEATURES останется пустым
    else:
        print(f"Найдены колонки с данными новостей: {news_cols_found}")
        news_df.set_index('TRADEDATE', inplace=True)
        NEWS_FEATURES = news_df.columns.tolist() # Получаем список ПОСЛЕ set_index

        # --- ОТЛАДКА 3: Проверка данных ПЕРЕД слиянием ---
        print("\nОТЛАДКА: Первые 5 строк news_df ПЕРЕД MERGE (индекс TRADEDATE):")
        print(news_df.head())
        print("\nОТЛАДКА: Статистика news_df ПЕРЕД MERGE (describe):")
        if NEWS_FEATURES: # Проверяем числовые колонки из NEWS_FEATURES
             news_numeric_cols_final = news_df[NEWS_FEATURES].select_dtypes(include=np.number).columns
             if len(news_numeric_cols_final) > 0:
                  print(news_df[news_numeric_cols_final].describe())
             else:
                  print("В NEWS_FEATURES нет числовых колонок.")
        else:
            print("Нет новостных колонок для статистики.")


        # --- Слияние (Merge) ---
        print("\nВыполнение слияния (merge) market_data_df с news_df...")
        market_data_df_before_news = len(market_data_df)

        # --- ОТЛАДКА 4: Проверка типов дат перед merge ---
        print(f"ОТЛАДКА: Тип market_data_df.index['TRADEDATE']: {market_data_df.index.get_level_values('TRADEDATE').dtype}")
        print(f"ОТЛАДКА: Тип news_df.index: {news_df.index.dtype}")
        if market_data_df.index.get_level_values('TRADEDATE').dtype != news_df.index.dtype:
            print("ОШИБКА ОТЛАДКИ: Типы дат для слияния не совпадают!")
            # Можно попытаться привести к одному типу, но лучше разобраться в причине

        market_data_df = market_data_df.reset_index().merge(
            news_df, on='TRADEDATE', how='left'
        ).set_index(['TRADEDATE', 'SECID']) # Восстанавливаем индекс
        market_data_df_after_news = len(market_data_df)
        print("Слияние завершено.")

        if market_data_df_before_news != market_data_df_after_news:
             print(f"WARNING: Количество строк изменилось после merge новостей! ({market_data_df_before_news} -> {market_data_df_after_news})")

        # --- ОТЛАДКА 5: Проверка данных ПОСЛЕ слияния ---
        print("\nОТЛАДКА: Статистика новостных колонок в market_data_df ПОСЛЕ MERGE:")
        if NEWS_FEATURES:
             cols_to_describe = [col for col in NEWS_FEATURES if col in market_data_df.columns] # Берем только те, что есть
             if cols_to_describe:
                  # Проверим количество НЕ NaN значений
                  non_nan_counts = market_data_df[cols_to_describe].notna().sum()
                  print("Количество НЕ NaN значений ПОСЛЕ MERGE:")
                  print(non_nan_counts)
                  # Если везде 0 не-NaN, то слияние не нашло совпадений
                  if non_nan_counts.sum() == 0:
                       print("ОШИБКА ОТЛАДКИ: Все новостные значения - NaN после merge. Вероятно, нет совпадающих дат.")
                       # Проверим пересечение дат
                       market_dates = set(market_data_df.index.get_level_values('TRADEDATE').unique())
                       news_dates_index = set(news_df.index.unique())
                       common_dates = market_dates.intersection(news_dates_index)
                       print(f"ОТЛАДКА: Даты в market_data: {len(market_dates)} шт. Пример: {list(market_dates)[:3]}")
                       print(f"ОТЛАДКА: Даты в news_df: {len(news_dates_index)} шт. Пример: {list(news_dates_index)[:3]}")
                       print(f"ОТЛАДКА: Общие даты для слияния: {len(common_dates)} шт.")
                       if not common_dates:
                           print("ОШИБКА ОТЛАДКИ: Нет общих дат между рыночными данными и новостями!")

                  # Выведем describe только если есть не-NaN значения
                  if non_nan_counts.sum() > 0:
                      print("\nСтатистика (describe) новостных колонок ПОСЛЕ MERGE:")
                      print(market_data_df[cols_to_describe].describe())
                  else:
                       # Если все NaN, нет смысла в describe
                       pass

             else:
                  print("Новостные колонки не найдены в market_data_df после merge.")
        else:
             print("Список NEWS_FEATURES пуст.")


        # --- Добавление в регрессоры ---
        if NEWS_FEATURES and all(col in market_data_df.columns for col in NEWS_FEATURES): # Проверяем, что все колонки реально есть
            ALL_REGRESSION_FACTORS.extend(NEWS_FEATURES)
            print(f"Новостные признаки ({len(NEWS_FEATURES)}) добавлены в список регрессоров.")
        else:
             print("Новостные колонки не найдены после merge или список пуст, не добавлены в регрессоры.")

except FileNotFoundError:
    print(f"WARNING: Файл новостей '{NEWS_DATA_PATH}' не найден.")
except KeyError as e:
    print(f"ОШИБКА KeyError при обработке новостей: {e}") # Ловим ошибку отсутствия колонки 'date'
except Exception as e:
    print(f"WARNING: Ошибка при загрузке/обработке новостей: {e}")
# Финальная проверка добавления
if not NEWS_FEATURES or not (NEWS_FEATURES and ALL_REGRESSION_FACTORS[-1] in NEWS_FEATURES):
    print("ПРИМЕЧАНИЕ: Новостные признаки НЕ будут включены в регрессию.")

# ... (Конец Шага 2, переход к Шагу 3 и сохранению) ...

# ==============================================================================
# --- Шаг 3: Расчет Входных Признаков Модели (с Дополнительной Отладкой) ---
# ==============================================================================
print("\n--- Шаг 3: Расчет Входных Признаков Модели ---")
# Используем DataFrame market_data_df из предыдущего шага

# --- Расчет Momentum ---
print("\nРасчет Momentum...")
start_time = time.time()
try:
    # Используем .transform() для эффективности групповых операций shift
    df_grouped = market_data_df.groupby(level='SECID')['CLOSE']
    close_t_minus_21 = df_grouped.transform(lambda x: x.shift(MOMENTUM_SHORT_LAG))
    close_t_minus_252 = df_grouped.transform(lambda x: x.shift(MOMENTUM_LONG_LAG))

    # --- ОТЛАДКА 1: Проверка промежуточных сдвигов ---
    print(f"ОТЛАДКА (Momentum): Не-NaN в close_t_minus_21: {close_t_minus_21.notna().sum()}")
    print(f"ОТЛАДКА (Momentum): Не-NaN в close_t_minus_252: {close_t_minus_252.notna().sum()}")

    # Рассчитываем частное
    ratio = close_t_minus_21 / close_t_minus_252
    # --- ОТЛАДКА 2: Проверка частного перед логарифмом ---
    print(f"ОТЛАДКА (Momentum): Не-NaN в ratio (до log): {ratio.notna().sum()}")
    print(f"ОТЛАДКА (Momentum): inf в ratio (до log): {np.isinf(ratio).sum()}")
    print(f"ОТЛАДКА (Momentum): нули в ratio (до log): {(ratio == 0).sum()}")

    # Применяем логарифм
    momentum_calculated = np.log(ratio)
    # --- ОТЛАДКА 3: Проверка ПОСЛЕ логарифма, ПЕРЕД заменой inf ---
    print(f"ОТЛАДКА (Momentum): Не-NaN в momentum_calculated (после log): {momentum_calculated.notna().sum()}")
    print(f"ОТЛАДКА (Momentum): +/- inf в momentum_calculated (после log): {np.isinf(momentum_calculated).sum()}")

    # Замена inf и присвоение
    market_data_df['Momentum'] = momentum_calculated.replace([np.inf, -np.inf], np.nan)

    # --- ОТЛАДКА 4: Проверка колонки Momentum СРАЗУ ПОСЛЕ ПРИСВОЕНИЯ ---
    print(f"Momentum рассчитан и присвоен ({time.time() - start_time:.2f} сек).")
    print("ОТЛАДКА: Проверка колонки 'Momentum' СРАЗУ ПОСЛЕ РАСЧЕТА:")
    if 'Momentum' in market_data_df.columns:
        print(f"  Тип данных Momentum: {market_data_df['Momentum'].dtype}")
        print(f"  Количество НЕ-NaN Momentum: {market_data_df['Momentum'].notna().sum()}")
        if market_data_df['Momentum'].notna().any():
            print("  Статистика Momentum (describe):")
            print(market_data_df['Momentum'].describe())
            # --- ОТЛАДКА 5: Вывод нескольких НЕ-NaN значений Momentum ---
            print("  Примеры НЕ-NaN значений Momentum:")
            print(market_data_df['Momentum'].dropna().head())
        else:
            print("  Все значения Momentum являются NaN.")
    else:
        print("  ОШИБКА: Колонка 'Momentum' не добавилась в DataFrame!")

except Exception as e:
     print(f"ОШИБКА при расчете Momentum: {e}")
     market_data_df['Momentum'] = np.nan # Заполняем NaN в случае ошибки


# --- Расчет Size ---
print("\nРасчет Size (ln(MarketCap))...")
start_time = time.time()
try:
    if 'MarketCap' in market_data_df.columns and market_data_df['MarketCap'].notna().any():
        # --- ОТЛАДКА 6: Проверка MarketCap перед расчетом Size ---
        print(f"ОТЛАДКА (Size): Не-NaN в MarketCap: {market_data_df['MarketCap'].notna().sum()}")
        print(f"ОТЛАДКА (Size): Нули или меньше в MarketCap: {(market_data_df['MarketCap'] <= 0).sum()}")

        market_cap_positive = market_data_df['MarketCap'].where(market_data_df['MarketCap'] > 0, np.nan)
        # --- ОТЛАДКА 7: Проверка market_cap_positive перед логарифмом ---
        print(f"ОТЛАДКА (Size): Не-NaN в market_cap_positive (до log): {market_cap_positive.notna().sum()}")

        size_calculated = np.log(market_cap_positive)
        # --- ОТЛАДКА 8: Проверка ПОСЛЕ логарифма ---
        print(f"ОТЛАДКА (Size): Не-NaN в size_calculated (после log): {size_calculated.notna().sum()}")
        print(f"ОТЛАДКА (Size): +/- inf в size_calculated (после log): {np.isinf(size_calculated).sum()}")

        market_data_df['Size'] = size_calculated # Присваиваем результат

        # --- ОТЛАДКА 9: Проверка колонки Size СРАЗУ ПОСЛЕ ПРИСВОЕНИЯ ---
        print(f"Size рассчитан и присвоен ({time.time() - start_time:.2f} сек).")
        print("ОТЛАДКА: Проверка колонки 'Size' СРАЗУ ПОСЛЕ РАСЧЕТА:")
        if 'Size' in market_data_df.columns:
            print(f"  Тип данных Size: {market_data_df['Size'].dtype}")
            print(f"  Количество НЕ-NaN Size: {market_data_df['Size'].notna().sum()}")
            if market_data_df['Size'].notna().any():
                print("  Статистика Size (describe):")
                print(market_data_df['Size'].describe())
                # --- ОТЛАДКА 10: Вывод нескольких НЕ-NaN значений Size ---
                print("  Примеры НЕ-NaN значений Size:")
                print(market_data_df['Size'].dropna().head())
            else:
                print("  Все значения Size являются NaN.")
        else:
             print("  ОШИБКА: Колонка 'Size' не добавилась в DataFrame!")

    else:
        print("WARNING: MarketCap отсутствует или пуст. 'Size' будет NaN.")
        market_data_df['Size'] = np.nan
except Exception as e:
     print(f"ОШИБКА при расчете Size: {e}")
     market_data_df['Size'] = np.nan

# Убедимся, что колонка Size существует, даже если расчет не удался
if 'Size' not in market_data_df.columns: market_data_df['Size'] = np.nan


# --- Создание Placeholder'ов для финансовых признаков ---
print("\nСоздание placeholder'ов для финансовых признаков (NaN)...")
for feature in FINANCIAL_FEATURES:
    # --- ОТЛАДКА 11: Проверяем, есть ли колонка ДО присвоения NaN ---
    if feature in market_data_df.columns:
         print(f"ОТЛАДКА: Колонка '{feature}' уже существует. Перезаписываем NaN.")
    else:
         print(f"ОТЛАДКА: Создаем колонку '{feature}' с NaN.")
    market_data_df[feature] = np.nan
# Проверим еще раз после цикла
fin_nan_counts = market_data_df[FINANCIAL_FEATURES].notna().sum()
if fin_nan_counts.sum() == 0:
    print(f"Колонки {FINANCIAL_FEATURES} успешно установлены в NaN.")
else:
     print(f"ОШИБКА: Финансовые колонки НЕ пусты после присвоения NaN! {fin_nan_counts}")


# --- Проверка Результата Шага 3 ---
print("\n--- Завершен Шаг 3: Расчет признаков ---")
print("Итоговые колонки после Шага 3:", market_data_df.columns.tolist())
# ... (проверка наличия колонок) ...
expected_cols_step3 = ['CLOSE', 'VOLUME', 'VALUE', 'issuesize', 'MarketCap', 'IMOEX_CLOSE'] \
                      + NEWS_FEATURES + ['Momentum', 'Size'] + FINANCIAL_FEATURES
if 'Beta' in FEATURES_TO_NORMALIZE: expected_cols_step3.append('Beta') # Добавляем Beta, если она ожидается
else: expected_cols_step3.append('Beta') # Даже если исключена, проверим, не создалась ли случайно

missing_cols = [col for col in expected_cols_step3 if col not in market_data_df.columns]
extra_cols = [col for col in market_data_df.columns if col not in expected_cols_step3]

if missing_cols: print(f"WARNING: Ожидаемые колонки ОТСУТСТВУЮТ: {missing_cols}")
if extra_cols: print(f"WARNING: Найдены НЕОЖИДАННЫЕ колонки: {extra_cols}") # Поможет найти дубли или ошибки

print(f"Размер DataFrame: {market_data_df.shape}")
# print(market_data_df.info())

# --- Сохранение результата ПОСЛЕ Шага 3 (код без изменений) ---
print(f"\nСохранение данных после Шага 3 в {OUTPUT_FILE_STEP3}...")
# ... (try/except блок сохранения с проверкой колонок перед записью) ...
try:
    if not market_data_df.index.is_unique: print("WARNING: Индекс не уникален перед сохранением!")
    df_to_save = market_data_df.reset_index()
    print("Колонки ПЕРЕД сохранением файла Шага 3:", df_to_save.columns.tolist()) # ФИНАЛЬНАЯ ПРОВЕРКА КОЛОНОК
    if OUTPUT_FILE_STEP3.endswith('.parquet'): df_to_save.to_parquet(OUTPUT_FILE_STEP3)
    elif OUTPUT_FILE_STEP3.endswith('.csv'): df_to_save.to_csv(OUTPUT_FILE_STEP3, index=False)
    print("Сохранение завершено.")
except Exception as e: print(f"\nОшибка при сохранении файла: {e}")

# --- Подготовка к следующему шагу ---
print("\nDataFrame готов к Шагу 4.")