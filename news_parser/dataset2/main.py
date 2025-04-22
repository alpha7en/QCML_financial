import pandas as pd
import numpy as np
import requests
import apimoex
import statsmodels.api as sm
from datetime import date
import time # Для пауз между запросами
import os # Для работы с файлами

# --- Конфигурация (без изменений) ---
START_DATE = '2020-04-17'
END_DATE = '2025-04-17'
BOARD = 'TQBR' # Основной режим торгов T+2
IMOEX_TICKER = 'IMOEX'
NEWS_DATA_PATH = 'final_dataset.csv' # Укажите путь к вашему файлу
# OUTPUT_DATASET_PATH = 'moex_qc_dataset.parquet' # Выходной файл датасета (пока не используется)

# --- Списки Признаков (Обновлены с учетом отсутствия Sector) ---
# Short Utilization и GICS/Sector исключены
# Accruals, EBITDA_to_TEV, OpEff, ProfitMargin, Value - зависят от фин.данных (будут NaN пока)
FEATURES_TO_NORMALIZE = [
    'Accruals', 'EBITDA_to_TEV', 'Momentum', 'Operating_Efficiency',
    'Profit_Margin', 'Size', 'Value', 'Beta'
]
# ALL_REGRESSION_FACTORS не будет содержать GICS dummies
ALL_REGRESSION_FACTORS = FEATURES_TO_NORMALIZE[:] # Копируем нормализуемые
# Новостные признаки добавятся позже, GICS Dummies не будет

# --- Инициализация ---
session = requests.Session()
print(f"Старт обработки данных для периода: {START_DATE} - {END_DATE}")

# --- Шаг 1: Определение Вселенной Акций MOEX (apimoex only) ---
print("\nШаг 1: Определение вселенной акций MOEX и получение ISSUESIZE через apimoex...")

initial_tickers = []
ticker_info_df = pd.DataFrame() # DataFrame для хранения info (issuesize)

try:
    # Запрашиваем ISSUESIZE. Убираем SECTORID и CURRENCYID из запроса.
    print(f"Запрос списка бумаг и их параметров для режима {BOARD}...")
    board_securities_data = apimoex.get_board_securities(
        session=session,
        board=BOARD,
        columns=('SECID', 'SHORTNAME', 'SECTYPE', 'LOTSIZE', 'ISSUESIZE') # Запрашиваем ISSUESIZE
    )
    board_securities_df = pd.DataFrame(board_securities_data)
    print(f"Получено {len(board_securities_df)} записей из справочника {BOARD}.")

    # --- Фильтрация акций ---
    # Убедитесь, что эти коды SECTYPE верны для MOEX!
    target_sectypes = ['1', '2'] # Пример: обыкновенные и привилегированные
    stock_tickers_df = board_securities_df[board_securities_df['SECTYPE'].isin(target_sectypes)].copy()
    print(f"Отфильтровано {len(stock_tickers_df)} акций (типы: {target_sectypes}).")

    # --- Убрана Фильтрация по Валюте ---

    if stock_tickers_df.empty:
         print(f"WARNING: Не найдено акций с SECTYPE {target_sectypes} в режиме {BOARD}.")
    else:
        initial_tickers = stock_tickers_df['SECID'].unique().tolist()
        print(f"Найдено {len(initial_tickers)} потенциальных тикеров для загрузки.")

        # --- Подготовка ticker_info_df (только ISSUESIZE) ---
        cols_to_extract = ['SECID']
        if 'ISSUESIZE' in stock_tickers_df.columns:
            cols_to_extract.append('ISSUESIZE')
            stock_tickers_df['ISSUESIZE'] = pd.to_numeric(stock_tickers_df['ISSUESIZE'], errors='coerce')
            print(f"Колонка 'ISSUESIZE' найдена. non-NaN: {stock_tickers_df['ISSUESIZE'].notna().sum()}")
        else:
            print("WARNING: Колонка 'ISSUESIZE' не найдена в ответе get_board_securities.")
            stock_tickers_df['ISSUESIZE'] = np.nan

        # Создаем ticker_info_df с индексом SECID
        ticker_info_df = stock_tickers_df[cols_to_extract].drop_duplicates(subset=['SECID']).set_index('SECID')
        ticker_info_df.rename(columns={'ISSUESIZE': 'issuesize'}, inplace=True) # Переименуем

except Exception as e:
    print(f"Критическая ошибка при получении списка тикеров через apimoex: {e}")
    initial_tickers = []
    ticker_info_df = pd.DataFrame()

# --- Final Check ---
if not initial_tickers:
    print("\nКРИТИЧЕСКАЯ ОШИБКА: Не удалось получить список тикеров. Выполнение прервано.")
    exit()
else:
     print(f"\nИтоговый список тикеров ({len(initial_tickers)} шт.): {initial_tickers[:5]}...")
     print("\nПример статической информации (ticker_info_df с issuesize):")
     print(ticker_info_df.head())


# --- Шаг 2: Сбор Исходных Данных ---
print("\nШаг 2: Сбор исходных рыночных данных...")

# --- Загрузка данных по IMOEX (код без изменений) ---
imoex_data = None
try:
    print(f"Загрузка данных для индекса {IMOEX_TICKER}...")
    # ... (код загрузки IMOEX) ...
    imoex_hist = apimoex.get_market_history(
        session=session, security=IMOEX_TICKER, start=START_DATE, end=END_DATE,
        engine='stock', market='index', columns=('TRADEDATE', 'CLOSE')
    )
    imoex_data = pd.DataFrame(imoex_hist)
    if not imoex_data.empty:
        imoex_data['TRADEDATE'] = pd.to_datetime(imoex_data['TRADEDATE'])
        imoex_data = imoex_data.drop_duplicates(subset=['TRADEDATE'], keep='last')
        imoex_data.set_index('TRADEDATE', inplace=True)
        imoex_data = imoex_data[['CLOSE']].rename(columns={'CLOSE': f'{IMOEX_TICKER}_CLOSE'})
        print(f"Загружены данные для {IMOEX_TICKER} ({len(imoex_data)} дней).")
    else:
        print(f"Для {IMOEX_TICKER} не найдено данных.")
        imoex_data = None
except Exception as e:
    print(f"Ошибка при загрузке данных для {IMOEX_TICKER}: {e}")
    imoex_data = None


# --- Загрузка данных по акциям ---
all_market_data = []
valid_tickers = []
PAUSE_DURATION = 0.1
print(f"\nНачинаем загрузку истории для {len(initial_tickers)} тикеров...")
for i, ticker in enumerate(initial_tickers):
    # ... (цикл загрузки без изменений) ...
    if not ticker: continue
    print(f"Загрузка данных для {ticker} ({i+1}/{len(initial_tickers)})...", end=" ")
    try:
        stock_hist = apimoex.get_board_history(
            session=session, security=ticker, start=START_DATE, end=END_DATE,
            board=BOARD, columns=('TRADEDATE', 'CLOSE', 'VOLUME', 'VALUE')
        )
        stock_df = pd.DataFrame(stock_hist)
        if not stock_df.empty:
            stock_df['TRADEDATE'] = pd.to_datetime(stock_df['TRADEDATE'])
            stock_df['SECID'] = ticker
            # Отбираем только нужные колонки СРАЗУ
            all_market_data.append(stock_df[['TRADEDATE', 'SECID', 'CLOSE', 'VOLUME', 'VALUE']])
            valid_tickers.append(ticker)
            #print(f"Успешно ({len(stock_df)} строк).")
        else:
            print(f"Нет данных.")
    except Exception as e:
        print(f"Ошибка для {ticker}: {e}") # Сообщаем об ошибке
    time.sleep(PAUSE_DURATION)
print(f"\nЗавершена загрузка истории. Успешно для {len(valid_tickers)} тикеров.")

if not all_market_data:
    print("\nКРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить исторические данные ни для одной акции. Выполнение прервано.")
    exit()

# --- Объединение и **ОБЕСПЕЧЕНИЕ УНИКАЛЬНОСТИ** ---
print("\nОбъединение данных по акциям...")
market_data_raw_df = pd.concat(all_market_data, ignore_index=True)
print(f"Объединенный DataFrame содержит {len(market_data_raw_df)} строк.")

# --- ПРОВЕРКА и УДАЛЕНИЕ ДУБЛИКАТОВ (TRADEDATE, SECID) ---
initial_rows = len(market_data_raw_df)
duplicates_count = market_data_raw_df.duplicated(subset=['TRADEDATE', 'SECID']).sum()
if duplicates_count > 0:
    print(f"WARNING: Обнаружено {duplicates_count} дубликатов пар (TRADEDATE, SECID). Удаляем дубликаты, оставляя последнюю запись...")
    market_data_df = market_data_raw_df.drop_duplicates(subset=['TRADEDATE', 'SECID'], keep='last')
    print(f"Осталось {len(market_data_df)} строк после удаления дубликатов.")
else:
    print("Дубликатов (TRADEDATE, SECID) не обнаружено.")
    market_data_df = market_data_raw_df # Переименовываем для дальнейшей работы

# --- Установка Индекса ---
print("\nУстановка индекса ['TRADEDATE', 'SECID']...")
# Теперь мы уверены (или должны быть уверены), что дубликатов нет
market_data_df.set_index(['TRADEDATE', 'SECID'], inplace=True)
market_data_df.sort_index(inplace=True)
# Проверка уникальности индекса ПОСЛЕ установки
if not market_data_df.index.is_unique:
     print("КРИТИЧЕСКАЯ ОШИБКА: Индекс НЕ уникален даже после удаления дубликатов! Проблема с логикой.")
     # Дополнительно можно вывести дубликаты для анализа:
     # print(market_data_df[market_data_df.index.duplicated(keep=False)])
     exit()
else:
     print("Индекс успешно установлен и является уникальным.")


# --- Присоединение статической информации (ISSUESIZE) и расчет MarketCap ---
print("\nПрисоединение issuesize и расчет MarketCap...")
market_data_df['Sector'] = 'Unknown' # Устанавливаем сектор
if not ticker_info_df.empty and 'issuesize' in ticker_info_df.columns:
    market_data_df = market_data_df.join(ticker_info_df[['issuesize']], on='SECID') # Используем join по SECID
    if 'issuesize' in market_data_df.columns and market_data_df['issuesize'].notna().any():
        market_data_df['MarketCap'] = market_data_df['CLOSE'] * market_data_df['issuesize']
        print(f"MarketCap рассчитан.")
    else:
        market_data_df['MarketCap'] = np.nan
        print("INFO: MarketCap не рассчитан (issuesize отсутствует или NaN).")
else:
    market_data_df['MarketCap'] = np.nan
    print("INFO: MarketCap не рассчитан (ticker_info_df пуст).")


# --- Объединение с IMOEX ---
if imoex_data is not None:
    market_data_df = market_data_df.join(imoex_data, on='TRADEDATE') # Join по уровню индекса TRADEDATE
    print(f"\nРазмер DataFrame после присоединения IMOEX: {market_data_df.shape}")
    # ... (проверка наличия колонки IMOEX и исключение Beta, если нужно) ...
    if f'{IMOEX_TICKER}_CLOSE' not in market_data_df.columns or market_data_df[f'{IMOEX_TICKER}_CLOSE'].isnull().all():
         print(f"WARNING: Колонка {IMOEX_TICKER}_CLOSE не найдена или пуста после join.")
         if 'Beta' in FEATURES_TO_NORMALIZE: FEATURES_TO_NORMALIZE.remove('Beta')
         if 'Beta' in ALL_REGRESSION_FACTORS: ALL_REGRESSION_FACTORS.remove('Beta')
         print("Признак 'Beta' будет исключен.")
else:
    # ... (исключение Beta, если imoex_data is None) ...
    print("\nДанные IMOEX не загружены. Beta не может быть рассчитана.")
    if 'Beta' in FEATURES_TO_NORMALIZE: FEATURES_TO_NORMALIZE.remove('Beta')
    if 'Beta' in ALL_REGRESSION_FACTORS: ALL_REGRESSION_FACTORS.remove('Beta')
    print("Признак 'Beta' будет исключен.")


# --- PLACEHOLDER: Загрузка Финансовой Отчетности (без изменений) ---
print("\nPLACEHOLDER: Финансовые данные...")
# ...
print("ПРИМЕЧАНИЕ: Финансовые данные не загружены.")


# --- Загрузка Глобальных Новостных Признаков ---
print(f"\nЗагрузка новостных признаков из {NEWS_DATA_PATH}...")
NEWS_FEATURES = []
try:
    # ... (код загрузки news_df без изменений) ...
    news_df = pd.read_csv(NEWS_DATA_PATH)
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df.rename(columns={'date': 'TRADEDATE'}, inplace=True)
    news_df.set_index('TRADEDATE', inplace=True)
    NEWS_FEATURES = news_df.columns.tolist()

    # Используем merge, так как market_data_df имеет MultiIndex
    # Сбрасываем индекс market_data_df временно для merge
    market_data_df = market_data_df.reset_index().merge(
        news_df, on='TRADEDATE', how='left'
    ).set_index(['TRADEDATE', 'SECID']) # Восстанавливаем индекс

    print(f"Загружены и присоединены новостные признаки. Колонки: {NEWS_FEATURES}")
    print(f"Размер DataFrame после новостей: {market_data_df.shape}")
    if NEWS_FEATURES:
         ALL_REGRESSION_FACTORS.extend(NEWS_FEATURES) # Добавляем к списку регрессоров
except FileNotFoundError:
    print(f"WARNING: Файл новостей '{NEWS_DATA_PATH}' не найден.")
except Exception as e:
    print(f"WARNING: Ошибка при загрузке/обработке новостей: {e}")
if not NEWS_FEATURES:
    print("ПРИМЕЧАНИЕ: Новостные признаки не будут включены в регрессию.")


# --- Промежуточный результат Шага 2 ---
print("\nПромежуточный результат после Шага 2 (проверка уникальности):")
print(market_data_df.info(verbose=True, show_counts=True)) # Используем info без reset_index
print("\nПример данных (head):")
print(market_data_df.head())


# --- Сохранение промежуточного результата (Используем явный reset_index) ---
OUTPUT_FILE = 'intermediate_market_data_step2_unique_check.parquet'
# OUTPUT_FILE = 'intermediate_market_data_step2_unique_check.csv'
try:
    # ПРОВЕРКА перед reset_index:
    if not market_data_df.index.is_unique:
         print("КРИТИЧЕСКАЯ ОШИБКА перед сохранением: Индекс все еще не уникален!")
         # exit() # Можно остановить выполнение
    else:
         print("\nПроверка перед сохранением: Индекс уникален.")

    df_to_save = market_data_df.reset_index()
    print("Колонки DataFrame ПЕРЕД сохранением:", df_to_save.columns.tolist())

    print(f"\nСохранение данных в {OUTPUT_FILE}...")
    if OUTPUT_FILE.endswith('.parquet'):
        df_to_save.to_parquet(OUTPUT_FILE)
    elif OUTPUT_FILE.endswith('.csv'):
         df_to_save.to_csv(OUTPUT_FILE, index=False)
    print("Сохранение завершено.")

except Exception as e:
    print(f"\nОшибка при сохранении файла: {e}")


# --- Загрузка для следующего шага (с проверками) ---
# ... (Код загрузки из предыдущего ответа с проверками колонок и типов) ...
df_step3 = None
try:
    print(f"\nЗагрузка сохраненных данных из {OUTPUT_FILE} для следующего шага...")
    if OUTPUT_FILE.endswith('.parquet'):
        df = pd.read_parquet(OUTPUT_FILE)
    elif OUTPUT_FILE.endswith('.csv'):
         df = pd.read_csv(OUTPUT_FILE, parse_dates=['TRADEDATE'])

    print("Колонки DataFrame ПОСЛЕ загрузки:", df.columns.tolist()) # Проверка

    if 'TRADEDATE' not in df.columns: raise ValueError("Колонка 'TRADEDATE' отсутствует.")
    if not pd.api.types.is_datetime64_any_dtype(df['TRADEDATE']):
         df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])

    if 'SECID' not in df.columns: raise ValueError("Колонка 'SECID' отсутствует.")

    print("Установка индекса...")
    df.set_index(['TRADEDATE', 'SECID'], inplace=True)
    df.sort_index(inplace=True)

    # Финальная проверка уникальности индекса ПОСЛЕ загрузки
    if not df.index.is_unique:
        print("WARNING: Индекс НЕ уникален ПОСЛЕ загрузки! Возможно, проблема в файле.")
        # df = df[~df.index.duplicated(keep='last')] # Можно попытаться исправить
    else:
         print("Данные успешно загружены, индекс установлен и уникален.")
         df_step3 = df # Передаем в переменную для след. шага

except FileNotFoundError:
    print(f"\nОШИБКА: Файл {OUTPUT_FILE} не найден.")
except Exception as e:
    print(f"\nОШИБКА при загрузке/обработке файла {OUTPUT_FILE}: {e}")

if df_step3 is None:
     print("\nКРИТИЧЕСКАЯ ОШИБКА: Не удалось подготовить данные для следующего шага. Выход.")
     exit()
else:
      print("\nDataFrame готов к Шагу 3.")