# -*- coding: utf-8 -*-
import requests
import apimoex
from apimoex import ISSClient # Убедимся, что ISSClient импортирован
import pandas as pd
import numpy as np
import pandas_ta as ta # Для ADX
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# --- Параметры ---
START_DATE_REQUEST = '2019-11-25' # Начнем раньше для расчета скользящих средних (63 дня + запас)
START_DATE_OUTPUT = '2020-04-17'
END_DATE_OUTPUT = '2025-04-21'
OUTPUT_CSV_FILE = '../merge/moex_metrics_final_v4.csv'  # Новое имя файла
USD_RUB_TICKER_OLD = 'USD000UTSTOM' # Старый, до ~августа 2024
USD_RUB_TICKER_NEW = 'USDRUBTOMOTC' # Новый, с июня 2024
USD_RUB_SWITCH_DATE = '2024-06-13' # Дата начала данных для USDRUBTOMOTC


# Идентификаторы инструментов на MOEX
IMOEX_TICKER = 'IMOEX'
RTSOG_TICKER = 'RTSOG'
USD_RUB_TICKER = 'USDRUBTOMOTC'
RGBITR_TICKER = 'RGBITR'
RUCBTRNS_TICKER = 'RUCBTRNS'

# Окна для расчетов
WINDOW_21 = 21
WINDOW_63 = 63
ADX_WINDOW = 14

# --- Функции ---

def get_market_data(session, ticker, start_date, end_date):
    """
    Загружает исторические данные для одного инструмента, используя оптимальный метод:
    - ISSClient для USDRUBTOMOTC.
    - get_board_history для USD000UTSTOM.
    - get_market_history для индексов.
    """
    print(f"--- Загрузка для {ticker} ({start_date} - {end_date}) ---")
    data = []
    df = pd.DataFrame()
    engine = 'stock' # Значения по умолчанию
    market = 'shares'
    board = None
    columns = None
    func_to_use = None # Явно определим функцию

    # 1. Определение метода и параметров по тикеру
    if ticker in ['IMOEX', 'RGBITR', 'RUCBTRNS', RTSOG_TICKER]:
        engine = 'stock'
        market = 'index'
        func_to_use = apimoex.get_market_history
        if ticker == 'IMOEX':
            print(f"Параметры для {ticker}: func=get_market_history, engine={engine}, market={market} (запрос OHLCV)")
            columns = ('TRADEDATE', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME', 'VALUE', 'NUMTRADES', 'WAPRICE')
        else:
             print(f"Параметры для {ticker}: func=get_market_history, engine={engine}, market={market} (запрос CLOSE/VALUE)")
             columns = ('TRADEDATE', 'CLOSE', 'VALUE', 'OPEN', 'HIGH', 'LOW')
        try:
            data = func_to_use(session, security=ticker, start=start_date, end=end_date,
                               engine=engine, market=market, columns=columns)
            # Откат для IMOEX
            if not data and ticker == 'IMOEX' and columns and 'OPEN' in columns:
                 print(f"Тикер {ticker}: Попытка отката - Запрос только CLOSE/VALUE")
                 fallback_cols = ('TRADEDATE', 'CLOSE', 'VALUE')
                 data = func_to_use(session, security=ticker, start=start_date, end=end_date,
                                      market='index', engine='stock', columns=fallback_cols)
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА при запросе {func_to_use.__name__} для {ticker}: {e}")
            data = []

    elif ticker == USD_RUB_TICKER_NEW: # --- Логика для НОВОГО тикера ---
        print(f"Параметры для {ticker}: Использование ISSClient с URL currency/otcindices/COTF")
        request_url = f"https://iss.moex.com/iss/history/engines/currency/markets/otcindices/boards/COTF/securities/{ticker}.json"
        params = {"from": start_date, "till": end_date, "history.columns": "TRADEDATE,OPEN,LOW,HIGH,CLOSE"}
        print(f"URL: {request_url}")
        print(f"Params для ISSClient: {params}")
        try:
            client = ISSClient(session, request_url, params)
            response_data = client.get_all()
            if 'history' in response_data and isinstance(response_data['history'], list):
                data = response_data['history']
                func_to_use = ISSClient # Указываем что сработал ISSClient
                print(f"Данные получены из блока 'history'.")
            else:
                print("Блок 'history' не найден или пуст в ответе ISSClient.")
                data = []
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА при запросе данных для {ticker} через ISSClient: {e}")
            data = []

    elif ticker == USD_RUB_TICKER_OLD: # --- Логика для СТАРОГО тикера ---
         engine = 'currency'
         market = 'selt'
         board = 'CETS'
         func_to_use = apimoex.get_board_history
         print(f"Параметры для {ticker}: func=get_board_history, engine={engine}, market={market}, board={board}")
         columns = ('TRADEDATE', 'WAPRICE', 'CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'VALUE', 'NUMTRADES')
         try:
            data = func_to_use(session, security=ticker, start=start_date, end=end_date,
                               engine=engine, market=market, board=board, columns=tuple(columns))
         except Exception as e:
             print(f"КРИТИЧЕСКАЯ ОШИБКА при запросе {func_to_use.__name__} для {ticker}: {e}")
             data = []

    else: # Логика по умолчанию (например, для акций)
        engine = 'stock'
        market = 'shares'
        board = 'TQBR'
        func_to_use = apimoex.get_board_history
        print(f"Параметры для {ticker} (по умолчанию): func=get_board_history, engine={engine}, market={market}, board={board}")
        columns = ('TRADEDATE', 'CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'VALUE')
        try:
             data = func_to_use(session, security=ticker, start=start_date, end=end_date,
                                engine=engine, market=market, board=board, columns=columns)
        except Exception as e:
             print(f"КРИТИЧЕСКАЯ ОШИБКА при запросе {func_to_use.__name__} для {ticker}: {e}")
             data = []


    # --- Постобработка и создание DataFrame ---
    if not data:
        print(f"--- Финальный результат: Данные для {ticker} не получены. ---")
        return pd.DataFrame()

    source_name = "ISSClient" if func_to_use == ISSClient else (func_to_use.__name__ if func_to_use else "Неизвестно")
    print(f"Данные для {ticker} получены ({len(data)} записей) с помощью '{source_name}'. Начинаем обработку DataFrame...")
    try:
        df = pd.DataFrame(data)
        if df.empty: return df

        # 1. Дата как индекс
        date_col = None
        for col_opt in ['TRADEDATE', 'tradedate']:
            if col_opt in df.columns: date_col = col_opt; break
        if not date_col: print(f"Критическая ошибка: не найдена TRADEDATE для {ticker}."); return pd.DataFrame()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col); df.index.name = 'TRADEDATE'
        print(f"Индекс TRADEDATE установлен.")

        # 2. Определение и проверка колонки цены
        price_col_found = None
        # Разный приоритет для старого и нового тикера
        price_options = ['CLOSE', 'OPEN', 'HIGH', 'LOW'] if ticker == USD_RUB_TICKER_NEW else ['WAPRICE', 'CLOSE', 'OPEN', 'HIGH', 'LOW', 'VALUE']
        for col_name in price_options:
            if col_name in df.columns:
                if pd.api.types.is_numeric_dtype(df[col_name]):
                     if df[col_name].notna().any(): price_col_found = col_name; break
                else:
                     try:
                          temp_series = pd.to_numeric(df[col_name], errors='coerce')
                          if temp_series.notna().any(): price_col_found = col_name; df[col_name] = temp_series; break
                     except: pass
        if not price_col_found: print(f"Критическая ошибка: не найдена колонка цены для {ticker}."); return pd.DataFrame()
        print(f"Выбрана колонка цены: '{price_col_found}'")

        # 3. Переименование и выбор колонок
        rename_map = {}
        col_standard_map = {
            'OPEN': 'OPEN', 'HIGH': 'HIGH', 'LOW': 'LOW', 'CLOSE': 'CLOSE',
            'VOLUME': 'VOLUME', 'VALUE': 'VALUE', 'NUMTRADES': 'NUMTRADES',
            'WAPRICE': 'CLOSE', 'RATE': 'CLOSE'
        }
        col_standard_map_lower = {k.lower(): v for k, v in col_standard_map.items()}
        cols_to_keep = []
        processed_std_names = set()

        for col in df.columns:
            standard_name = None; col_lower = col.lower()
            if col_lower in col_standard_map_lower:
                standard_name_part = col_standard_map_lower[col_lower]
                if col == price_col_found: standard_name = f"{ticker}_CLOSE" # Найденную цену всегда в _CLOSE
                elif standard_name_part != 'CLOSE': standard_name = f"{ticker}_{standard_name_part}"
            if standard_name:
                 if standard_name not in processed_std_names:
                      rename_map[col] = standard_name; cols_to_keep.append(standard_name); processed_std_names.add(standard_name)
                 elif standard_name == f"{ticker}_CLOSE" and standard_name in processed_std_names: pass

        df.rename(columns=rename_map, inplace=True)
        final_cols_exist = [c for c in cols_to_keep if c in df.columns]
        print(f"Колонки после стандартизации: {final_cols_exist}")

        if not final_cols_exist: return pd.DataFrame()
        else:
            close_col_final = f"{ticker}_CLOSE"
            if close_col_final not in final_cols_exist:
                 value_col_final = f"{ticker}_VALUE"
                 if ticker in [RGBITR_TICKER, RUCBTRNS_TICKER] and value_col_final in final_cols_exist:
                      df[close_col_final] = df[value_col_final]; final_cols_exist.append(close_col_final)
                 else: return pd.DataFrame()
            df = df[final_cols_exist]

    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА при обработке DataFrame для {ticker}: {e}")
        return pd.DataFrame()

    print(f"--- Успешная обработка для {ticker}. Итого строк: {len(df)}. ---")
    return df

# --- Остальной код без изменений ---
# ... (скопируйте сюда весь остальной код вашего скрипта, начиная с `def calculate_zscore...` и до конца) ...

# --- Основной скрипт ---
# ... (без изменений) ...
# 2. Объединение и предварительная обработка
# ... (без изменений) ...
# 3. Расчет метрик
# ... (без изменений) ...
# 4. Фильтрация по дате и сохранение
# ... (без изменений) ...

def calculate_zscore(series, window):
    """Расчет Z-Score для временного ряда."""
    roll = series.rolling(window=window, min_periods=max(1, int(window*0.8)))
    mean = roll.mean()
    std = roll.std(ddof=0)
    zscore = (series - mean) / std.replace(0, np.nan)
    zscore[std == 0] = 0
    return zscore

# --- Основной скрипт ---
all_data = []
with requests.Session() as session:
    # IMOEX
    imoex_df = get_market_data(session, IMOEX_TICKER, START_DATE_REQUEST, END_DATE_OUTPUT) # engine/market определяются внутри
    has_ohlc = False
    if not imoex_df.empty:
        all_data.append(imoex_df)
        # Проверяем наличие стандартно переименованных колонок
        has_ohlc = all(col in imoex_df.columns for col in [f'{IMOEX_TICKER}_HIGH', f'{IMOEX_TICKER}_LOW', f'{IMOEX_TICKER}_CLOSE'])
        if not has_ohlc:
             print(f"-> Итоговый DataFrame для {IMOEX_TICKER} не содержит OHLC колонок.")
    else:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить данные для {IMOEX_TICKER}. Расчеты невозможны.")
        exit()

    # --- ИЗМЕНЕНО: USD/RUB ---
    usd_rub_df_OLD = get_market_data(session, USD_RUB_TICKER_OLD, START_DATE_REQUEST, END_DATE_OUTPUT) # engine/market/board определяются внутри
    if not usd_rub_df_OLD.empty:
        all_data.append(usd_rub_df_OLD)
    else:
        print(f"Предупреждение: Не удалось загрузить данные для {USD_RUB_TICKER}. Метрика Z_USD_RUB_trend21 будет пропущена.")

    usd_rub_df_NEW = get_market_data(session, USD_RUB_TICKER_NEW, START_DATE_REQUEST,
                                 END_DATE_OUTPUT)  # engine/market/board определяются внутри
    if not usd_rub_df_NEW.empty:
        all_data.append(usd_rub_df_NEW)

    # --- ИЗМЕНЕНО: RTSOG (замена Urals) ---
    rtsog_df = get_market_data(session, RTSOG_TICKER, START_DATE_REQUEST, END_DATE_OUTPUT) # engine/market определяются внутри
    if not rtsog_df.empty:
        all_data.append(rtsog_df)
    else:
        print(f"Предупреждение: Не удалось загрузить данные для {RTSOG_TICKER}. Метрики Rel_IMO_RTSOG и Z_RTSOG_trend63 будут пропущены.")

    # RGBITR
    rgbitr_df = get_market_data(session, RGBITR_TICKER, START_DATE_REQUEST, END_DATE_OUTPUT) # engine/market определяются внутри
    if not rgbitr_df.empty:
        all_data.append(rgbitr_df)
    else:
        print(f"Предупреждение: Не удалось загрузить данные для {RGBITR_TICKER}. Метрика Z_RGBITR_trend63 будет пропущена.")

    # RUCBTRNS
    rucbtrns_df = get_market_data(session, RUCBTRNS_TICKER, START_DATE_REQUEST, END_DATE_OUTPUT) # engine/market определяются внутри
    if not rucbtrns_df.empty:
        all_data.append(rucbtrns_df)
    else:
         print(f"Предупреждение: Не удалось загрузить данные для {RUCBTRNS_TICKER}. Метрика Z_RUCBTRNS_trend63 будет пропущена.")

# 2. Объединение и предварительная обработка
if not all_data:
    print("Не удалось загрузить никакие данные. Завершение работы.")
    exit()

df = pd.concat(all_data, axis=1, join='outer')

# Удаляем дубликаты в индексе, оставляя последнюю запись
print(f"Размер df до удаления дубликатов индекса: {df.shape}")
# Убедимся что индекс - это DatetimeIndex перед использованием duplicated
if not isinstance(df.index, pd.DatetimeIndex):
    print("Предупреждение: Индекс не является DatetimeIndex. Попытка преобразования...")
    try:
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        print(f"Не удалось преобразовать индекс в DatetimeIndex: {e}. Пропуск удаления дубликатов.")
    else:
         if not df.index.is_unique:
             df = df[~df.index.duplicated(keep='last')]

elif not df.index.is_unique:
    df = df[~df.index.duplicated(keep='last')]

print(f"Размер df после удаления дубликатов индекса: {df.shape}")


# Переиндексация по торговым дням IMOEX
if not imoex_df.empty:
    if not imoex_df.index.is_unique:
        imoex_df = imoex_df[~imoex_df.index.duplicated(keep='last')]
    actual_trading_days = imoex_df.index.unique()
    # Убедимся что actual_trading_days отсортирован
    actual_trading_days = actual_trading_days.sort_values()

    # Переиндексируем основной DataFrame
    df = df.reindex(actual_trading_days)
    # Обрезаем диапазон df *после* переиндексации и перед ffill
    df = df[(df.index >= pd.to_datetime(START_DATE_REQUEST)) & (df.index <= pd.to_datetime(END_DATE_OUTPUT))]
    df.sort_index(inplace=True)
    print(f"DataFrame переиндексирован и обрезан по {len(df)} торговым дням IMOEX в диапазоне {START_DATE_REQUEST} - {END_DATE_OUTPUT}.")
else:
    print("Не удалось получить индекс торговых дней из IMOEX. Используется исходный объединенный индекс.")
    df.sort_index(inplace=True)
    # Обрезаем диапазон df перед ffill
    df = df[(df.index >= pd.to_datetime(START_DATE_REQUEST)) & (df.index <= pd.to_datetime(END_DATE_OUTPUT))]


# --- ИЗМЕНЕНО: Слияние данных USD/RUB ---
print(f"\nСлияние данных для USD/RUB...")
final_usd_rub_col = 'USDRUB_FINAL_CLOSE' # Новое имя для объединенной колонки
old_usd_col = f"{USD_RUB_TICKER_OLD}_CLOSE"
new_usd_col = f"{USD_RUB_TICKER_NEW}_CLOSE"
switch_date = pd.to_datetime(USD_RUB_SWITCH_DATE)

# Проверяем наличие обеих колонок
has_old = old_usd_col in df.columns
has_new = new_usd_col in df.columns

if has_old and has_new:
    print(f"Найдены обе колонки: {old_usd_col} и {new_usd_col}.")
    # Используем combine_first: приоритет у new_usd_col, NaN в ней заполняются из old_usd_col
    df[final_usd_rub_col] = df[new_usd_col].combine_first(df[old_usd_col])
    # Проверка: Сколько значений взято из нового тикера?
    count_new = df[df.index >= switch_date][final_usd_rub_col].notna().sum()
    print(f"Использовано {count_new} значений из {new_usd_col} (с {switch_date.date()})")
    # Удаляем исходные колонки
    df.drop(columns=[old_usd_col, new_usd_col], inplace=True)
    print(f"Создана объединенная колонка: {final_usd_rub_col}")
elif has_old:
    print(f"Предупреждение: Найден только старый тикер {old_usd_col}. Используем его.")
    df.rename(columns={old_usd_col: final_usd_rub_col}, inplace=True)
elif has_new:
    print(f"Предупреждение: Найден только новый тикер {new_usd_col}. Используем его.")
    df.rename(columns={new_usd_col: final_usd_rub_col}, inplace=True)
else:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Не найдены колонки цен ни для одного из USD/RUB тикеров.")
    # Создаем пустую колонку, чтобы скрипт не упал дальше
    df[final_usd_rub_col] = np.nan

# Проверяем результат слияния
if final_usd_rub_col in df:
     print(f"Проверка объединенной колонки '{final_usd_rub_col}':")
     print(df[[final_usd_rub_col]].tail(10)) # Посмотрим хвост
     if df[final_usd_rub_col].isnull().any():
           print(f"Предупреждение: В колонке {final_usd_rub_col} есть NaN после слияния.")



# Заполняем пропуски (выходные и т.д.)
df.ffill(inplace=True)
df.bfill(inplace=True) # Заполняем NaN в начале

# Обновленные имена колонок цен
imoex_close_col = f'{IMOEX_TICKER}_CLOSE'
# --- ИЗМЕНЕНО ---
rtsog_close_col = f'{RTSOG_TICKER}_CLOSE'
usd_rub_close_col = final_usd_rub_col
# --- ---
rgbitr_close_col = f'{RGBITR_TICKER}_CLOSE'
rucbtrns_close_col = f'{RUCBTRNS_TICKER}_CLOSE'

# 3. Расчет метрик
metrics = pd.DataFrame(index=df.index)

# --- Метрика 1: Z_IMO_return --- (без изменений)
if imoex_close_col in df.columns:
    df['IMO_log_return'] = np.log(df[imoex_close_col] / df[imoex_close_col].shift(1))
    metrics['Z_IMO_return'] = calculate_zscore(df['IMO_log_return'], WINDOW_21)
else:
    print(f"Пропуск метрики 1: Отсутствует колонка {imoex_close_col}")

# --- Метрика 2: Z_IMO_vol_real21 --- (без изменений)
if 'IMO_log_return' in df.columns:
    df['IMO_real_vol_21'] = df['IMO_log_return'].rolling(window=WINDOW_21, min_periods=max(1, int(WINDOW_21*0.8))).std(ddof=0) * np.sqrt(WINDOW_21)
    metrics['Z_IMO_vol_real21'] = calculate_zscore(df['IMO_real_vol_21'], WINDOW_21)
else:
     print(f"Пропуск метрики 2: Отсутствует 'IMO_log_return'")

# --- Метрика 3: Rel_IMO_RTSOG --- (ИЗМЕНЕНО)
print(f"Расчет метрики 3: Rel_IMO_RTSOG (замена Urals)...")
if 'IMO_log_return' in df.columns and rtsog_close_col in df.columns:
    # Лог-доходность RTSOG
    df['RTSOG_log_return'] = np.log(df[rtsog_close_col] / df[rtsog_close_col].shift(1))
    # Относительная доходность (логарифмическая)
    # Используем разность лог-доходностей, что эквивалентно логарифму отношения цен
    metrics['Rel_IMO_RTSOG'] = df['IMO_log_return'] - df['RTSOG_log_return']
    # Если нужна простая доходность:
    # df['IMO_simple_return'] = df[imoex_close_col].pct_change()
    # df['RTSOG_simple_return'] = df[rtsog_close_col].pct_change()
    # metrics['Rel_IMO_RTSOG'] = df['IMO_simple_return'] / df['RTSOG_simple_return'].replace(0, np.nan) # Опасно делением на 0
else:
    print(f"Пропуск метрики 3: Отсутствуют 'IMO_log_return' или '{rtsog_close_col}'")
    metrics['Rel_IMO_RTSOG'] = np.nan

# --- Метрика 4: Z_USD_RUB_trend21 --- (ИЗМЕНЕНО название колонки)
if usd_rub_close_col in df.columns:
    df['USD_RUB_SMA21'] = df[usd_rub_close_col].rolling(window=WINDOW_21, min_periods=max(1, int(WINDOW_21*0.8))).mean()
    metrics['Z_USD_RUB_trend21'] = calculate_zscore(df['USD_RUB_SMA21'], WINDOW_21)
else:
    print(f"Пропуск метрики 4: Отсутствует колонка {usd_rub_close_col}")
    metrics['Z_USD_RUB_trend21'] = np.nan

# --- Метрика 5: Z_ADX_IMOEX --- (без изменений в логике, но использует флаг has_ohlc)
imoex_high_col = f'{IMOEX_TICKER}_HIGH'
imoex_low_col = f'{IMOEX_TICKER}_LOW'
if has_ohlc and all(col in df.columns for col in [imoex_high_col, imoex_low_col, imoex_close_col]):
    try:
        df_temp_adx = df[[imoex_high_col, imoex_low_col, imoex_close_col]].copy()
        df_temp_adx.columns = ['high', 'low', 'close']
        df_temp_adx.dropna(inplace=True)
        if not df_temp_adx.empty:
            adx_result = ta.adx(df_temp_adx['high'], df_temp_adx['low'], df_temp_adx['close'], length=ADX_WINDOW)
            df['IMO_ADX14'] = adx_result[f'ADX_{ADX_WINDOW}']
            metrics['Z_ADX_IMOEX'] = calculate_zscore(df['IMO_ADX14'], ADX_WINDOW)
        else:
            print("Данные для расчета ADX пусты после удаления NaN.")
            metrics['Z_ADX_IMOEX'] = np.nan
    except Exception as e:
        print(f"Ошибка при расчете ADX: {e}")
        metrics['Z_ADX_IMOEX'] = np.nan
else:
    print(f"Пропуск метрики 5 (Z_ADX_IMOEX): Отсутствуют данные HIGH/LOW/CLOSE для {IMOEX_TICKER}.")
    metrics['Z_ADX_IMOEX'] = np.nan

# --- Метрика 6: Z_IMO_trend21 --- (без изменений)
if imoex_close_col in df.columns:
    df['IMO_SMA21'] = df[imoex_close_col].rolling(window=WINDOW_21, min_periods=max(1, int(WINDOW_21*0.8))).mean()
    metrics['Z_IMO_trend21'] = calculate_zscore(df['IMO_SMA21'], WINDOW_21)
else:
     print(f"Пропуск метрики 6: Отсутствует колонка {imoex_close_col}")
     metrics['Z_IMO_trend21'] = np.nan

# --- Метрика 7: Z_RTSOG_trend63 --- (ИЗМЕНЕНО)
print(f"Расчет метрики 7: Z_RTSOG_trend63 (замена Urals)...")
if rtsog_close_col in df.columns:
    df['RTSOG_SMA63'] = df[rtsog_close_col].rolling(window=WINDOW_63, min_periods=max(1, int(WINDOW_63*0.8))).mean()
    metrics['Z_RTSOG_trend63'] = calculate_zscore(df['RTSOG_SMA63'], WINDOW_63)
else:
     print(f"Пропуск метрики 7: Отсутствует колонка {rtsog_close_col}")
     metrics['Z_RTSOG_trend63'] = np.nan

# --- Метрика 8: Z_IMO_trend63 --- (без изменений)
if imoex_close_col in df.columns:
    df['IMO_SMA63'] = df[imoex_close_col].rolling(window=WINDOW_63, min_periods=max(1, int(WINDOW_63*0.8))).mean()
    metrics['Z_IMO_trend63'] = calculate_zscore(df['IMO_SMA63'], WINDOW_63)
else:
     print(f"Пропуск метрики 8: Отсутствует колонка {imoex_close_col}")
     metrics['Z_IMO_trend63'] = np.nan

# --- Метрика 9: Z_RGBITR_trend63 --- (без изменений)
if rgbitr_close_col in df.columns:
    df['RGBITR_SMA63'] = df[rgbitr_close_col].rolling(window=WINDOW_63, min_periods=max(1, int(WINDOW_63*0.8))).mean()
    metrics['Z_RGBITR_trend63'] = calculate_zscore(df['RGBITR_SMA63'], WINDOW_63)
else:
     print(f"Пропуск метрики 9: Отсутствует колонка {rgbitr_close_col}")
     metrics['Z_RGBITR_trend63'] = np.nan

# --- Метрика 10: Z_RUCBTRNS_trend63 --- (без изменений)
if rucbtrns_close_col in df.columns:
    df['RUCBTRNS_SMA63'] = df[rucbtrns_close_col].rolling(window=WINDOW_63, min_periods=max(1, int(WINDOW_63*0.8))).mean()
    metrics['Z_RUCBTRNS_trend63'] = calculate_zscore(df['RUCBTRNS_SMA63'], WINDOW_63)
else:
     print(f"Пропуск метрики 10: Отсутствует колонка {rucbtrns_close_col}")
     metrics['Z_RUCBTRNS_trend63'] = np.nan

print(f"Расчет метрики 11: IMO_fwd_log_ret_3d...")
if imoex_close_col in df.columns:
    # Цена закрытия 2 дня вперед
    close_fwd_2 = df[imoex_close_col].shift(-2)
    # Цена закрытия 1 день назад
    close_bwd_1 = df[imoex_close_col].shift(1)

    # Рассчитываем лог-доходность за 3-дневный период (Close[n+2] / Close[n-1])
    # Делим на 3 для получения среднедневной доходности
    # Добавляем small value для избежания log(0) или деления на 0, если цены нулевые
    small_value = 1e-10
    log_ret_3d_total = np.log( (close_fwd_2 + small_value) / (close_bwd_1 + small_value) )*1000
    metrics['IMO_fwd_log_ret_3d'] = log_ret_3d_total / 3
else:
    print(f"Пропуск метрики 11: нет {imoex_close_col}")
    metrics['IMO_fwd_log_ret_3d'] = np.nan

# 4. Фильтрация по дате и сохранение
final_df = metrics.loc[START_DATE_OUTPUT:END_DATE_OUTPUT].copy()

# --- ИЗМЕНЕНО: Обновленный список колонок ---
required_cols = [
    'Z_IMO_return', 'Z_IMO_vol_real21', 'Rel_IMO_RTSOG', 'Z_USD_RUB_trend21',
    'Z_ADX_IMOEX', 'Z_IMO_trend21', 'Z_RTSOG_trend63', 'Z_IMO_trend63',
    'Z_RGBITR_trend63', 'Z_RUCBTRNS_trend63', 'IMO_fwd_log_ret_3d'
]
# Добавляем пропущенные колонки
for col in required_cols:
    if col not in final_df.columns:
        final_df[col] = np.nan
final_df = final_df[required_cols] # Устанавливаем порядок

# Округление
final_df = final_df.round(6)

# Сохранение в CSV
try:
    final_df.to_csv(OUTPUT_CSV_FILE, index=True, date_format='%Y-%m-%d')
    print(f"\nРасчеты завершены.")
    print(f"Проверьте наличие данных OHLC для IMOEX: {'Да' if has_ohlc else 'Нет (ADX может быть не рассчитан или NaN)'}")
    # --- ИЗМЕНЕНО ---
    print(f"Метрики 3 и 7 теперь используют RTSOG вместо Urals.")
    print(f"Метрика 4 использует USDRUBTOMOTC.")
    # --- ---
    print(f"Результаты сохранены в файл: {OUTPUT_CSV_FILE}")
    print("\nПервые 5 строк итоговой таблицы:")
    print(final_df.head())
    print("\nПоследние 5 строк итоговой таблицы:")
    print(final_df.tail())
    print("\nИнформация о таблице:")
    final_df.info()
except Exception as e:
     print(f"Ошибка при сохранении CSV файла: {e}")