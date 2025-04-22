import requests
# Убрали импорт Timeout, т.к. убираем логику таймаута
import apimoex
from apimoex import ISSClient
import pandas as pd
from datetime import datetime
import io

# --- Параметры для диагностики ---
TICKER = 'USDRUBTOMOTC'
START_DATE = '2024-08-05'
END_DATE = '2024-08-15'
BOARD = 'COTF'
# REQUEST_TIMEOUT = 30 # Убрали таймаут для простоты

HISTORY_COLUMNS = (
    'TRADEDATE', 'CLOSE', 'RATE', 'WAPRICE', 'OPEN', 'HIGH', 'LOW',
    'VOLUME', 'VALUE', 'NUMTRADES', 'BOARDID', 'SECID', 'SHORTNAME'
)
CANDLE_COLUMNS = (
    'begin', 'open', 'high', 'low', 'close', 'value', 'volume'
)
INDEX_COLUMNS = (
    'tradedate', 'secid', 'value', 'open', 'high', 'low', 'close', 'rate'
)

# --- Функция диагностики (без логики таймаута) ---
def diagnose_fetch_attempt(description, fetch_func, *args, **kwargs):
    print(f"\n{'='*10} ДИАГНОСТИКА: {description} {'='*10}")
    print(f"Параметры вызова (кроме session, security, dates): {kwargs}")
    data = None
    df = pd.DataFrame()

    try:
        print("Вызов функции...")
        # Обработка ISSClient отдельно
        if hasattr(fetch_func, '__self__') and isinstance(fetch_func.__self__, ISSClient) and fetch_func.__name__ == 'get_all':
             response_data = fetch_func()
             print("Вызов ISSClient.get_all завершен.")
             data_source_key = None
             actual_data = None
             if isinstance(response_data, dict):
                 keys_to_check = ['history', 'analytics', 'data', 'securities']
                 for key in keys_to_check:
                     if key in response_data and isinstance(response_data[key], list) and response_data[key]:
                         data_source_key = key
                         actual_data = response_data[key]
                         print(f"Найден ключ '{data_source_key}' с данными.")
                         break
                 if not actual_data:
                      for key, value in response_data.items():
                           if isinstance(value, list) and value and isinstance(value[0], dict):
                                data_source_key = key
                                actual_data = value
                                print(f"Найден другой ключ '{data_source_key}' с данными.")
                                break
             data = actual_data
             if data: print(f"Данные извлечены из ключа '{data_source_key}'.")
             else: print("Не удалось извлечь данные из ответа ISSClient:", response_data)

        else: # Обычный вызов функций apimoex
            data = fetch_func(*args, **kwargs)
            print("Вызов функции завершен.")

        if not data:
            print(">>> Результат: Данные не получены.")
            return df

        print(f">>> Результат: Получено {len(data)} записей.")

        # --- Анализ DataFrame (без изменений) ---
        try:
            df = pd.DataFrame(data)
            if df.empty:
                 print(">>> Результат: DataFrame пуст.")
                 return df

            print("\n--- Анализ DataFrame ---")
            print(f"Колонки: {df.columns.tolist()}")
            buffer = io.StringIO(); df.info(buf=buffer); info_str = buffer.getvalue()
            print(f"Типы данных и информация:\n{info_str}")
            print("\nПервые 5 строк:"); print(df.head())

            date_col = None; # ... (остальной анализ как в предыдущей версии) ...
            for col in ['TRADEDATE', 'tradedate', 'Date', 'date', 'begin']: # ...
                 if col in df.columns: date_col = col; break # ...
            if date_col: # ...
                 print(f"\nАнализ даты (колонка '{date_col}'):") # ...
                 try: # ...
                     df[date_col] = pd.to_datetime(df[date_col]) # ...
                     num_valid_dates = df[date_col].notna().sum() # ...
                     dates_in_range = df[(df[date_col] >= pd.to_datetime(START_DATE)) & (df[date_col] <= pd.to_datetime(END_DATE))] # ...
                     print(f" - Корректных дат: {num_valid_dates}") # ...
                     print(f" - Дат в запрошенном диапазоне ({START_DATE} - {END_DATE}): {len(dates_in_range)}") # ...
                 except Exception as date_e: print(f" - Ошибка анализа даты: {date_e}") # ...
            else: print("\nКолонка даты не найдена.") # ...

            print("\nАнализ колонок цен:") # ...
            price_cols_to_check = ['RATE', 'WAPRICE', 'CLOSE', 'rate', 'waprice', 'close', 'VALUE', 'value', 'PRICE', 'price'] # ...
            found_valid_price = False # ...
            for p_col in price_cols_to_check: # ...
                if p_col in df.columns: # ...
                    try: # ...
                        price_series = pd.to_numeric(df[p_col], errors='coerce') # ...
                        num_nan = price_series.isna().sum(); num_zero = (price_series == 0).sum(); num_valid = (~price_series.isna() & (price_series != 0)).sum() # ...
                        print(f" - Колонка '{p_col}': NaN={num_nan}, Нули={num_zero}, Валидные={num_valid}") # ...
                        if num_valid > 0: # ...
                            found_valid_price = True # ...
                            print(f"     Пример валидных: {price_series[price_series.notna() & (price_series != 0)].head(3).tolist()}") # ...
                    except Exception as price_e: print(f"   - Ошибка анализа '{p_col}': {price_e}") # ...

            if found_valid_price: print(">>> ОБНАРУЖЕНА КОЛОНКА С ВАЛИДНЫМИ ЦЕНАМИ!") # ...
            else: print(">>> Валидные цены не обнаружены.") # ...

        except Exception as df_e:
            print(f"\n>>> Ошибка при создании/анализе DataFrame: {df_e}")
            print("Сырые данные (первые 3 записи):", data[:3])

    except Exception as e: # Ловим общие ошибки
        print(f">>> КРИТИЧЕСКАЯ ОШИБКА при вызове функции: {e}")

    print(f"{'='*10} Конец диагностики: {description} {'='*10}")
    return df


# --- Запуск диагностики для USDRUBTOMOTC ---
print(f"*** Диагностика параметров для {TICKER} ***")
print(f"Период: {START_DATE} - {END_DATE}\n")

# Используем стандартную сессию без модификаций
with requests.Session() as session:

    # --- Тестируем разные комбинации engine/market/function ---

    # Комбинация 1: currency / otc
    engine1 = 'currency'
    market1 = 'otc'
    board1 = BOARD # COTF
    desc1a = f"get_market_history (engine='{engine1}', market='{market1}')"
    diagnose_fetch_attempt(desc1a, apimoex.get_market_history, session=session, security=TICKER, start=START_DATE, end=END_DATE, engine=engine1, market=market1, columns=HISTORY_COLUMNS)
    desc1b = f"get_board_history (engine='{engine1}', market='{market1}', board='{board1}')"
    diagnose_fetch_attempt(desc1b, apimoex.get_board_history, session=session, security=TICKER, start=START_DATE, end=END_DATE, engine=engine1, market=market1, board=board1, columns=HISTORY_COLUMNS)
    desc1c = f"get_market_candles (engine='{engine1}', market='{market1}', interval=24)"
    diagnose_fetch_attempt(desc1c, apimoex.get_market_candles, session=session, security=TICKER, start=START_DATE, end=END_DATE, engine=engine1, market=market1, interval=24, columns=CANDLE_COLUMNS)
    desc1d = f"get_board_candles (engine='{engine1}', market='{market1}', board='{board1}', interval=24)"
    diagnose_fetch_attempt(desc1d, apimoex.get_board_candles, session=session, security=TICKER, start=START_DATE, end=END_DATE, engine=engine1, market=market1, board=board1, interval=24, columns=CANDLE_COLUMNS)

    # Комбинация 2: otc / index
    engine2 = 'otc'
    market2 = 'index'
    board2 = BOARD # COTF
    desc2a = f"get_market_history (engine='{engine2}', market='{market2}')"
    diagnose_fetch_attempt(desc2a, apimoex.get_market_history, session=session, security=TICKER, start=START_DATE, end=END_DATE, engine=engine2, market=market2, columns=INDEX_COLUMNS)
    desc2b = f"get_board_history (engine='{engine2}', market='{market2}', board='{board2}')"
    diagnose_fetch_attempt(desc2b, apimoex.get_board_history, session=session, security=TICKER, start=START_DATE, end=END_DATE, engine=engine2, market=market2, board=board2, columns=INDEX_COLUMNS)
    print(f"\n{'='*10} ПРОПУСК ТЕСТОВ: Свечи для engine='otc', market='index' (вызывали зависание) {'='*10}")


    # Комбинация 3: stock / index
    engine3 = 'stock'
    market3 = 'index'
    desc3a = f"get_market_history (engine='{engine3}', market='{market3}')"
    diagnose_fetch_attempt(desc3a, apimoex.get_market_history, session=session, security=TICKER, start=START_DATE, end=END_DATE, engine=engine3, market=market3, columns=INDEX_COLUMNS)


    # --- Тестируем ISSClient с разными URL ---

    # URL 1: Статистика индикативных курсов
    desc_iss1 = "ISSClient - URL статистики indicativerates"
    url_iss1 = f"https://iss.moex.com/iss/statistics/engines/currency/markets/indicativerates/securities/{TICKER}.json"
    # !!! Убрали iss.meta из params !!!
    params_iss1 = {"from": START_DATE, "till": END_DATE}
    print(f"\nПараметры для {desc_iss1}:\nURL: {url_iss1}\nParams: {params_iss1}")
    client1 = ISSClient(session, url_iss1, params_iss1)
    diagnose_fetch_attempt(desc_iss1, client1.get_all)

    # URL 2: История otc/currency/COTF
    desc_iss2 = "ISSClient - URL истории otc/currency/COTF"
    url_iss2 = f"https://iss.moex.com/iss/history/engines/otc/markets/currency/boards/{BOARD}/securities/{TICKER}.json"
    # !!! Убрали iss.meta, iss.only, history.columns из params !!!
    params_iss2 = {"from": START_DATE, "till": END_DATE}
    print(f"\nПараметры для {desc_iss2}:\nURL: {url_iss2}\nParams: {params_iss2}")
    client2 = ISSClient(session, url_iss2, params_iss2)
    diagnose_fetch_attempt(desc_iss2, client2.get_all)

    # URL 3: История otc/index/COTF
    desc_iss3 = "ISSClient - URL истории otc/index/COTF"
    url_iss3 = f"https://iss.moex.com/iss/history/engines/otc/markets/index/boards/{BOARD}/securities/{TICKER}.json"
    # !!! Убрали iss.meta, iss.only, history.columns из params !!!
    params_iss3 = {"from": START_DATE, "till": END_DATE}
    print(f"\nПараметры для {desc_iss3}:\nURL: {url_iss3}\nParams: {params_iss3}")
    client3 = ISSClient(session, url_iss3, params_iss3)
    diagnose_fetch_attempt(desc_iss3, client3.get_all)


print("\n*** Диагностика завершена ***")