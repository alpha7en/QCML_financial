print(f"--- Начало загрузки для {ticker} ({start_date} - {end_date}) ---")
data = []

func_to_use = None  # Будет определен в попытках
data_source_key = 'history'  # Ключ в ответе ISSClient, где лежат данные (предположение)

# Попытка 1: Стандартная логика определения параметров
current_engine = engine
current_market = market
current_board = board
current_func = apimoex.get_market_history
current_cols = list(columns)

if ticker in ['IMOEX', 'RGBITR', 'RUCBTRNS', RTSOG_TICKER]:
    current_engine = 'stock'
    current_market = 'index'
    if ticker == 'IMOEX':
        print("Тикер IMOEX: попытка OHLCV...")
        current_cols = ['TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'VALUE']


elif ticker == USD_RUB_TICKER:

    current_engine = 'currency'

    current_market = 'selt'

    current_board = 'CETS'

    current_func = apimoex.get_board_history  # Используем board_history

    print(
        f"Параметры для {ticker}: func=get_board_history, engine={current_engine}, market={current_market}, board={current_board} (запрос WAPRICE/CLOSE/OHLC)")

    # Запрашиваем WAPRICE и CLOSE как основные цены, OHLC для возможного анализа

    current_cols = ('TRADEDATE', 'WAPRICE', 'CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'VALUE', 'NUMTRADES')


elif engine == 'currency':  # Другие валюты
    current_market = 'selt'

print(
    f"Параметры попытки 1: func={current_func.__name__}, engine={current_engine}, market={current_market}, board={current_board}, cols={current_cols}")
try:
    if current_func == apimoex.get_board_history:
        if current_board is None: raise ValueError("Board не указан для get_board_history")
        data = current_func(session, security=ticker, start=start_date, end=end_date,
                            engine=current_engine, market=current_market, board=current_board,
                            columns=tuple(current_cols))
    else:
        data = current_func(session, security=ticker, start=start_date, end=end_date,
                            engine=current_engine, market=current_market, columns=tuple(current_cols))
    if data:
        func_to_use = current_func  # Сохраняем успешную функцию
    else:
        print("Попытка 1: Данные не получены.")
except Exception as e:
    print(f"Попытка 1 ОШИБКА: {e}")
    data = []

