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
BOARD = 'TQBR' # Основной режим торгов T+2
IMOEX_TICKER = 'IMOEX'
NEWS_DATA_PATH = 'news_features.csv' # Укажите путь к вашему файлу
OUTPUT_DATASET_PATH = 'moex_qc_dataset.parquet' # Выходной файл датасета

# Признаки из статьи, которые будем нормализовать (включая Beta и Size для очистки)
# Short Utilization исключен из-за проблем с данными
FEATURES_TO_NORMALIZE = [
    'Accruals', 'EBITDA_to_TEV', 'Momentum', 'Operating_Efficiency',
    'Profit_Margin', 'Size', 'Value', 'Beta' # Beta и Size для ортогонализации, но тоже нормализуются
]

# Список всех признаков, включая GICS dummies и новости, для регрессии очистки
# Этот список будет формироваться динамически после загрузки всех данных
ALL_REGRESSION_FACTORS = FEATURES_TO_NORMALIZE[:] # Копируем нормализуемые
# Добавятся GICS dummies (будут иметь префикс 'GICS_') и новости (ваши названия колонок)


# --- Инициализация ---
session = requests.Session()

print(f"Старт обработки данных для периода: {START_DATE} - {END_DATE}")

# --- Шаг 1: Определение Вселенной Акций MOEX ---
print("\nШаг 1: Определение вселенной акций MOEX...")

try:
    # Получаем список всех акций в режиме TQBR
    # Документация apimoex.get_board_securities: table='securities' (справочник), board='TQBR'
    # Добавляем 'SECTYPE' чтобы потом отфильтровать не-акции, если нужно
    board_securities_data = apimoex.get_board_securities(
        session=session,
        table="securities",
        board=BOARD,
        columns=('SECID', 'SHORTNAME', 'SECTYPE', 'LOTSIZE', 'ISSUESIZE') # ISSUESIZE может помочь для MarketCap, но его историчность под вопросом
    )
    board_securities_df = pd.DataFrame(board_securities_data)

    # Фильтруем только акции ('common stock')
    # SEC TYPE 'stocks'
    # https://iss.moex.com/iss/reference/32 -> securities -> type
    stock_tickers_df = board_securities_df[board_securities_df['SECTYPE'] == '1'].copy()

    # Можно добавить дополнительные фильтры:
    # 1. Исключить фонды, ETF, и т.д. (уже сделано по 'SECTYPE')
    # 2. Фильтр по минимальной капитализации (если есть ISSUESIZE) или объему торгов (нужны исторические данные)
    # 3. Фильтр по наличию истории торгов за наш период (сделаем позже при загрузке)
    # 4. Фильтр по наличию финансовой отчетности (главная проблема, сделаем позже, отсеивая тикеры без данных)

    initial_tickers = stock_tickers_df['SECID'].tolist()
    print(f"Найдено {len(initial_tickers)} потенциальных тикеров в режиме {BOARD}.")

except Exception as e:
    print(f"Ошибка при получении списка тикеров: {e}")
    initial_tickers = [] # Если ошибка, список тикеров будет пустым

# --- Шаг 2: Сбор Исходных Данных ---
print("\nШаг 2: Сбор исходных рыночных данных...")

all_market_data = []
valid_tickers = [] # Список тикеров, для которых удалось загрузить данные

# Загрузка данных по IMOEX
imoex_data = None
try:
    print(f"Загрузка данных для индекса {IMOEX_TICKER}...")
    # get_board_history для индекса, режим 'MRNW' обычно
    # https://iss.moex.com/iss/reference/65
    # engine='stock', market='index'
    imoex_hist = apimoex.get_board_history(
        session=session,
        security=IMOEX_TICKER,
        start=START_DATE,
        end=END_DATE,
        board='MRNW', # Режим для индексов
        market='index',
        columns=('TRADEDATE', 'CLOSE', 'VOLUME', 'VALUE') # VOLUME и VALUE для индекса не нужны, но запросим основные
    )
    imoex_data = pd.DataFrame(imoex_hist)
    imoex_data['TRADEDATE'] = pd.to_datetime(imoex_data['TRADEDATE'])
    imoex_data.set_index('TRADEDATE', inplace=True)
    imoex_data = imoex_data[['CLOSE']].rename(columns={'CLOSE': f'{IMOEX_TICKER}_CLOSE'})
    print(f"Загружены данные для {IMOEX_TICKER} ({len(imoex_data)} дней).")

except Exception as e:
    print(f"Ошибка при загрузке данных для {IMOEX_TICKER}: {e}")
    imoex_data = None # Если ошибка, данные IMOEX будут None

# Загрузка данных по акциям
# Пауза для соблюдения лимитов API
PAUSE_DURATION = 0.1 # секунды

for i, ticker in enumerate(initial_tickers):
    print(f"Загрузка данных для {ticker} ({i+1}/{len(initial_tickers)})...")
    try:
        # get_board_history для акций, режим TQBR
        # https://iss.moex.com/iss/reference/65
        # engine='stock', market='shares', board='TQBR'
        stock_hist = apimoex.get_board_history(
            session=session,
            security=ticker,
            start=START_DATE,
            end=END_DATE,
            board=BOARD,
            columns=('TRADEDATE', 'CLOSE', 'VOLUME', 'VALUE')
        )
        stock_df = pd.DataFrame(stock_hist)

        if not stock_df.empty:
            stock_df['TRADEDATE'] = pd.to_datetime(stock_df['TRADEDATE'])
            stock_df['SECID'] = ticker
            # Добавляем MarketCap - PLACEHOLDER.
            # Нужно получить ISSUESIZE для SECID на каждую дату и умножить на CLOSE.
            # ISSUESIZE меняется со временем (допэмиссии и т.д.), его историю сложно получить общедоступно.
            # Для примера, добавим пустую колонку MarketCap, которую нужно заполнить из другого источника.
            # Либо, если ISSUESIZE не меняется, можно попробовать получить его статически.
            # Для целей демонстрации, пока оставим ее пустой или заполним NaN.
            # https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/SBER.json?iss.meta=off&iss.only=history&history.columns=TRADEDATE,ISSUESIZE
            # ISSUE SIZE доступен в истории, но это отдельный запрос для каждой бумаги!
            # Это сильно усложнит загрузку данных.
            # Временно добавим пустую колонку MarketCap:
            stock_df['MarketCap'] = np.nan # Эту колонку нужно заполнить реальными данными

            all_market_data.append(stock_df)
            valid_tickers.append(ticker)
            print(f"  Успешно загружено {len(stock_df)} строк.")
        else:
            print(f"  Нет данных для {ticker} в указанный период или режиме {BOARD}.")

    except Exception as e:
        print(f"  Ошибка при загрузке данных для {ticker}: {e}")

    time.sleep(PAUSE_DURATION) # Пауза между запросами

if not all_market_data:
    print("\nОшибка: Не удалось загрузить данные ни для одной акции. Проверьте тикеры, даты и доступность API.")
    exit() # Прерываем выполнение, если нет данных

# Объединяем все данные по акциям
market_data_df = pd.concat(all_market_data, ignore_index=True)
market_data_df.set_index(['TRADEDATE', 'SECID'], inplace=True)
market_data_df.sort_index(inplace=True)

print(f"\nЗагружены данные для {len(valid_tickers)} акций. Общее количество строк: {len(market_data_df)}.")
print("Пример загруженных данных:")
print(market_data_df.head())
print(market_data_df.tail())

# Объединяем с данными IMOEX
if imoex_data is not None and not imoex_data.empty:
    # Присоединяем IMOEX CLOSE к каждой строке по дате
    market_data_df = market_data_df.join(imoex_data, on='TRADEDATE')
    print(f"\nРазмер DataFrame после присоединения IMOEX: {market_data_df.shape}")
else:
    print("\nДанные IMOEX не загружены или пусты. Beta не может быть рассчитана.")
    # Если IMOEX нет, нужно будет исключить Beta из FEATURES_TO_NORMALIZE и ALL_REGRESSION_FACTORS

# --- PLACEHOLDER: Загрузка Финансовой Отчетности ---
print("\nPLACEHOLDER: Загрузка и обработка финансовой отчетности...")
# Этот шаг требует данных из внешнего источника.
# Предположим, что у вас есть DataFrame `financial_data_df`
# с колонками: 'TRADEDATE' (дата окончания квартала или дата публикации), 'SECID', 'TOTALASSETS', 'WORKINGCAPITAL', ...
# Нужно реализовать логику, которая для каждой ежедневной даты в `market_data_df`
# находит последние доступные данные из `financial_data_df`.

# Пример создания пустого DataFrame для фин. данных
financial_data_df = pd.DataFrame(columns=[
    'TRADEDATE', 'SECID', 'TOTALASSETS', 'WORKINGCAPITAL', 'TOTALLIABILITIES',
    'LONGTERMINVESTMENTS', 'LONGTERMDEBT', 'REVENUES', 'NETINCOME', 'EBITDA',
    'CASHEQUIVALENTS' # Для TEV
])
# financial_data_df['TRADEDATE'] = pd.to_datetime(financial_data_df['TRADEDATE'])
# financial_data_df.set_index(['TRADEDATE', 'SECID'], inplace=True)
# Пример: Загрузка из файла CSV (если у вас есть такой файл)
# try:
#     financial_data_df = pd.read_csv('your_financial_data.csv')
#     financial_data_df['TRADEDATE'] = pd.to_datetime(financial_data_df['TRADEDATE'])
#     financial_data_df.set_index(['TRADEDATE', 'SECID'], inplace=True)
#     print(f"Загружены финансовые данные. Строк: {len(financial_data_df)}")
# except FileNotFoundError:
#     print("Файл 'your_financial_data.csv' не найден. Финансовые признаки будут NaN.")
# except Exception as e:
#      print(f"Ошибка при загрузке финансового файла: {e}. Финансовые признаки будут NaN.")

# Логика для присоединения последних доступных фин. данных к ежедневным рыночным данным:
# Это сложный merge. Нужно для каждой строки (date, ticker) в market_data_df
# найти ближайшую по дате назад запись в financial_data_df для того же тикера.
# Можно использовать `pd.merge_asof`. Но financial_data_df должна быть отсортирована по дате.
# merged_df = pd.merge_asof(
#     market_data_df.reset_index().sort_values('TRADEDATE'), # Отсортировать по дате
#     financial_data_df.reset_index().sort_values('TRADEDATE'), # Отсортировать по дате
#     on='TRADEDATE',
#     by='SECID',
#     allowexact=True, # Разрешить совпадение дат (если день публикации совпадает с торговым днем)
#     direction='backward' # Искать назад от торговой даты
# )
# merged_df.set_index(['TRADEDATE', 'SECID'], inplace=True)
# market_data_df = merged_df # Обновляем основной DataFrame

print("ПРИМЕЧАНИЕ: Финансовые данные не загружены. Признаки, зависящие от них, будут NaN.")
# Продолжим, предполагая, что financial_data_df будет интегрирован позже или данные будут NaN.
# Если финансовые данные критически важны и недоступны, дальнейшие шаги по расчету признаков
# (Accruals, EBITDA_to_TEV, Operating_Efficiency, Profit_Margin, Value) приведут к NaN.

# --- PLACEHOLDER: Загрузка Секторальной Классификации (GICS аналог) ---
print("\nPLACEHOLDER: Загрузка секторальной классификации...")
# Нужно получить сектор для каждого SECID. apimoex может предоставлять эту инфрмацию
# например, через get_board_securities или find_security_description, но не всегда в чистом виде GICS
# и не факт, что исторически.
# Для примера, добавим пустую колонку 'Sector'
market_data_df['Sector'] = 'Unknown' # Эту колонку нужно заполнить реальными данными
# Пример: Загрузка из файла CSV (если у вас есть такой файл)
# try:
#     sector_map_df = pd.read_csv('your_sector_mapping.csv') # columns: 'SECID', 'Sector'
#     sector_map_df.set_index('SECID', inplace=True)
#     market_data_df = market_data_df.join(sector_map_df, on='SECID', rsuffix='_from_map')
#     market_data_df['Sector'] = market_data_df['Sector_from_map'].fillna(market_data_df['Sector'])
#     market_data_df.drop(columns='Sector_from_map', inplace=True)
#     print("Загружена и присоединена секторальная классификация.")
# except FileNotFoundError:
#     print("Файл 'your_sector_mapping.csv' не найден. Сектора будут 'Unknown'.")
# except Exception as e:
#      print(f"Ошибка при загрузке секторального файла: {e}. Сектора будут 'Unknown'.")

print("ПРИМЕЧАНИЕ: Секторальная классификация не загружена. Будет использоваться 'Unknown'.")

# --- Загрузка Глобальных Новостных Признаков ---
print(f"\nЗагрузка глобальных новостных признаков из {NEWS_DATA_PATH}...")
try:
    news_df = pd.read_csv(NEWS_DATA_PATH)
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df.rename(columns={'date': 'TRADEDATE'}, inplace=True)
    news_df.set_index('TRADEDATE', inplace=True)

    # Сохраним названия новостных колонок для использования в регрессии
    NEWS_FEATURES = news_df.columns.tolist()
    print(f"Загружены новостные признаки: {NEWS_FEATURES}")
    print(f"Пример новостных данных:\n{news_df.head()}")

    # Объединяем новостные данные с основными по дате
    # Используем `left_join` чтобы сохранить все торговые дни из market_data_df
    market_data_df = market_data_df.join(news_df, on='TRADEDATE')
    print(f"Размер DataFrame после присоединения новостных признаков: {market_data_df.shape}")

except FileNotFoundError:
    print(f"Ошибка: Файл с новостными признаками '{NEWS_DATA_PATH}' не найден.")
    NEWS_FEATURES = [] # Список новостных признаков пуст
except Exception as e:
    print(f"Ошибка при загрузке или обработке новостных признаков: {e}")
    NEWS_FEATURES = [] # Список новостных признаков пуст

if not NEWS_FEATURES:
    print("ПРИМЕЧАНИЕ: Новостные признаки не будут включены в регрессию очистки.")
else:
     ALL_REGRESSION_FACTORS.extend(NEWS_FEATURES) # Добавляем новостные признаки в список регрессоров

# Добавляем GICS dummies в список регрессоров (будут сформированы позже)
# ALL_REGRESSION_FACTORS будет окончательно сформирован после создания GICS dummies

# --- Промежуточный результат Шага 2 ---
print("\nПромежуточный результат после Шага 2:")
print(market_data_df.info())
print(market_data_df.head())

# Сохраняем промежуточный результат на случай сбоя
market_data_df.to_parquet('intermediate_market_data.parquet')
print("\nПромежуточные рыночные данные сохранены в 'intermediate_market_data.parquet'.")

# На этом этапе у нас есть market_data_df с ежедневными ценами, объемами, IMOEX,
# пустыми колонками для MarketCap и фин.данных (или заполненными NaN),
# колонкой для Sector (возможно, 'Unknown'), и присоединенными новостными признаками.