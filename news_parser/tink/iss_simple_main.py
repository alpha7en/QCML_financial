import requests
import apimoex
import pandas as pd

# Тикер для поиска
ticker_to_find = 'USDRUB' # !!! ИЩЕМ ПРОСТО USDRUB !!!

print(f"Пытаюсь найти информацию об инструменте: {ticker_to_find}...\n")

with requests.Session() as session:
    print(f"--- Поиск информации для: {ticker_to_find} ---")
    try:
        # Запрашиваем ВСЕ доступные колонки
        data = apimoex.find_securities(session, ticker_to_find, columns=None)

        if data:
            df_info = pd.DataFrame(data)
            print(f"Найденная информация ({len(df_info)} строк):")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(df_info)
            # Выделим потенциально полезные колонки
            relevant_cols = [col for col in df_info.columns if 'secid' == col.lower() or 'board' in col.lower() or 'market' in col.lower() or 'engine' in col.lower() or 'type' in col.lower() or 'group' in col.lower()]
            if relevant_cols:
                print("\nПотенциально релевантные колонки:")
                # Показываем только релевантные колонки для найденных строк
                print(df_info[relevant_cols].to_string())
            else:
                print("\nНе найдено колонок с 'secid', 'board', 'market', 'engine', 'type', 'group'.")

        else:
            print("Инструмент не найден.")

    except Exception as e:
        print(f"Произошла ошибка при поиске {ticker_to_find}: {e}")
    print("-" * (len(ticker_to_find) + 28) + "\n")

print("Поиск завершен.")