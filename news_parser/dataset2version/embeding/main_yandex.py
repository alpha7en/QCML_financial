import pandas as pd
import numpy as np
# >>> ИМПОРТ Yandex Cloud ML SDK <<<
from yandex_cloud_ml_sdk import YCloudML
# <<<------------------------------>>>
import time
import os
import json
import pickle

# --- КОНФИГУРАЦИЯ ---
YC_FOLDER_ID = "b1gu0dd26bpgkokh9fsk"  # <<<--- ЗАМЕНИТЕ НА ID ВАШЕГО КАТАЛОГА
YC_AUTH_TOKEN = "AQVNzgJ3h6AJ57br7A1PMYnN7NFQtpfijLSmTyTi" # <<<--- ЗАМЕНИТЕ НА ВАШ IAM-ТОКЕН или API-ключ


# >>> Используем КОРОТКИЕ имена моделей из примера <<<
YC_EMBEDDING_MODEL_NAME = "text-search-doc" # Для описаний используем 'doc'
EXPECTED_EMBEDDING_DIM = 1024 # Уточните, если необходимо

base_path = './'
descriptions_cache_file = os.path.join(base_path, 'ticker_descriptions_cache.json')
embeddings_yandex_cache_file = os.path.join(base_path, 'ticker_embeddings_yandex_cache.pkl')

EMBEDDING_BATCH_SIZE_YC = 50 # Размер пачки для эмбеддингов
EMBEDDING_REQUEST_DELAY_SEC_YC = 0.2

# --- Инициализация Yandex Cloud ML SDK ---
try:
    sdk = YCloudML(folder_id=YC_FOLDER_ID, auth=YC_AUTH_TOKEN)
    print("Yandex Cloud ML SDK инициализирован.")
    # >>> Получаем объект модели ОДИН РАЗ <<<
    embedding_model_runner = sdk.models.text_embeddings("doc")
    print(f"Объект модели '{YC_EMBEDDING_MODEL_NAME}' получен.")
except Exception as e:
    raise SystemExit(f"Ошибка инициализации SDK или получения объекта модели: {e}")


# --- Функция для получения эмбеддингов от Yandex FM (с кэшированием) ---
# --- Функция для получения эмбеддингов от Yandex FM (с кэшированием) ---
# --- Функция для получения эмбеддингов от Yandex FM (с кэшированием) ---
def get_yandex_embeddings_with_caching(descriptions_dict, model_runner, cache_file=embeddings_yandex_cache_file):
    """
    Получает эмбеддинги для текстов из словаря описаний через YC FM SDK,
    используя кэш и вызывая .run() для КАЖДОГО текста.
    Возвращает словарь {ticker: embedding_vector} и размерность эмбеддинга.
    """
    ticker_to_embedding_map = {}
    embedding_dimension = None

    # Загружаем кэш (без изменений)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if isinstance(cached_data, dict):
                    ticker_to_embedding_map = cached_data
                    if ticker_to_embedding_map:
                        # Ищем первый валидный эмбеддинг для определения размерности
                        for emb in ticker_to_embedding_map.values():
                            if isinstance(emb, np.ndarray) and emb.size > 0:
                                embedding_dimension = len(emb)
                                break
                        print(f"Загружены Yandex эмбеддинги ({len(ticker_to_embedding_map)} шт., размерность {embedding_dimension}) из {cache_file}")
                    else: print(f"Кэш Yandex эмбеддингов {cache_file} пуст.")
                else: print(f"Неверный формат кэша Yandex эмбеддингов в {cache_file}."); ticker_to_embedding_map = {}
        except Exception as e: print(f"Ошибка загрузки кэша Yandex эмбеддингов: {e}."); ticker_to_embedding_map = {}

    # Определяем тикеры для обработки
    tickers_in_descriptions = list(descriptions_dict.keys())
    # Обрабатываем тикеры, которых нет в кэше ИЛИ для которых в кэше пустой эмбеддинг
    tickers_to_embed = [t for t in tickers_in_descriptions if t not in ticker_to_embedding_map or not isinstance(ticker_to_embedding_map.get(t), np.ndarray) or ticker_to_embedding_map.get(t).size == 0]
    tickers_to_embed = sorted(list(set(tickers_to_embed))) # Уникальные

    if not tickers_to_embed:
        print("Все Yandex эмбеддинги уже есть в кэше.")
        if not embedding_dimension and ticker_to_embedding_map: # Повторная попытка определить размерность
             for emb in ticker_to_embedding_map.values():
                 if isinstance(emb, np.ndarray) and emb.size > 0:
                     embedding_dimension = len(emb); break
        return ticker_to_embedding_map, embedding_dimension

    print(f"Нужно запросить/перезапросить Yandex эмбеддинги для {len(tickers_to_embed)} тикеров (модель: {YC_EMBEDDING_MODEL_NAME})...")

    new_embeddings = {}
    processed_count = 0
    total_to_process = len(tickers_to_embed)

    # Обработка ПО ОДНОМУ тикеру
    for ticker in tickers_to_embed:
        processed_count += 1
        description = descriptions_dict.get(ticker)

        if not description:
            print(f"({processed_count}/{total_to_process}) Пропуск тикера {ticker}: нет описания.")
            new_embeddings[ticker] = np.array([], dtype=np.float32)
            continue

        print(f"({processed_count}/{total_to_process}) Запрос эмбеддинга для {ticker}...")

        try:
            # >>> Вызов .run() для одного текста <<<
            embedding_result = model_runner.run(description) # Ожидаем вектор (list или tuple)

            # >>> ИСПРАВЛЕННАЯ ПРОВЕРКА РЕЗУЛЬТАТА <<<
            # Проверяем, что результат - это итерируемый объект (list/tuple) и он не пустой
            if not hasattr(embedding_result, '__iter__') or not embedding_result:
                 raise ValueError(f"Ответ API не является итерируемым или пуст: {type(embedding_result)}")

            # Преобразуем в numpy массив сразу
            embedding_vector = np.array(embedding_result, dtype=np.float32)

            # Определяем размерность (только один раз)
            current_dim = len(embedding_vector)
            if not embedding_dimension:
                 embedding_dimension = current_dim
                 if embedding_dimension != EXPECTED_EMBEDDING_DIM: print(f"ПРЕДУПРЕЖДЕНИЕ: Размерность ({embedding_dimension}) != ожидаемой ({EXPECTED_EMBEDDING_DIM})!")
                 print(f"  Определена размерность Yandex эмбеддинга: {embedding_dimension}")
            elif current_dim != embedding_dimension:
                 print(f"  !!! ОШИБКА: Размерность для {ticker} ({current_dim}) != предыдущей ({embedding_dimension})!!!")
                 new_embeddings[ticker] = np.array([], dtype=np.float32) # Сохраняем пустой
                 continue

            # Сохраняем эмбеддинг
            new_embeddings[ticker] = embedding_vector
            # print(f"  Успешно получен эмбеддинг для {ticker}.")

        except Exception as e:
            print(f"  !!! ОШИБКА при запросе Yandex эмбеддинга для {ticker}: {e} !!!")
            new_embeddings[ticker] = np.array([], dtype=np.float32)

        # Пауза между запросами
        time.sleep(EMBEDDING_REQUEST_DELAY_SEC_YC)


    # Обновление и сохранение кэша
    ticker_to_embedding_map.update(new_embeddings)
    if new_embeddings:
        try:
            with open(cache_file, 'wb') as f: pickle.dump(ticker_to_embedding_map, f)
            print(f"\nОбновленный кэш Yandex эмбеддингов сохранен в {cache_file}")
        except Exception as e: print(f"\nОшибка сохранения кэша Yandex эмбеддингов: {e}")

    if not embedding_dimension and ticker_to_embedding_map:
         for emb in ticker_to_embedding_map.values():
             if isinstance(emb, np.ndarray) and emb.size > 0: embedding_dimension = len(emb); break

    return ticker_to_embedding_map, embedding_dimension

# --- Пример использования (без изменений) ---
if __name__ == "__main__":
    # ... (код загрузки описаний) ...
    ticker_descriptions = {}
    if os.path.exists(descriptions_cache_file):
        try:
            with open(descriptions_cache_file, 'r', encoding='utf-8') as f:
                ticker_descriptions = json.load(f)
            print(f"Загружены описания ({len(ticker_descriptions)} шт.) из {descriptions_cache_file}")
        except Exception as e: print(f"Ошибка загрузки файла описаний: {e}")
    else: print(f"Файл с описаниями '{descriptions_cache_file}' не найден.")

    if ticker_descriptions:
        # Получаем/обновляем эмбеддинги
        final_embeddings_map, final_dim = get_yandex_embeddings_with_caching(
            ticker_descriptions,
            embedding_model_runner, # Передаем созданный объект модели
            # cache_file - используется значение по умолчанию
        )
        # ... (вывод результатов без изменений) ...
        if final_embeddings_map:
            print("\n--- Итоговые Yandex Эмбеддинги (пример) ---")
            count = 0
            for ticker, embedding in final_embeddings_map.items():
                if isinstance(embedding, np.ndarray) and embedding.size > 0:
                    print(f"{ticker}: shape={embedding.shape}, dtype={embedding.dtype}, first 5 values={embedding[:5]}")
                    count += 1
                    if count >= 5: break
            print(f"\nРазмерность эмбеддинга: {final_dim}"); print(f"Всего записей в карте эмбеддингов: {len(final_embeddings_map)}")
            tickers_with_desc = set(ticker_descriptions.keys()); tickers_with_emb = set(final_embeddings_map.keys())
            missing_embedding_tickers = list(tickers_with_desc - tickers_with_emb)
            empty_embedding_tickers = [t for t, emb in final_embeddings_map.items() if not isinstance(emb, np.ndarray) or emb.size == 0]
            if missing_embedding_tickers or empty_embedding_tickers:
                 print(f"\nПредупреждение: Проблемы с эмбеддингами для:")
                 if missing_embedding_tickers: print(f"  - Отсутствуют в карте: {missing_embedding_tickers}")
                 if empty_embedding_tickers: print(f"  - Пустые: {empty_embedding_tickers}")
        else: print("\nНе удалось создать Yandex эмбеддинги.")
    else: print("\nНет описаний для создания эмбеддингов.")