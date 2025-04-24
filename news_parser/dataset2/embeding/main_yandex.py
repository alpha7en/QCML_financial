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
# Параметры Yandex Cloud (ЗАМЕНИТЕ НА ВАШИ!)
YC_FOLDER_ID = "b1gu0dd26bpgkokh9fsk"  # <<<--- ЗАМЕНИТЕ НА ID ВАШЕГО КАТАЛОГА
YC_AUTH_TOKEN = "AQVNzgJ3h6AJ57br7A1PMYnN7NFQtpfijLSmTyTi" # <<<--- ЗАМЕНИТЕ НА ВАШ IAM-ТОКЕН или API-ключ

# Модели эмбеддингов Yandex
# Используем модель для ДОКУМЕНТОВ, т.к. описания компаний - это скорее документы
YC_EMBEDDING_MODEL_URI = f"emb://{YC_FOLDER_ID}/text-search-doc/latest"
# Размерность эмбеддинга для моделей Yandex (уточните в документации, если изменится)
# На момент написания кода, размерность была 1024
EXPECTED_EMBEDDING_DIM = 1024

# Пути к файлам кэша
base_path = './' # Текущая директория (или ваш путь)
descriptions_cache_file = os.path.join(base_path, 'ticker_descriptions_cache.json')
# >>> НОВЫЙ ФАЙЛ КЭША для YANDEX эмбеддингов <<<
embeddings_yandex_cache_file = os.path.join(base_path, 'ticker_embeddings_yandex_cache.pkl')

# Параметры запросов к Yandex Embedding API
# SDK может обрабатывать списки, но большие списки могут вызывать таймауты или ошибки.
# Будем обрабатывать пачками для надежности.
EMBEDDING_BATCH_SIZE_YC = 50 # Можно поэкспериментировать с размером
EMBEDDING_REQUEST_DELAY_SEC_YC = 0.2 # Задержка между запросами (200 мс)

# --- Инициализация Yandex Cloud ML SDK ---
try:
    sdk = YCloudML(folder_id=YC_FOLDER_ID, auth=YC_AUTH_TOKEN)
    print("Yandex Cloud ML SDK инициализирован.")
    # Проверим доступность модели (опционально)
    # available_models = sdk.models.list() # Может потребовать доп. прав или вызвать ошибку, если API изменился
    # print("Доступные модели:", available_models)
except Exception as e:
    raise SystemExit(f"Ошибка инициализации Yandex Cloud ML SDK: {e}. Проверьте folder_id и токен/ключ.")


# --- Функция для получения эмбеддингов от Yandex FM (с кэшированием) ---
def get_yandex_embeddings_with_caching(descriptions_dict, sdk_instance, model_uri, cache_file=embeddings_yandex_cache_file):
    """
    Получает эмбеддинги для текстов из словаря описаний через YC FM SDK, используя кэш.
    Возвращает словарь {ticker: embedding_vector} и размерность эмбеддинга.
    """
    ticker_to_embedding_map = {}
    embedding_dimension = None

    # Загружаем кэш
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if isinstance(cached_data, dict):
                    ticker_to_embedding_map = cached_data
                    if ticker_to_embedding_map:
                        embedding_dimension = len(next(iter(ticker_to_embedding_map.values())))
                        print(f"Загружены Yandex эмбеддинги ({len(ticker_to_embedding_map)} шт., размерность {embedding_dimension}) из {cache_file}")
                    else: print(f"Кэш Yandex эмбеддингов {cache_file} пуст.")
                else: print(f"Неверный формат кэша Yandex эмбеддингов в {cache_file}."); ticker_to_embedding_map = {}
        except Exception as e: print(f"Ошибка загрузки кэша Yandex эмбеддингов: {e}."); ticker_to_embedding_map = {}

    # Определяем тикеры для обработки
    tickers_in_descriptions = list(descriptions_dict.keys())
    tickers_to_embed = [t for t in tickers_in_descriptions if t not in ticker_to_embedding_map]

    if not tickers_to_embed:
        print("Все Yandex эмбеддинги уже есть в кэше.")
        if not embedding_dimension and ticker_to_embedding_map: embedding_dimension = len(next(iter(ticker_to_embedding_map.values())))
        return ticker_to_embedding_map, embedding_dimension

    print(f"Нужно запросить Yandex эмбеддинги для {len(tickers_to_embed)} тикеров (модель: ...{model_uri[-30:]})...")

    new_embeddings = {}
    # Пакетная обработка
    for i in range(0, len(tickers_to_embed), EMBEDDING_BATCH_SIZE_YC):
        batch_tickers = tickers_to_embed[i:i+EMBEDDING_BATCH_SIZE_YC]
        # Берем только тикеры с НЕПУСТЫМИ описаниями
        valid_tickers_in_batch = [t for t in batch_tickers if descriptions_dict.get(t)]
        batch_texts = [descriptions_dict[t] for t in valid_tickers_in_batch]

        if not batch_texts:
            print(f"  Пропуск пустой пачки текстов (тикеры: {batch_tickers}).")
            continue

        print(f"\nОбработка пачки {i//EMBEDDING_BATCH_SIZE_YC + 1} / { (len(tickers_to_embed) + EMBEDDING_BATCH_SIZE_YC - 1) // EMBEDDING_BATCH_SIZE_YC } ({len(batch_texts)} текстов)...")

        try:
            # >>> Вызов Yandex Embedding API через SDK <<<
            # SDK может сам обрабатывать списки, метод text_embeddings ожидает итератор
            # Подаем список текстов напрямую
            embeddings_response = sdk_instance.models.text_embeddings(
                model_uri=model_uri, # Используем URI модели для документов
                texts=batch_texts
            )
            # embeddings_response должен быть списком векторов (списков float)

            # Проверка результата
            if not isinstance(embeddings_response, list) or not embeddings_response:
                 raise ValueError("Ответ API не является списком или пуст.")
            if len(embeddings_response) != len(batch_texts):
                 raise ValueError(f"Количество текстов ({len(batch_texts)}) не совпадает с количеством эмбеддингов ({len(embeddings_response)})")

            # Определяем размерность, если еще не определена
            if not embedding_dimension:
                 embedding_dimension = len(embeddings_response[0])
                 # Проверка соответствия ожидаемой размерности
                 if embedding_dimension != EXPECTED_EMBEDDING_DIM:
                      print(f"ПРЕДУПРЕЖДЕНИЕ: Реальная размерность эмбеддинга ({embedding_dimension}) не совпадает с ожидаемой ({EXPECTED_EMBEDDING_DIM})!")
                 print(f"  Определена размерность Yandex эмбеддинга: {embedding_dimension}")

            # Сопоставляем эмбеддинги с тикерами
            current_batch_embeddings = {}
            for ticker, embedding_list in zip(valid_tickers_in_batch, embeddings_response):
                # Преобразуем список в numpy array
                current_batch_embeddings[ticker] = np.array(embedding_list, dtype=np.float32)
            new_embeddings.update(current_batch_embeddings)
            print(f"  Успешно получено {len(current_batch_embeddings)} Yandex эмбеддингов для пачки.")

        except Exception as e:
            print(f"  !!! ОШИБКА при запросе Yandex эмбеддингов для пачки начиная с {valid_tickers_in_batch[0] if valid_tickers_in_batch else 'N/A'}: {e} !!!")
            # Можно добавить логику повторных попыток, если API возвращает временные ошибки

        # Пауза между запросами
        if i + EMBEDDING_BATCH_SIZE_YC < len(tickers_to_embed):
             time.sleep(EMBEDDING_REQUEST_DELAY_SEC_YC)

    # Обновляем основной словарь и сохраняем кэш
    ticker_to_embedding_map.update(new_embeddings)
    if new_embeddings:
        try:
            with open(cache_file, 'wb') as f: pickle.dump(ticker_to_embedding_map, f)
            print(f"\nОбновленный кэш Yandex эмбеддингов сохранен в {cache_file}")
        except Exception as e: print(f"\nОшибка сохранения кэша Yandex эмбеддингов: {e}")

    # Переопределяем размерность на случай, если кэш был пуст и обработка прошла успешно
    if not embedding_dimension and ticker_to_embedding_map:
        embedding_dimension = len(next(iter(ticker_to_embedding_map.values())))

    return ticker_to_embedding_map, embedding_dimension

# --- Пример использования ---
if __name__ == "__main__":
    # 1. Загружаем описания из кэша (созданного предыдущим скриптом)
    ticker_descriptions = {}
    if os.path.exists(descriptions_cache_file):
        try:
            with open(descriptions_cache_file, 'r', encoding='utf-8') as f:
                ticker_descriptions = json.load(f)
            print(f"Загружены описания ({len(ticker_descriptions)} шт.) из {descriptions_cache_file}")
        except Exception as e:
            print(f"Ошибка загрузки файла описаний: {e}")
    else:
        print(f"Файл с описаниями '{descriptions_cache_file}' не найден. Запустите скрипт для их получения.")

    if ticker_descriptions:
        # 2. Получаем/обновляем эмбеддинги с помощью Yandex SDK
        final_embeddings_map, final_dim = get_yandex_embeddings_with_caching(
            ticker_descriptions,
            sdk, # Передаем инициализированный SDK
            YC_EMBEDDING_MODEL_URI # Передаем URI модели
        )

        if final_embeddings_map:
            print("\n--- Итоговые Yandex Эмбеддинги (пример) ---")
            count = 0
            for ticker, embedding in final_embeddings_map.items():
                print(f"{ticker}: shape={embedding.shape}, dtype={embedding.dtype}, first 5 values={embedding[:5]}")
                count += 1
                if count >= 5: break
            print(f"\nРазмерность эмбеддинга: {final_dim}")
            print(f"Всего эмбеддингов: {len(final_embeddings_map)}")

            # Проверка тикеров без эмбеддингов
            tickers_with_desc = set(ticker_descriptions.keys())
            tickers_with_emb = set(final_embeddings_map.keys())
            missing_embedding_tickers = list(tickers_with_desc - tickers_with_emb)
            if missing_embedding_tickers:
                print(f"\nПредупреждение: Нет эмбеддингов для тикеров с описаниями: {missing_embedding_tickers}")
        else:
            print("\nНе удалось создать Yandex эмбеддинги.")
    else:
        print("\nНет описаний для создания эмбеддингов.")

    # >>> Теперь словарь `final_embeddings_map` содержит эмбеддинги от Yandex <<<