from google import genai
from google.genai import types
import time
import os
import json
import numpy as np
import pickle

# --- КОНФИГУРАЦИЯ ---
# ВАЖНО: Замените на ваш реальный ключ API!
try:
    # Попытка получить ключ из секретов Colab
    from google.colab import userdata
    GEMINI_API_KEY = "AIzaSyAujzD44yEeb3wCgqM1RVGHmwI5DCWWa8A"
    if not GEMINI_API_KEY:
        print("Ключ API Gemini не найден в секретах Colab.")
        GEMINI_API_KEY = input("Введите ваш API ключ Gemini: ") # Только для демонстрации
except ImportError:
     # Если не в Colab, пытаемся получить из переменной окружения
     GEMINI_API_KEY = "AIzaSyAujzD44yEeb3wCgqM1RVGHmwI5DCWWa8A"
     if not GEMINI_API_KEY:
         print("Ключ API Gemini не найден в переменных окружения.")
         GEMINI_API_KEY = input("Введите ваш API ключ Gemini: ") # Только для демонстрации

if not GEMINI_API_KEY:
    raise ValueError("Необходимо предоставить API ключ Gemini!")

client = genai.Client(api_key=GEMINI_API_KEY)
# Модель эмбеддингов Gemini
GEMINI_EMBEDDING_MODEL = "gemini-embedding-exp-03-07" # Используем стабильное имя модели эмбеддингов Gemini

# Пути к файлам
base_path = './' # Текущая директория (или ваш путь на Google Drive)
descriptions_cache_file = os.path.join(base_path, 'ticker_descriptions_cache.json')
# >>> НОВЫЙ ФАЙЛ ДЛЯ КЭША ЭМБЕДДИНГОВ GEMINI <<<
embeddings_gemini_cache_file = os.path.join(base_path, 'ticker_embeddings_gemini_cache.pkl')

# Параметры запросов к Embedding API
EMBEDDING_BATCH_SIZE = 100 # API рекомендует не более 100 текстов за раз
EMBEDDING_REQUEST_DELAY_SEC = 1 # Пауза между запросами

# --- Функция для получения эмбеддингов от Gemini (с кэшированием) ---
def get_embeddings_with_caching(descriptions_dict, cache_file=embeddings_gemini_cache_file):
    """
    Получает эмбеддинги для текстов из словаря описаний, используя кэш.
    Возвращает словарь {ticker: embedding_vector} и размерность эмбеддинга.
    """
    ticker_to_embedding_map = {}
    embedding_dimension = None

    # Пытаемся загрузить кэш эмбеддингов
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                # Проверяем формат кэша (должен быть словарем)
                if isinstance(cached_data, dict):
                    ticker_to_embedding_map = cached_data
                    # Определяем размерность по первому элементу
                    if ticker_to_embedding_map:
                        first_ticker = next(iter(ticker_to_embedding_map))
                        embedding_dimension = len(ticker_to_embedding_map[first_ticker])
                        print(f"Загружены кэшированные эмбеддинги ({len(ticker_to_embedding_map)} шт., размерность {embedding_dimension}) из {cache_file}")
                    else:
                        print(f"Кэш эмбеддингов {cache_file} пуст.")
                else:
                    print(f"Предупреждение: Неверный формат кэша эмбеддингов в {cache_file}. Кэш будет перезаписан.")
                    ticker_to_embedding_map = {} # Очищаем, если формат неверный
        except Exception as e:
            print(f"Ошибка загрузки кэша эмбеддингов: {e}. Запрашиваем заново.")
            ticker_to_embedding_map = {}

    # Определяем, для каких тикеров еще нет эмбеддингов (или кэш пуст)
    tickers_in_descriptions = list(descriptions_dict.keys())
    tickers_to_embed = [t for t in tickers_in_descriptions if t not in ticker_to_embedding_map]

    if not tickers_to_embed:
        print("Все эмбеддинги уже есть в кэше.")
        # Убедимся, что размерность определена, если кэш не пуст
        if not embedding_dimension and ticker_to_embedding_map:
             embedding_dimension = len(next(iter(ticker_to_embedding_map.values())))
        return ticker_to_embedding_map, embedding_dimension

    print(f"Нужно запросить эмбеддинги для {len(tickers_to_embed)} тикеров у {GEMINI_EMBEDDING_MODEL}...")

    new_embeddings = {}
    # Пакетная обработка
    for i in range(0, len(tickers_to_embed), EMBEDDING_BATCH_SIZE):
        batch_tickers = tickers_to_embed[i:i+EMBEDDING_BATCH_SIZE]
        batch_texts = [descriptions_dict[t] for t in batch_tickers if descriptions_dict.get(t)] # Берем только непустые описания

        if not batch_texts:
            print(f"  Пропуск пустой пачки текстов (тикеры: {batch_tickers}).")
            continue

        print(f"\nОбработка пачки {i//EMBEDDING_BATCH_SIZE + 1} / { (len(tickers_to_embed) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE } ({len(batch_texts)} текстов)...")

        try:
            # >>> Вызов Gemini Embedding API <<<
            # Используем task_type='RETRIEVAL_DOCUMENT' как универсальный для описаний
            result = client.models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=batch_texts)
            print(result.embeddings)
            # Проверяем размерность первого полученного эмбеддинга
            if not embedding_dimension and result.embeddings:
                 embedding_dimension = len(result['embedding'][0])
                 print(f"  Определена размерность эмбеддинга: {embedding_dimension}")

            # Сопоставляем эмбеддинги с тикерами
            # Важно: порядок в result['embedding'] соответствует порядку в batch_texts
            current_batch_embeddings = {}
            valid_tickers_in_batch = [t for t in batch_tickers if descriptions_dict.get(t)] # Тикеры, для которых были тексты
            if len(valid_tickers_in_batch) == len(result.get('embedding', [])):
                for ticker, embedding_list in zip(valid_tickers_in_batch, result['embedding']):
                     # Преобразуем список в numpy array для единообразия
                    current_batch_embeddings[ticker] = np.array(embedding_list, dtype=np.float32)
                new_embeddings.update(current_batch_embeddings)
                print(f"  Успешно получено {len(current_batch_embeddings)} эмбеддингов для пачки.")
            else:
                 print(f"  ОШИБКА: Несовпадение количества тикеров ({len(valid_tickers_in_batch)}) и полученных эмбеддингов ({len(result.get('embedding', []))})!")

        except Exception as e:
            print(f"  !!! ОШИБКА при запросе эмбеддингов для пачки начиная с {batch_tickers[0]}: {e} !!!")
            # Можно добавить логику повторных попыток, если нужно

        # Пауза между запросами
        if i + EMBEDDING_BATCH_SIZE < len(tickers_to_embed):
            time.sleep(EMBEDDING_REQUEST_DELAY_SEC)

    # Обновляем основной словарь и сохраняем кэш
    ticker_to_embedding_map.update(new_embeddings)
    if new_embeddings: # Сохраняем только если что-то добавилось
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(ticker_to_embedding_map, f)
            print(f"\nОбновленный кэш эмбеддингов сохранен в {cache_file}")
        except Exception as e:
            print(f"\nОшибка сохранения кэша эмбеддингов: {e}")

    # Убедимся, что размерность определена, если карта не пуста
    if not embedding_dimension and ticker_to_embedding_map:
        embedding_dimension = len(next(iter(ticker_to_embedding_map.values())))


    return ticker_to_embedding_map, embedding_dimension

# --- Пример использования ---
if __name__ == "__main__":
    # 1. Загружаем описания из кэша (предполагается, что предыдущий скрипт их создал)
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
        # 2. Получаем/обновляем эмбеддинги
        final_embeddings_map, final_dim = get_embeddings_with_caching(ticker_descriptions)

        if final_embeddings_map:
            print("\n--- Итоговые Эмбеддинги (пример) ---")
            count = 0
            for ticker, embedding in final_embeddings_map.items():
                print(f"{ticker}: shape={embedding.shape}, dtype={embedding.dtype}, first 5 values={embedding[:5]}")
                count += 1
                if count >= 5: # Выведем несколько
                    break
            print(f"\nРазмерность эмбеддинга: {final_dim}")
            print(f"Всего эмбеддингов: {len(final_embeddings_map)}")

            # Проверка тикеров без эмбеддингов
            missing_embedding_tickers = [t for t in ticker_descriptions.keys() if t not in final_embeddings_map]
            if missing_embedding_tickers:
                print(f"\nПредупреждение: Нет эмбеддингов для тикеров: {missing_embedding_tickers}")
        else:
            print("\nНе удалось создать эмбеддинги.")
    else:
        print("\nНет описаний для создания эмбеддингов.")

    # >>> Теперь у вас есть словарь `final_embeddings_map` и размерность `final_dim` <<<
    # >>> Вы можете использовать их в вашем основном скрипте обучения           <<<