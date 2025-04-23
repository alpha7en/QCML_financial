from google import genai
import time
import os
import json # Для возможного сохранения/загрузки кэша

# --- КОНФИГУРАЦИЯ ---
# ВАЖНО: Замените на ваш реальный ключ API!
# Лучше хранить ключ в секретах Colab или переменных окружения.
try:
    from google.colab import userdata
    GEMINI_API_KEY = "AIzaSyAujzD44yEeb3wCgqM1RVGHmwI5DCWWa8A"
    if not GEMINI_API_KEY:
        print("Ключ API Gemini не найден в секретах Colab.")
        GEMINI_API_KEY = input("Введите ваш API ключ Gemini: ") # Только для демонстрации
except ImportError:
     GEMINI_API_KEY = "AIzaSyAujzD44yEeb3wCgqM1RVGHmwI5DCWWa8A"
     if not GEMINI_API_KEY:
         print("Ключ API Gemini не найден в переменных окружения.")
         GEMINI_API_KEY = input("Введите ваш API ключ Gemini: ") # Только для демонстрации

if not GEMINI_API_KEY:
    raise ValueError("Необходимо предоставить API ключ Gemini!")

client = genai.Client(api_key=GEMINI_API_KEY)

# Модель Gemini для генерации описаний
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17" # Или другая доступная модель
# Параметры запросов
BATCH_SIZE_LLM = 10   # По сколько тикеров за раз спрашивать
REQUEST_DELAY_SEC = 2 # Пауза между запросами к API (секунды)
RETRIES = 3           # Количество повторных попыток для батча

# Путь для кэширования (опционально, но полезно)
base_path = './' # Сохраняем в текущую директорию (или укажите ваш путь)
descriptions_cache_file = os.path.join(base_path, 'ticker_descriptions_cache2.json')

# --- Функция для получения описаний от Gemini (с кэшированием) ---
def get_descriptions_with_caching(all_tickers_list, cache_file=descriptions_cache_file):
    """
    Получает описания для списка тикеров, используя кэш.
    Возвращает словарь {ticker: description}.
    """
    ticker_descriptions = {}
    # Пытаемся загрузить кэш
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                ticker_descriptions = json.load(f)
            print(f"Загружены кэшированные описания ({len(ticker_descriptions)} шт.) из {cache_file}")
        except Exception as e:
            print(f"Ошибка загрузки кэша описаний: {e}. Запрашиваем заново.")
            ticker_descriptions = {}

    # Определяем, для каких тикеров еще нет описаний
    tickers_to_fetch = [t for t in all_tickers_list if t not in ticker_descriptions or not ticker_descriptions[t]]
    tickers_to_fetch = sorted(list(set(tickers_to_fetch))) # Уникальные и сортированные

    if not tickers_to_fetch:
        print("Все описания уже есть в кэше.")
        return ticker_descriptions

    print(f"Нужно запросить описания для {len(tickers_to_fetch)} тикеров у {GEMINI_MODEL}...")
    new_descriptions = {}



    for i in range(0, len(tickers_to_fetch), BATCH_SIZE_LLM):
        batch_tickers = tickers_to_fetch[i:i+BATCH_SIZE_LLM]
        print(f"\nОбработка пачки {i//BATCH_SIZE_LLM + 1} / { (len(tickers_to_fetch) + BATCH_SIZE_LLM - 1) // BATCH_SIZE_LLM } ({len(batch_tickers)} тикеров)...")
        print(f"  Тикеры: {', '.join(batch_tickers)}")

        # Формируем промпт
        prompt = f"Для каждого тикера из списка ниже, торгующегося на Московской Бирже, дай очень краткое описание (1, максимум 2 предложения) основного бизнеса и основного сектора экономики, меньше конкретных названий или регалий, тебе нужно фактически описать занимаемую нишу в экономике. Описание кратко. Формат ответа: 'ТИКЕР: Описание'. Каждый тикер на новой строке.\n\nСписок тикеров:\n"
        prompt += "\n".join(batch_tickers)

        current_retries = 0
        success = False
        while current_retries < RETRIES and not success:
            if current_retries > 0:
                print(f"  Попытка {current_retries + 1}/{RETRIES}...")
            try:
                response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                response_text = response.text # Получаем текст ответа

                # Парсим ответ
                batch_results = {}
                lines = response_text.strip().split('\n')
                for line in lines:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        ticker = parts[0].replace('-', '').strip().upper() # Очистка тикера
                        desc = parts[1].strip()
                        if ticker in batch_tickers: # Проверяем, что это тикер из нашего запроса
                            batch_results[ticker] = desc
                            print(f"    + Получено для {ticker}: {desc[:60]}...")
                        # else:
                        #     print(f"    Предупреждение: Тикер '{ticker}' из ответа не был в текущем запросе.")
                    # else:
                    #     print(f"    Предупреждение: Не удалось распарсить строку: '{line[:60]}...'")

                # Проверяем, все ли тикеры из батча получили описание
                missing_in_batch = [t for t in batch_tickers if t not in batch_results]
                if not missing_in_batch:
                    print("  Все тикеры из пачки обработаны успешно.")
                    new_descriptions.update(batch_results)
                    success = True # Пачка обработана
                else:
                    print(f"  Не получены описания для: {missing_in_batch}. Повтор запроса для пачки...")
                    current_retries += 1
                    if current_retries >= RETRIES:
                        print(f"  ПРЕВЫШЕНО ЧИСЛО ПОПЫТОК для пачки. Пропуск тикеров: {missing_in_batch}")
                        # Добавляем пустые строки для пропущенных, чтобы кэш обновился
                        for t in missing_in_batch:
                            new_descriptions[t] = "" # Пустое описание для пропущенных
                    else:
                        print(f"  Пауза {REQUEST_DELAY_SEC} сек перед повторной попыткой...")
                        time.sleep(REQUEST_DELAY_SEC)

            except Exception as e:
                print(f"  !!! ОШИБКА при запросе к Gemini: {e} !!!")
                current_retries += 1
                if current_retries >= RETRIES:
                    print(f"  ПРЕВЫШЕНО ЧИСЛО ПОПЫТОК из-за ошибки. Пропуск тикеров: {batch_tickers}")
                    for t in batch_tickers: new_descriptions[t] = "" # Пустое описание
                    break # Выход из цикла while retries
                else:
                    print(f"  Пауза {REQUEST_DELAY_SEC * (current_retries + 1)} сек перед повторной попыткой...") # Увеличим паузу при ошибке
                    time.sleep(REQUEST_DELAY_SEC * (current_retries + 1))

        # Небольшая пауза между успешными запросами пачек
        if success and i + BATCH_SIZE_LLM < len(tickers_to_fetch):
            time.sleep(REQUEST_DELAY_SEC / 2) # Уменьшенная пауза между успешными

    # Обновляем основной словарь и сохраняем кэш
    ticker_descriptions.update(new_descriptions)
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(ticker_descriptions, f, ensure_ascii=False, indent=4)
        print(f"\nОбновленный кэш описаний сохранен в {cache_file}")
    except Exception as e:
        print(f"\nОшибка сохранения кэша описаний: {e}")

    return ticker_descriptions

# --- Пример использования ---
if __name__ == "__main__":
    # >>> ЗАМЕНИТЕ ЭТОТ СПИСОК НА ВАШ РЕАЛЬНЫЙ СПИСОК ТИКЕРОВ <<<
    # Например, загрузите его из файла или получите из вашего DataFrame
    list_of_tickers_to_describe = [ 'UNKL', 'UPRO', 'URKZ', 'USBN', 'UTAR', 'UWGN', 'VEON-RX', 'VGSB', 'VGSBP', 'VJGZ', 'VJGZP', 'VKCO', 'VLHZ', 'VRSB', 'VRSBP', 'VSEH', 'VSMO', 'VSYD', 'VSYDP', 'VTBR', 'WTCM', 'WTCMP', 'WUSH', 'X5', 'YAKG', 'YDEX', 'YKEN', 'YKENP', 'YRSB', 'YRSBP', 'ZAYM', 'ZILL', 'ZVEZ']

    # Убедитесь, что тикеры в верхнем регистре и без лишних символов

    print(f"Запрашиваем описания для {len(list_of_tickers_to_describe)} тикеров...")

    # Получаем описания (функция сама обработает кэш)
    final_descriptions = get_descriptions_with_caching(list_of_tickers_to_describe)

    print("\n--- Итоговые Описания ---")
    # Выведем несколько для примера
    count = 0
    for ticker, desc in final_descriptions.items():
        if ticker in list_of_tickers_to_describe: # Выводим только запрошенные
            print(f"{ticker}: {desc}")
            count += 1
            if count >= 15: # Ограничим вывод
                break
    print("-" * 20)
    print(f"Всего получено/загружено описаний: {len(final_descriptions)}")

    # Проверка пропущенных
    missing_tickers = [t for t in list_of_tickers_to_describe if t not in final_descriptions or not final_descriptions[t]]
    if missing_tickers:
        print(f"\nНе удалось получить описания для следующих тикеров: {missing_tickers}")
    else:
        print("\nВсе запрошенные тикеры имеют описания.")