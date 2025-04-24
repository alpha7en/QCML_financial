import requests
import json
from datetime import datetime, timedelta
import time
import logging
import sys # Для выхода, если разведка не нужна

# --- Настройки ---
# Сколько дней просканировать для сбора рубрик
DAYS_TO_SCAN_FOR_RUBRICS = 50
YEARS_BACK = 5 # Глубина для определения диапазона дат
REQUEST_DELAY_SECONDS = 1.0 # Можно чуть уменьшить задержку для разведки
REQUEST_TIMEOUT_SECONDS = 30

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Базовый URL для AJAX-запросов
NEWSLINE_URL = "https://www.kommersant.ru/news/newsline"

# Заголовки (оставляем как были)
BASE_HEADERS = {
    "accept": "*/*",
    "accept-language": "ru,en;q=0.9,en-GB;q=0.8,en-US;q=0.7",
    "priority": "u=1, i",
    "sec-ch-ua": "\"Microsoft Edge\";v=\"135\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-kl-kfa-ajax-request": "Ajax_Request",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0"
}

def fetch_rubrics_from_day(date_str, session):
    """
    Запрашивает новости за день и возвращает set уникальных рубрик ('CrumbName') за этот день.
    Возвращает None в случае ошибки.
    """
    referer_url = f"https://www.kommersant.ru/archive/news/{date_str}"
    headers = BASE_HEADERS.copy()
    headers["Referer"] = referer_url

    try:
        resp = session.get(NEWSLINE_URL, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()

        rubrics_today = set() # Используем set для уникальности
        articles_list = data.get('docs', [])

        if not isinstance(articles_list, list):
             logging.warning(f"[{date_str}] Ожидался список в ключе 'docs', получен {type(articles_list)}. Ответ: {str(data)[:200]}...")
             # Возвращаем пустой set, чтобы не считать день ошибочным, если просто нет 'docs'
             return set()

        for item in articles_list:
            if not isinstance(item, dict):
                continue # Просто пропускаем некорректные элементы

            crumb_name = item.get('CrumbName')
            # Добавляем только непустые строки
            if crumb_name and isinstance(crumb_name, str) and crumb_name.strip():
                rubrics_today.add(crumb_name.strip())

        return rubrics_today

    except requests.exceptions.Timeout:
        logging.error(f"[{date_str}] Ошибка: Запрос превысил таймаут ({REQUEST_TIMEOUT_SECONDS} сек).")
        return None # Ошибка запроса
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 'N/A'
        logging.error(f"[{date_str}] Ошибка сети или HTTP (статус {status_code}) при запросе {NEWSLINE_URL} с Referer {referer_url}: {e}")
        return None # Ошибка запроса
    except json.JSONDecodeError:
        logging.error(f"[{date_str}] Ошибка: Не удалось декодировать JSON. Ответ: {resp.text[:200]}...")
        return None # Ошибка запроса
    except Exception as e:
        logging.error(f"[{date_str}] Непредвиденная ошибка при обработке рубрик: {e}", exc_info=False) # Traceback не нужен для разведки
        return None # Ошибка обработки


def discover_all_rubrics(days_to_scan=DAYS_TO_SCAN_FOR_RUBRICS, years_back=YEARS_BACK):
    """
    Сканирует указанное количество дней и собирает все уникальные рубрики (CrumbName).
    """
    all_found_rubrics = set()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years_back * 365 + (years_back // 4))
    delta = timedelta(days=1)
    days_processed = 0
    days_with_errors = 0
    current_date = start_date
    total_days_in_range = (end_date - start_date).days + 1
    actual_days_to_scan = min(days_to_scan, total_days_in_range) # Не сканируем больше дней, чем есть в диапазоне

    logging.info(f"--- РЕЖИМ РАЗВЕДКИ РУБРИК (CrumbName) ---")
    logging.info(f"Сканируем первые {actual_days_to_scan} дней из диапазона {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
    logging.info(f"Цель: собрать список всех уникальных значений поля 'CrumbName'.")

    with requests.Session() as session:
        while current_date <= end_date and days_processed < actual_days_to_scan:
            day_str = current_date.strftime("%Y-%m-%d")
            logging.info(f"Сканирование дня {day_str} ({days_processed + 1}/{actual_days_to_scan})...")

            rubrics_from_day = fetch_rubrics_from_day(day_str, session)

            if rubrics_from_day is not None: # Запрос прошел (даже если рубрик 0)
                new_rubrics_found = rubrics_from_day - all_found_rubrics
                if new_rubrics_found:
                    logging.info(f"Найдены новые рубрики ({len(new_rubrics_found)}): {', '.join(sorted(list(new_rubrics_found)))}")
                    all_found_rubrics.update(rubrics_from_day)
                else:
                    # Лог, что новых рубрик нет, можно закомментировать для краткости
                    # logging.info(f"Новых рубрик за {day_str} не найдено.")
                    pass
                days_processed += 1
            else:
                # Ошибка уже залогирована в fetch_rubrics_from_day
                logging.warning(f"Пропуск дня {day_str} из-за ошибки запроса/обработки.")
                days_with_errors += 1
                # Если много ошибок подряд, возможно, стоит прерваться
                if days_with_errors > 5 and days_processed == 0:
                     logging.error("Слишком много ошибок в начале сканирования. Прерывание.")
                     break
                if days_with_errors > 10:
                     logging.error("Слишком много ошибок при сканировании. Прерывание.")
                     break


            current_date += delta
            # Пауза только если не последний день сканирования
            if days_processed < actual_days_to_scan and current_date <= end_date:
                 time.sleep(REQUEST_DELAY_SECONDS)

    logging.info(f"--- Разведка рубрик завершена ---")
    logging.info(f"Успешно обработано дней: {days_processed}")
    if days_with_errors > 0:
        logging.warning(f"Дней с ошибками (пропущено): {days_with_errors}")

    if not all_found_rubrics:
        logging.warning("Не удалось обнаружить ни одной рубрики.")
        return []

    logging.info(f"Обнаружено уникальных рубрик ({len(all_found_rubrics)}):")
    sorted_rubrics = sorted(list(all_found_rubrics))
    for rubric in sorted_rubrics:
        # Используем print для чистого вывода списка в конце, чтобы легко скопировать
        print(f"- {rubric}")

    logging.info("--- Конец списка рубрик ---")
    return sorted_rubrics

if __name__ == "__main__":
    try:
        # Запускаем только разведку рубрик
        discovered_rubrics = discover_all_rubrics()
        if discovered_rubrics:
            logging.info("Скопируй список рубрик выше и реши, какие из них тебе нужны для фильтрации.")
        else:
            logging.info("Не удалось получить список рубрик. Проверь логи ошибок.")

    except KeyboardInterrupt:
        logging.info("Процесс разведки прерван пользователем.")
    except Exception as e:
        logging.error(f"Критическая ошибка в процессе разведки: {e}", exc_info=True)

    # Этот скрипт предназначен только для разведки, поэтому основной парсинг не запускаем.
    # Если бы это был основной скрипт, здесь мог бы быть выбор режима работы.
    logging.info("Скрипт завершил работу (режим разведки рубрик).")