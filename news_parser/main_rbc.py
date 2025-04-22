import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import time
import logging
import os

# --- Настройки ---
OUTPUT_CSV = "lenta_economics_last_5_years.csv" # Новое имя файла
TARGET_CATEGORY = "economics" # Целевая категория Lenta.ru
BATCH_SIZE = 100
YEARS_TO_SCRAPE = 5
REQUEST_DELAY_SECONDS = 1.5 # Задержка между запросами (к разным страницам/датам)
REQUEST_TIMEOUT_SECONDS = 40

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Заголовки для HTML запросов
HTML_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36 (LentaScraper)"
}

def fetch_lenta_articles_for_day(date_obj, category, session):
    """
    Получает и парсит HTML-страницы архива Lenta.ru за указанную дату
    для указанной категории, обрабатывая пагинацию.
    Возвращает список словарей {'date': date_str, 'headline': headline}.
    """
    articles_for_day = []
    page = 1
    # Форматируем дату для URL: YYYY/MM/DD
    date_url_part = date_obj.strftime("%Y/%m/%d")
    date_str_iso = date_obj.strftime("%Y-%m-%d") # Для записи в CSV

    while True: # Цикл по страницам пагинации
        # Формируем URL для текущей страницы
        url = f"https://lenta.ru/rubrics/{category}/{date_url_part}/page/{page}/"
        logging.info(f"-----> [ЗАПРОС HTML] Дата: {date_str_iso}, Страница: {page}, URL: {url}")

        try:
            resp = session.get(url, headers=HTML_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            logging.info(f"<----- [ОТВЕТ HTML] Дата: {date_str_iso}, Страница: {page}, Статус: {resp.status_code}")

            # Lenta возвращает 200 даже если страница пустая или ее нет,
            # поэтому проверяем наличие контента или кнопки "Дальше"
            if resp.status_code == 404:
                 logging.warning(f"[{date_str_iso}] Страница {page} не найдена (404). Завершение для этой даты.")
                 break # Прерываем цикл по страницам для этой даты

            resp.raise_for_status() # Проверка на другие ошибки HTTP

            soup = BeautifulSoup(resp.text, 'lxml')

            # Ищем контейнеры новостей на текущей странице
            article_elements = soup.select("li.archive-page__item._news")
            logging.info(f"[{date_str_iso}] На странице {page} найдено {len(article_elements)} новостей.")

            if not article_elements and page > 1 : # Если на странице > 1 нет статей, значит пагинация закончилась
                logging.info(f"[{date_str_iso}] На странице {page} статьи не найдены, пагинация завершена.")
                break
            elif not article_elements and page == 1:
                 logging.info(f"[{date_str_iso}] Новостей в разделе '{category}' за эту дату нет.")
                 break # Нет новостей за этот день

            # Извлекаем заголовки
            for item in article_elements:
                title_tag = item.select_one("h3.card-full-news__title")
                headline = title_tag.get_text(strip=True) if title_tag else ""

                if headline:
                    articles_for_day.append({
                        "date": date_str_iso,
                        "headline": headline,
                        "summary": "" # Добавляем пустую колонку для единообразия
                    })
                else:
                    logging.warning(f"[{date_str_iso} стр. {page}] Найден элемент без заголовка.")

            # Проверяем наличие кнопки "Дальше"
            # Ищем ссылку с классом loadmore, которая НЕ содержит текст "Назад"
            next_page_link = soup.select_one(f'a.loadmore[href$="/page/{page+1}/"]')
            # Альтернативный, менее строгий вариант:
            # next_page_link = soup.select_one('a.loadmore:-soup-contains("Назад")')

            if next_page_link:
                logging.info(f"[{date_str_iso}] Найдена ссылка на следующую страницу ({page+1}). Продолжаем.")
                page += 1
                time.sleep(REQUEST_DELAY_SECONDS) # Пауза перед запросом следующей страницы
            else:
                logging.info(f"[{date_str_iso}] Ссылка на следующую страницу не найдена. Завершение для этой даты.")
                break # Конец пагинации

        except requests.exceptions.Timeout:
            logging.error(f"[{date_str_iso} стр. {page}] Ошибка: Запрос HTML превысил таймаут ({REQUEST_TIMEOUT_SECONDS} сек) для URL {url}.")
            # Можно добавить логику повторных попыток или просто пропустить день/страницу
            return None # Сигнализируем об ошибке
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else 'N/A'
            logging.error(f"[{date_str_iso} стр. {page}] Ошибка сети или HTTP (статус {status_code}) при запросе HTML {url}: {e}")
            return None # Сигнализируем об ошибке
        except Exception as e:
            logging.error(f"[{date_str_iso} стр. {page}] Непредвиденная ошибка при обработке HTML {url}: {e}", exc_info=True)
            return None # Сигнализируем об ошибке

    return articles_for_day # Возвращаем список статей, собранных со всех страниц за день

# --- Функция основного цикла ---
def scrape_lenta_last_n_years(category=TARGET_CATEGORY, years=YEARS_TO_SCRAPE, batch_size=BATCH_SIZE):
    """
    Главная функция для скрапинга архива Lenta.ru за последние N лет по категории.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365 + (years // 4)) # Примерный учет високосных лет
    delta = timedelta(days=1)

    logging.info(f"--- Начало скрапинга Lenta.ru (категория: {category}) ---")
    logging.info(f"Диапазон дат: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
    logging.info(f"Результаты будут сохранены в: {OUTPUT_CSV}")
    logging.info(f"Задержка между запросами: {REQUEST_DELAY_SECONDS} сек.")

    # Определяем колонки CSV
    csv_columns = ["date", "headline", "summary"]
    try:
        pd.DataFrame(columns=csv_columns) \
          .to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        logging.info(f"Файл {OUTPUT_CSV} создан/перезаписан с заголовками: {', '.join(csv_columns)}.")
    except Exception as e:
        logging.error(f"Не удалось создать/перезаписать файл {OUTPUT_CSV}: {e}")
        return

    buffer = []
    total_saved = 0
    current_date = start_date
    processed_days = 0
    total_days = (end_date - start_date).days + 1

    with requests.Session() as session:
        while current_date <= end_date:
            processed_days += 1
            progress = (processed_days / total_days) * 100 if total_days > 0 else 0
            # Логируем прогресс реже
            if processed_days % 50 == 1 or current_date >= end_date or processed_days <= 3:
                logging.info(f"Обработка даты: {current_date.strftime('%Y-%m-%d')} ({processed_days}/{total_days}, {progress:.1f}%)")

            # Вызываем функцию, которая обрабатывает пагинацию для текущей даты
            day_articles = fetch_lenta_articles_for_day(current_date, category, session)

            if day_articles is not None: # Если не было критической ошибки при запросе
                if day_articles: # Если статьи за этот день были найдены
                    buffer.extend(day_articles)
                    logging.info(f"[{current_date.strftime('%Y-%m-%d')}] Добавлено в буфер {len(day_articles)} статей. Всего в буфере: {len(buffer)}.")
                # else: # Логирование дней без статей происходит внутри fetch_...
            else:
                logging.warning(f"[{current_date.strftime('%Y-%m-%d')}] Пропуск дня из-за ошибки запроса/обработки.")

            # Сброс в файл
            if len(buffer) >= batch_size or (current_date >= end_date and buffer):
                try:
                    df = pd.DataFrame(buffer, columns=csv_columns)
                    df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding='utf-8-sig')
                    saved_count = len(buffer)
                    total_saved += saved_count
                    logging.info(f"→ Записано в CSV: {saved_count} статей (всего сохранено {total_saved})")
                    buffer.clear()
                except Exception as e:
                     logging.error(f"Не удалось записать пакет в {OUTPUT_CSV}: {e}")

            current_date += delta
            # Паузу между запросами к РАЗНЫМ ДАТАМ можно сделать чуть больше,
            # так как пауза между страницами ОДНОГО дня уже есть
            if current_date <= end_date:
                 time.sleep(REQUEST_DELAY_SECONDS * 1.2) # Небольшое увеличение паузы между днями

    logging.info(f"--- Скрапинг Lenta.ru (категория: {category}) завершён ---")
    logging.info(f"Всего сохранено статей: {total_saved}")
    logging.info(f"Данные сохранены в файле: {OUTPUT_CSV}")


if __name__ == "__main__":
    # Убедись, что установлены: pip install requests pandas beautifulsoup4 lxml
    try:
        scrape_lenta_last_n_years(category=TARGET_CATEGORY, years=YEARS_TO_SCRAPE)
    except KeyboardInterrupt:
        logging.info("Процесс прерван пользователем.")
    except Exception as e:
        logging.error(f"Критическая ошибка в основном процессе: {e}", exc_info=True)