import requests
from bs4 import BeautifulSoup # Убедись, что установлена: pip install beautifulsoup4 lxml
from datetime import datetime, timedelta
import pandas as pd
import time
import logging
import os # Больше не нужен для отладки, но оставим на всякий случай

# --- Настройки ---
OUTPUT_CSV = "kommersant_business_politics_archive_last_5_years.csv" # Имя файла
TARGET_RUBRICS = ["Бизнес", "Экономика"] # Целевые рубрики
BATCH_SIZE = 100
YEARS_TO_SCRAPE = 5 # Количество лет для парсинга
REQUEST_DELAY_SECONDS = 2.0 # Задержка между запросами
REQUEST_TIMEOUT_SECONDS = 45 # Таймаут запроса

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Заголовки для HTML запросов
HTML_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36 (Scraper)"
}

def fetch_articles_by_date_html(date_str, session):
    """
    Получает и парсит HTML-страницу архива за указанную дату,
    используя КОРРЕКТНЫЕ селекторы, и фильтрует статьи по TARGET_RUBRICS.
    """
    url = f"https://www.kommersant.ru/archive/news/{date_str}"
    logging.info(f"-----> [ЗАПРОС HTML ДЛЯ ДАТЫ: {date_str}] URL: {url}")

    try:
        resp = session.get(url, headers=HTML_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        logging.info(f"<----- [ОТВЕТ HTML ДЛЯ {date_str}] Статус код: {resp.status_code}")
        resp.raise_for_status()

        # Парсим HTML
        soup = BeautifulSoup(resp.text, 'lxml')

        results = []
        # ИЗМЕНЕНО: Используем КОРРЕКТНЫЙ селектор для контейнера новости
        article_elements = soup.select("article.rubric_lenta__item")
        total_found_today = len(article_elements)
        saved_today = 0

        logging.info(f"[{date_str}] Найдено {total_found_today} элементов <article class='rubric_lenta__item'> при парсинге.")

        if not article_elements:
            # Если и этот селектор не найдет, значит страница действительно пустая или снова изменилась
            logging.warning(f"[{date_str}] Статей на странице не найдено. Возможно, день без новостей или структура HTML изменилась.")
            return []

        for item in article_elements:
            # ИЗМЕНЕНО: Извлекаем данные КОРРЕКТНЫМИ селекторами
            title_tag = item.select_one("h2.rubric_lenta__item_name a")
            # ИЗМЕНЕНО: Селектор для рубрики - ищем первую ссылку в li с классом '--plus'
            rubric_tag = item.select_one("ul.crumbs li.tag_list__item--plus a.tag_list__link")
            # Описание (summary) пропускаем, так как его нет в списке

            headline = title_tag.get_text(strip=True) if title_tag else ""
            rubric_name = rubric_tag.get_text(strip=True) if rubric_tag else ""

            # --- ФИЛЬТРАЦИЯ (по рубрике) ---
            if headline and rubric_name and rubric_name in TARGET_RUBRICS:
                results.append({
                    "date": date_str,
                    "headline": headline,
                    # Добавляем пустую строку для описания, чтобы колонки совпадали
                    "summary": ""
                })
                saved_today += 1
            # --- КОНЕЦ ФИЛЬТРАЦИИ ---
            # Можно добавить else для логирования отфильтрованных статей, если нужно
            # else:
            #     if headline and rubric_name: # Логируем только если рубрика была, но не подошла
            #         logging.debug(f"[{date_str}] Статья отфильтрована (рубрика '{rubric_name}'): {headline[:50]}...")

        filtered_out_today = total_found_today - saved_today
        logging.info(f"[{date_str}] Сохранено после фильтрации (рубрики {', '.join(TARGET_RUBRICS)}): {saved_today}. Отфильтровано: {filtered_out_today}.")

        return results

    except requests.exceptions.Timeout:
        logging.error(f"[{date_str}] Ошибка: Запрос HTML превысил таймаут ({REQUEST_TIMEOUT_SECONDS} сек) для URL {url}.")
        return None
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 'N/A'
        logging.error(f"[{date_str}] Ошибка сети или HTTP (статус {status_code}) при запросе HTML {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"[{date_str}] Непредвиденная ошибка при обработке HTML {url}: {e}", exc_info=True)
        return None

# --- Функция scrape_last_n_years остается почти такой же ---
# Убедимся, что колонки CSV включают 'summary'
def scrape_last_n_years(years=YEARS_TO_SCRAPE, batch_size=BATCH_SIZE):
    """
    Главная функция для скрапинга HTML-архива за последние N лет с фильтрацией по рубрикам.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365 + (years // 4))
    delta = timedelta(days=1)

    logging.info(f"--- Начало скрапинга HTML-АРХИВА с фильтрацией (исправленные селекторы) ---")
    logging.info(f"Целевые рубрики: {', '.join(TARGET_RUBRICS)}")
    logging.info(f"Диапазон дат: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
    logging.info(f"Результаты будут сохранены в: {OUTPUT_CSV}")
    logging.info(f"Задержка между запросами: {REQUEST_DELAY_SECONDS} сек.")

    # ИЗМЕНЕНО: Убедимся, что колонка summary есть в CSV, даже если она пустая
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
            day_str = current_date.strftime("%Y-%m-%d")
            processed_days += 1
            progress = (processed_days / total_days) * 100 if total_days > 0 else 0
            # Логируем прогресс реже для чистоты вывода
            if processed_days % 50 == 1 or current_date == end_date or processed_days <= 3:
                logging.info(f"Обработка даты: {day_str} ({processed_days}/{total_days}, {progress:.1f}%)")

            day_articles_filtered = fetch_articles_by_date_html(day_str, session)

            if day_articles_filtered is not None:
                if day_articles_filtered:
                    buffer.extend(day_articles_filtered)
            else:
                logging.warning(f"[{day_str}] Пропуск дня из-за ошибки запроса/обработки HTML.")

            # Сброс в файл
            if len(buffer) >= batch_size or (current_date >= end_date and buffer):
                try:
                    # Создаем DataFrame ПЕРЕД записью, чтобы убедиться, что структура верна
                    df = pd.DataFrame(buffer, columns=csv_columns) # Указываем колонки явно
                    df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding='utf-8-sig')
                    saved_count = len(buffer)
                    total_saved += saved_count
                    logging.info(f"→ Записано в CSV: {saved_count} статей (всего сохранено {total_saved})")
                    buffer.clear()
                except Exception as e:
                     logging.error(f"Не удалось записать пакет в {OUTPUT_CSV}: {e}")
                     # Важно: возможно, стоит очистить буфер или остановить скрипт при ошибке записи
                     # buffer.clear() # Очищаем, чтобы не пытаться записать те же данные снова

            current_date += delta
            if current_date <= end_date:
                 time.sleep(REQUEST_DELAY_SECONDS) # Пауза

    logging.info(f"--- Скрапинг HTML-архива завершён ---")
    logging.info(f"Всего сохранено статей (с рубриками '{', '.join(TARGET_RUBRICS)}'): {total_saved}")
    logging.info(f"Данные сохранены в файле: {OUTPUT_CSV}")


if __name__ == "__main__":
    # Убедись, что установлены: pip install requests pandas beautifulsoup4 lxml
    try:
        scrape_last_n_years(years=YEARS_TO_SCRAPE)
    except KeyboardInterrupt:
        logging.info("Процесс прерван пользователем.")
    except Exception as e:
        logging.error(f"Критическая ошибка в основном процессе: {e}", exc_info=True)