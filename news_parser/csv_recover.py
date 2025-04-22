import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import time
import logging
import os
import sys

# --- Настройки ---
OUTPUT_CSV = "lenta_economics_last_5_years.csv" # Имя файла
TARGET_CATEGORY = "economics" # Целевая категория Lenta.ru
YEARS_TO_SCRAPE = 5
REQUEST_DELAY_SECONDS = 1.5 # Задержка между запросами (к разным страницам/датам)
REQUEST_TIMEOUT_SECONDS = 40
# Убрали BATCH_SIZE, т.к. запись будет в конце
TEMP_CSV_SUFFIX = ".new_data.tmp" # Суффикс для временного файла с новыми данными

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Заголовки для HTML запросов
HTML_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36 (LentaScraper)"
}

# --- Функция fetch_lenta_articles_for_day (без изменений) ---
def fetch_lenta_articles_for_day(date_obj, category, session):
    """
    Получает и парсит HTML-страницы архива Lenta.ru за указанную дату
    для указанной категории, обрабатывая пагинацию.
    Возвращает список словарей {'date': date_str, 'headline': headline, 'summary': ''}.
    """
    articles_for_day = []
    page = 1
    date_url_part = date_obj.strftime("%Y/%m/%d")
    date_str_iso = date_obj.strftime("%Y-%m-%d")

    while True:
        url = f"https://lenta.ru/rubrics/{category}/{date_url_part}/page/{page}/"
        logging.debug(f"-----> [ЗАПРОС HTML] Дата: {date_str_iso}, Страница: {page}, URL: {url}") # Уровень debug для менее важных логов

        try:
            resp = session.get(url, headers=HTML_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            logging.debug(f"<----- [ОТВЕТ HTML] Дата: {date_str_iso}, Страница: {page}, Статус: {resp.status_code}")

            if resp.status_code == 404:
                 logging.info(f"[{date_str_iso}] Страница {page} не найдена (404). Завершение для этой даты.")
                 break

            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'lxml')
            article_elements = soup.select("li.archive-page__item._news")
            logging.debug(f"[{date_str_iso}] На странице {page} найдено {len(article_elements)} новостей.")

            if not article_elements and page == 1:
                 logging.info(f"[{date_str_iso}] Новостей в разделе '{category}' за эту дату нет.")
                 break
            if not article_elements: # На page > 1 нет статей -> конец пагинации
                logging.debug(f"[{date_str_iso}] На странице {page} статьи не найдены, пагинация завершена.")
                break

            for item in article_elements:
                title_tag = item.select_one("h3.card-full-news__title")
                headline = title_tag.get_text(strip=True) if title_tag else ""
                summary = "" # Поле summary

                if headline:
                    articles_for_day.append({
                        "date": date_str_iso,
                        "headline": headline,
                        "summary": summary
                    })
                else:
                    logging.warning(f"[{date_str_iso} стр. {page}] Найден элемент без заголовка.")

            next_page_link = soup.select_one(f'a.loadmore[href$="/page/{page+1}/"]')
            if next_page_link:
                logging.debug(f"[{date_str_iso}] Найдена ссылка на следующую страницу ({page+1}).")
                page += 1
                time.sleep(REQUEST_DELAY_SECONDS)
            else:
                logging.debug(f"[{date_str_iso}] Ссылка на следующую страницу не найдена. Завершение для этой даты.")
                break

        except requests.exceptions.Timeout:
            logging.error(f"[{date_str_iso} стр. {page}] Ошибка: Запрос HTML превысил таймаут ({REQUEST_TIMEOUT_SECONDS} сек) для URL {url}.")
            return None
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else 'N/A'
            logging.error(f"[{date_str_iso} стр. {page}] Ошибка сети или HTTP (статус {status_code}) при запросе HTML {url}: {e}")
            return None
        except Exception as e:
            logging.error(f"[{date_str_iso} стр. {page}] Непредвиденная ошибка при обработке HTML {url}: {e}", exc_info=True)
            return None

    return articles_for_day

# --- Функция get_dates_to_scrape (без изменений) ---
def get_dates_to_scrape(output_file, start_date, end_date):
    """
    Определяет даты, которые нужно спарсить.
    Сравнивает полный диапазон дат с датами, уже имеющимися в CSV-файле.
    Возвращает отсортированный список объектов datetime для пропущенных дат.
    """
    all_dates_in_range = set()
    current_d = start_date
    while current_d <= end_date:
        all_dates_in_range.add(current_d.strftime("%Y-%m-%d"))
        current_d += timedelta(days=1)

    existing_dates = set()
    file_exists = os.path.exists(output_file)
    if file_exists:
        try:
            logging.info(f"Чтение существующих дат из {output_file}...")
            # Читаем только колонку 'date' и обрабатываем возможные ошибки
            df_existing = pd.read_csv(
                output_file,
                usecols=['date'],
                on_bad_lines='warn', # Предупреждать о плохих строках
                encoding='utf-8-sig' # Убедимся, что читаем в правильной кодировке
            )
            # Конвертируем в строки и убираем некорректные значения перед добавлением в set
            existing_dates = set(df_existing['date'].astype(str).dropna().unique())
            logging.info(f"Обнаружено {len(existing_dates)} уникальных дат в существующем файле.")
        except pd.errors.EmptyDataError:
            logging.warning(f"Файл {output_file} пуст. Будут скачаны все даты.")
            file_exists = False
        except FileNotFoundError:
             logging.info(f"Файл {output_file} не найден. Будут скачаны все даты.")
             file_exists = False
        except KeyError:
            logging.error(f"В файле {output_file} отсутствует колонка 'date'. Невозможно определить пропущенные даты.")
            # Можно было бы выйти sys.exit(1), но попробуем продолжить, скачав все
            existing_dates = set()
            file_exists = False # Считаем, что файла нет для определения дат
        except ValueError as e:
             logging.error(f"Ошибка при обработке дат в {output_file}: {e}. Возможно, некорректный формат.")
             existing_dates = set()
             file_exists = False
        except Exception as e:
            logging.error(f"Ошибка при чтении файла {output_file}: {e}. Попытка скачать все даты.")
            existing_dates = set()
            file_exists = False

    # Определяем даты, которые нужно докачать
    missing_dates_str = sorted(list(all_dates_in_range - existing_dates))
    missing_dates_obj = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in missing_dates_str]

    if file_exists:
        if missing_dates_obj:
            logging.info(f"Найдено {len(missing_dates_obj)} пропущенных дат для скачивания.")
        else:
            logging.info("Пропущенных дат не найдено. Все данные в диапазоне уже присутствуют.")
    else:
        logging.info(f"Будут скачаны данные за {len(missing_dates_obj)} дат в указанном диапазоне.")

    return missing_dates_obj, file_exists

# --- МОДИФИЦИРОВАННАЯ функция основного цикла ---
def scrape_and_merge_lenta(category=TARGET_CATEGORY, years=YEARS_TO_SCRAPE):
    """
    Главная функция для скрапинга/дополнения архива Lenta.ru.
    Скачивает данные за пропущенные даты и затем объединяет их
    с существующими данными, сортирует и перезаписывает файл.
    """
    end_date = datetime.today()
    end_date = datetime(end_date.year, end_date.month, end_date.day) # Убираем время
    start_date = end_date - timedelta(days=years * 365 + (years // 4))

    logging.info(f"--- Начало скрапинга/дополнения Lenta.ru (категория: {category}) ---")
    logging.info(f"Целевой диапазон дат: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
    logging.info(f"Основной файл: {OUTPUT_CSV}")

    dates_to_process, file_already_exists = get_dates_to_scrape(OUTPUT_CSV, start_date, end_date)

    if not dates_to_process:
        logging.info("--- Завершение: Нет дат для обработки. ---")
        return

    csv_columns = ["date", "headline", "summary"]
    newly_scraped_articles = [] # Список для хранения всех новых статей
    total_days_to_process = len(dates_to_process)
    processed_days_count = 0

    logging.info(f"--- Начало сбора данных для {total_days_to_process} пропущенных дат ---")

    with requests.Session() as session:
        for current_date in dates_to_process:
            processed_days_count += 1
            progress = (processed_days_count / total_days_to_process) * 100 if total_days_to_process > 0 else 0
            logging.info(f"Сбор данных: {current_date.strftime('%Y-%m-%d')} ({processed_days_count}/{total_days_to_process}, {progress:.1f}%)")

            day_articles = fetch_lenta_articles_for_day(current_date, category, session)

            if day_articles is not None:
                if day_articles:
                    newly_scraped_articles.extend(day_articles)
                    logging.info(f"[{current_date.strftime('%Y-%m-%d')}] Собрано {len(day_articles)} статей.")
            else:
                logging.warning(f"[{current_date.strftime('%Y-%m-%d')}] Пропуск дня из-за ошибки.")

            if processed_days_count < total_days_to_process:
                 time.sleep(REQUEST_DELAY_SECONDS * 1.1) # Пауза между днями

    logging.info(f"--- Сбор данных для пропущенных дат завершен. Собрано {len(newly_scraped_articles)} новых статей. ---")

    # --- Этап объединения, сортировки и перезаписи ---
    if not newly_scraped_articles and not file_already_exists:
         logging.warning("Не было собрано новых статей и исходный файл не существует. Результат будет пустым.")
         # Создаем пустой файл с заголовками, если нужно
         try:
             if not os.path.exists(OUTPUT_CSV):
                 pd.DataFrame(columns=csv_columns).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
                 logging.info(f"Создан пустой файл {OUTPUT_CSV} с заголовками.")
         except Exception as e:
             logging.error(f"Не удалось создать пустой файл {OUTPUT_CSV}: {e}")
         return
    elif not newly_scraped_articles:
        logging.info("Новых статей для добавления нет. Файл остается без изменений.")
        return

    logging.info("Начало объединения и сортировки данных...")

    try:
        df_new = pd.DataFrame(newly_scraped_articles, columns=csv_columns)
        df_combined = None

        if file_already_exists:
            logging.info(f"Чтение существующего файла {OUTPUT_CSV} для объединения...")
            try:
                df_existing = pd.read_csv(OUTPUT_CSV, dtype=str, encoding='utf-8-sig') # Читаем все как строки сначала
                logging.info(f"Прочитано {len(df_existing)} строк из существующего файла.")
                # Проверка на наличие необходимых колонок в существующем файле
                if not all(col in df_existing.columns for col in csv_columns):
                    logging.warning(f"В существующем файле {OUTPUT_CSV} отсутствуют некоторые колонки. Будут использованы колонки из новых данных.")
                    # При конкатенации Pandas сам добавит NaN где нужно, но лучше убедиться что колонки есть
                    for col in csv_columns:
                        if col not in df_existing.columns:
                           df_existing[col] = pd.NA # или ''

                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                logging.info(f"Данные объединены. Всего строк до обработки: {len(df_combined)}")
            except pd.errors.EmptyDataError:
                logging.warning(f"Существующий файл {OUTPUT_CSV} пуст. Будут сохранены только новые данные.")
                df_combined = df_new
            except Exception as e:
                logging.error(f"Ошибка при чтении существующего файла {OUTPUT_CSV}: {e}. Попытка сохранить только новые данные.")
                df_combined = df_new # В случае ошибки чтения старого, сохраняем хотя бы новое
        else:
            logging.info("Исходный файл не найден. Будут сохранены только новые данные.")
            df_combined = df_new

        # --- Очистка и сортировка ---
        logging.info("Преобразование дат и удаление дубликатов...")

        # Убедимся, что работаем с правильными типами данных перед сортировкой/удалением дублей
        df_combined['date'] = pd.to_datetime(df_combined['date'], errors='coerce') # errors='coerce' заменит невалидные даты на NaT
        df_combined.dropna(subset=['date'], inplace=True) # Удаляем строки с невалидными датами

        # Заполняем NaN в текстовых колонках пустыми строками перед удалением дубликатов
        df_combined['headline'] = df_combined['headline'].fillna('')
        df_combined['summary'] = df_combined['summary'].fillna('')

        # Удаляем полные дубликаты строк
        initial_rows = len(df_combined)
        df_combined.drop_duplicates(subset=['date', 'headline'], keep='first', inplace=True) # Удаляем дубли по дате и заголовку
        removed_count = initial_rows - len(df_combined)
        if removed_count > 0:
            logging.info(f"Удалено {removed_count} дублирующихся строк.")

        logging.info("Сортировка данных по дате...")
        # Сортируем по дате (сначала новые), затем по заголовку для стабильности
        df_combined.sort_values(by=['date', 'headline'], ascending=[True, True], inplace=True)

        # Преобразуем дату обратно в строку перед сохранением
        df_combined['date'] = df_combined['date'].dt.strftime('%Y-%m-%d')

        # --- Перезапись файла ---
        logging.info(f"Перезапись файла {OUTPUT_CSV} отсортированными данными ({len(df_combined)} строк)...")
        df_combined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig', mode='w') # mode='w' для перезаписи
        logging.info(f"Файл {OUTPUT_CSV} успешно обновлен.")

    except Exception as e:
        logging.error(f"Критическая ошибка на этапе объединения/сортировки/записи: {e}", exc_info=True)
        logging.error("Файл мог остаться в неконсистентном состоянии или не обновиться.")

    logging.info(f"--- Процесс завершён. ---")


if __name__ == "__main__":
    # Убедись, что установлены: pip install requests pandas beautifulsoup4 lxml
    try:
        scrape_and_merge_lenta(category=TARGET_CATEGORY, years=YEARS_TO_SCRAPE)
    except KeyboardInterrupt:
        logging.info("Процесс прерван пользователем.")
    except Exception as e:
        logging.error(f"Критическая ошибка в основном процессе: {e}", exc_info=True)