# main_rbc.py (РЕЖИМ РАЗВЕДКИ РУБРИК)

from scrape_rbc_articles import Scraper
import pandas as pd
import traceback
from settings import url # TARGET_RUBRICS пока не импортируем

# ИЗМЕНЕНО: Имя файла для ВСЕХ статей
OUTPUT_FILENAME_ALL = 'rbc_ALL_articles_temp.csv'

print("--- RBC Article Scraper (РЕЖИМ РАЗВЕДКИ РУБРИК) ---")
print(f"Target URL (check date range and keywords): {url}")
# print(f"Target Rubrics: {', '.join(TARGET_RUBRICS)}") # Фильтрация отключена
print(f"Output file (ALL articles): {OUTPUT_FILENAME_ALL}")
print("Цель: собрать все статьи и посмотреть уникальные значения в колонке 'category'.")

try:
    scraper = Scraper()

    # 1. Scrape all links from the search results page using Selenium
    df_links = scraper.scrape_links(url)

    if not df_links.empty:
        # 2. Scrape content (date, title, desc, category) for each link
        df_articles_all = scraper.scrape_all(df_links) # Собираем все данные

        if not df_articles_all.empty:
            print(f"\nВсего статей собрано: {len(df_articles_all)}")

            # 3. СОХРАНЯЕМ ВСЕ СОБРАННЫЕ ДАННЫЕ (БЕЗ ФИЛЬТРАЦИИ)
            try:
                df_articles_all.to_csv(OUTPUT_FILENAME_ALL, index=False, encoding='utf-8-sig')
                print(f"ВСЕ собранные статьи сохранены в {OUTPUT_FILENAME_ALL}")
            except Exception as e_save:
                print(f"Ошибка при сохранении файла {OUTPUT_FILENAME_ALL}: {e_save}")

            # 4. АНАЛИЗ УНИКАЛЬНЫХ ЗНАЧЕНИЙ В КОЛОНКЕ 'category'
            if 'category' in df_articles_all.columns:
                # Заменяем NaN на строку 'Нет категории' для наглядности
                unique_categories = df_articles_all['category'].fillna('Нет категории').unique()
                print(f"\n--- Уникальные значения, найденные в колонке 'category' ({len(unique_categories)}) ---")
                # Сортируем для удобства просмотра
                for category in sorted(list(unique_categories)):
                    print(f"- {category}")
                print("--- Конец списка уникальных категорий ---")
                print("\nПроанализируй этот список и файл CSV. Какие значения соответствуют 'Бизнес' и 'Политика'?")
            else:
                print("\nКолонка 'category' не найдена в собранных данных.")

        else:
            print("Сбор контента статей не дал результатов (DataFrame пуст).")
    else:
        print("Сбор ссылок не дал результатов (DataFrame пуст).")

except Exception as e:
    print(f"\n--- Произошла ошибка ---")
    print(e)
    print(traceback.format_exc())

print("\n--- Скрипт (в режиме разведки) завершил работу ---")