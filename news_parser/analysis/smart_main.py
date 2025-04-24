from yandex_cloud_ml_sdk import YCloudML
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import time
import numpy as np
import os # Добавлено для проверки существования файла

RATE_LIMIT_DELAY = 0.1  # 200 мс
OUTPUT_FILENAME = "raw_features.csv"
INPUT_FILENAME = "kommersant_business_politics_archive_last_5_years.csv"
SAVE_INTERVAL = 10 # Сохранять каждые N дней

# --- Инициализация клиента Yandex Cloud ---
try:
    # Замените на ваши параметры или используйте переменные окружения/метаданные ВМ
    sdk = YCloudML(
        folder_id="b1gu0dd26bpgkokh9fsk",
        auth="AQVNyz-waLh_zQFPyVCl-LhOUQdwCMNc2bJtfCud"
    )
    classifier = (
        sdk.models.text_classifiers("yandexgpt-lite")
    )
    print("Yandex Cloud ML SDK инициализирован успешно.")
except Exception as e:
    print(f"Ошибка инициализации Yandex Cloud ML SDK: {e}")
    print("Пожалуйста, проверьте ваши folder_id и параметры аутентификации.")
    exit() # Выход, если SDK не инициализирован

# --- Функции для классификации ---
def classify_text(text: str, task_description: str, labels: list[str]) -> dict[str, float]:
    """
    Отправляем один Zero-shot запрос к Yandex Cloud, затем ждём RATE_LIMIT_DELAY.
    Возвращаем: {label: confidence}.
    """
    resp = classifier.configure(
        task_description=task_description,
        labels=labels
    )
    # небольшая задержка, чтобы не перегружать API
    time.sleep(RATE_LIMIT_DELAY)
    # print(f"Classifying: {text[:50]}...") # Отладочный вывод (можно закомментировать)
    result = resp.run(text)
    return {pred.label: pred.confidence for pred in result}

# --- Описания задач и метки ---
sentiment_task = (
    "Определите тональность новости в диапазоне "
    "двух классов: positive, neutral и negative."
)
topic_task = (
    "Определите интенсивность следующих тем в тексте новости:\n"
    "MacroRF: ВВП, инфляция, безработица, промпроизводство, "
    "розничные продажи, Росстат, Минэкономразвития, экономический рост/спад.\n"
    "MonetaryPolicyRF: ЦБ РФ, ключевая ставка, денежно-кредитная политика, "
    "инфляционные ожидания, валютные интервенции, валютные интервенции, банковское регулирование.\n"
    "GeopoliticsSanctions: санкции, внешняя политика, международные отношения, "
    "СВО и ее экономические последствия, торговые войны.\n"
    "OilGasEnergy: нефть Brent, Urals, газ, ОПЕК+, Газпром, Роснефть, энергетический рынок, экспорт и добыча нефти/газа в РФ, Новатэк\n"
    "FiscalPolicyCorp: Бюджетная политика и Корпорации РФ: Бюджет России, Минфин, налоги (НДС, НДПИ, налог на прибыль), госрасходы, госдолг, крупные госкомпании"
)
sentiment_labels = ["positive", "neutral", "negative"]
topic_labels     = [
    "MacroRF", "MonetaryPolicyRF", "GeopoliticsSanctions",
    "OilGasEnergy", "FiscalPolicyCorp"
]

def analyze_article(text: str) -> tuple[float, dict[str,float]]:
    """Анализирует одну статью на тональность и темы."""
    # Классификация тональности
    sent_conf = classify_text(text, sentiment_task, sentiment_labels)
    score = sent_conf.get("positive", 0.0) - sent_conf.get("negative", 0.0) # Используем .get для надежности

    # Классификация тем
    topic_conf = classify_text(text, topic_task, topic_labels)
    # Убедимся, что все метки присутствуют в результате, даже если с нулевой уверенностью
    full_topic_conf = {label: topic_conf.get(label, 0.0) for label in topic_labels}

    return score, full_topic_conf

# --- Новая функция для обработки и инкрементного сохранения ---
def compute_and_save_daily_features_incrementally(
    df: pd.DataFrame,
    output_filename: str,
    save_interval: int,
    write_header: bool
):
    """
    Обрабатывает DataFrame по дням, рассчитывает признаки и сохраняет результаты
    инкрементно каждые 'save_interval' дней.

    Args:
        df (pd.DataFrame): Отфильтрованный и отсортированный DataFrame с данными для обработки.
                           Ожидаемые колонки: 'date', 'title', 'content'.
        output_filename (str): Путь к файлу для сохранения результатов.
        save_interval (int): Количество дней для обработки перед сохранением батча.
        write_header (bool): True, если нужно записать заголовок в CSV (при первой записи).
    """
    daily_batch = [] # Список для накопления результатов текущего батча
    days_processed_since_last_save = 0
    is_first_write = write_header # Флаг для контроля записи заголовка

    # Убедимся, что колонка 'date' имеет правильный тип для группировки
    # Если даты уже строки 'YYYY-MM-DD', можно пропустить. Если объекты datetime, тоже ок.
    # df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d') # Пример конвертации в строки

    grouped = df.groupby("date")
    total_days_to_process = len(grouped)
    processed_day_count = 0

    print(f"Начинаем обработку {total_days_to_process} новых дней...")

    for date, group in grouped:
        processed_day_count += 1
        print(f"Обработка даты: {date} ({processed_day_count}/{total_days_to_process})")
        scores = []
        topic_vecs = []
        extreme_count = 0
        article_count_in_day = len(group)
        processed_articles_in_day = 0

        for index, row in group.iterrows():
            try:
                # Используем 'title' как в оригинальном коде.
                # Если нужен 'content' или комбинация, измените здесь:
                text_to_analyze = str(row.get("title", "")) # Используем .get и str() для надежности
                if not text_to_analyze:
                     print(f"  Предупреждение: Пустой 'title' для строки {index} в дате {date}. Пропуск строки.")
                     continue

                score, topics = analyze_article(text_to_analyze)
                scores.append(score)
                # Убедимся, что порядок тем соответствует topic_labels
                topic_vecs.append([topics[label] for label in topic_labels])
                if abs(score) > 0.75:
                    extreme_count += 1
                processed_articles_in_day += 1

            except Exception as e:
                print(f"  Ошибка при обработке строки {index} для даты {date}: {e}. Пропуск строки.")
                # Можно добавить более детальное логирование ошибки при необходимости
                # import traceback
                # print(traceback.format_exc())
                time.sleep(RATE_LIMIT_DELAY * 2) # Дополнительная пауза после ошибки API

        # Рассчитываем дневные признаки только если были успешно обработанные статьи
        if processed_articles_in_day > 0:
            avg_sent = sum(scores) / processed_articles_in_day
            # np.mean требует хотя бы одного вектора
            avg_vec = np.mean(topic_vecs, axis=0).tolist() if topic_vecs else [0.0] * len(topic_labels)

            # Энтропия требует вероятностного распределения (сумма=1, все >= 0)
            # Преобразуем интенсивности в подобие вероятностей (если нужно строго)
            # Здесь просто считаем энтропию от интенсивностей как есть
            # Добавим небольшое значение, чтобы избежать log(0)
            probs = np.array(avg_vec) + 1e-9
            probs /= probs.sum() # Нормализуем
            ent = entropy(probs, base=2)

            extreme_rate = extreme_count / processed_articles_in_day

            daily_batch.append({
                "date": date,
                "news_sentiment_1d_raw": avg_sent,
                "news_topic_intensities_1d_raw": avg_vec, # Сохраняем как список
                "news_topic_entropy_1d_raw": ent,
                "news_sentiment_extreme_1d_raw": extreme_rate
            })
            days_processed_since_last_save += 1
        else:
            print(f"  Предупреждение: Не удалось обработать ни одной статьи за дату {date}. Пропуск дня.")

        # Проверяем, не пора ли сохранить батч
        if days_processed_since_last_save >= save_interval:
            if daily_batch: # Сохраняем только если есть что сохранять
                print(f"--> Сохранение батча из {len(daily_batch)} дней...")
                batch_df = pd.DataFrame(daily_batch)
                # Преобразуем список векторов в строку для CSV (или используйте JSON/другой формат)
                batch_df['news_topic_intensities_1d_raw'] = batch_df['news_topic_intensities_1d_raw'].astype(str)

                batch_df.to_csv(
                    output_filename,
                    mode='a',       # 'a' - append (добавить в конец)
                    index=False,
                    sep=",",
                    encoding="utf-8",
                    header=is_first_write # Записать заголовок только при первой записи
                )
                print(f"--> Батч сохранен в {output_filename}")
                is_first_write = False # Заголовок больше не пишем
                daily_batch = []       # Очищаем батч
                days_processed_since_last_save = 0 # Сбрасываем счетчик
            else:
                 # Если батч пуст (например, все дни в интервале были пропущены), просто сбрасываем счетчик
                 days_processed_since_last_save = 0


    # Сохраняем оставшиеся данные после завершения цикла
    if daily_batch:
        print(f"--> Сохранение финального батча из {len(daily_batch)} дней...")
        batch_df = pd.DataFrame(daily_batch)
        batch_df['news_topic_intensities_1d_raw'] = batch_df['news_topic_intensities_1d_raw'].astype(str)
        batch_df.to_csv(
            output_filename,
            mode='a',
            index=False,
            sep=",",
            encoding="utf-8",
            header=is_first_write # Записать заголовок, если это была самая первая запись вообще
        )
        print(f"--> Финальный батч сохранен в {output_filename}")

    print("Обработка завершена.")


# --- Основной блок выполнения ---
if __name__ == "__main__":

    processed_dates = set()
    write_header = True

    # 1. Проверка и чтение существующего файла результатов
    if os.path.exists(OUTPUT_FILENAME):
        print(f"Найден файл результатов '{OUTPUT_FILENAME}'. Чтение обработанных дат...")
        try:
            # Читаем только колонку 'date', чтобы сэкономить память
            df_processed = pd.read_csv(OUTPUT_FILENAME, usecols=['date'], low_memory=False)
            # Убедимся, что даты читаются как строки для согласованности
            processed_dates = set(df_processed['date'].astype(str).unique())
            print(f"Найдено {len(processed_dates)} уникальных обработанных дат.")
            if not df_processed.empty:
                write_header = False # Файл не пустой, заголовок не нужен
            else:
                print("Файл результатов пуст. Заголовок будет записан.")
        except pd.errors.EmptyDataError:
            print("Файл результатов пуст. Заголовок будет записан.")
        except ValueError as e:
             print(f"Предупреждение: Не удалось прочитать колонку 'date' из '{OUTPUT_FILENAME}'. Возможно, файл поврежден или имеет неверный формат. {e}")
             print("Продолжаем без учета предыдущих результатов (может привести к дублированию).")
             # В этой ситуации лучше либо остановить выполнение, либо переименовать старый файл
             # Для продолжения, оставляем processed_dates пустым и write_header = True
        except Exception as e:
            print(f"Ошибка при чтении файла '{OUTPUT_FILENAME}': {e}")
            print("Продолжаем без учета предыдущих результатов (может привести к дублированию).")
            # processed_dates = set() # Уже инициализировано
            # write_header = True # Уже инициализировано


    # 2. Загрузка исходных данных
    print(f"Загрузка исходных данных из '{INPUT_FILENAME}'...")
    try:
        # Укажите нужные колонки и типы данных для экономии памяти
        # Убедимся, что 'date' читается как строка для согласованности сравнения
        df_all = pd.read_csv(INPUT_FILENAME, usecols=['date', 'title', 'content'], dtype={'date': str, 'title': str, 'content': str})
        print(f"Загружено {len(df_all)} строк.")
        # Очистка от пустых дат, если такие есть
        df_all.dropna(subset=['date'], inplace=True)
        df_all = df_all[df_all['date'] != '']
        print(f"Строк после удаления пустых дат: {len(df_all)}")

    except FileNotFoundError:
        print(f"Ошибка: Исходный файл '{INPUT_FILENAME}' не найден.")
        exit()
    except ValueError as e:
         print(f"Ошибка: Не удалось прочитать ожидаемые колонки ('date', 'title', 'content') из '{INPUT_FILENAME}'. Проверьте структуру файла. {e}")
         exit()
    except Exception as e:
        print(f"Неизвестная ошибка при загрузке '{INPUT_FILENAME}': {e}")
        exit()

    # 3. Фильтрация данных: оставляем только необработанные даты
    if processed_dates:
        initial_rows = len(df_all)
        # Фильтруем, используя строковое представление дат
        df_to_process = df_all[~df_all['date'].astype(str).isin(processed_dates)].copy()
        print(f"Отфильтровано {initial_rows - len(df_to_process)} строк (уже обработанные даты).")
        print(f"Осталось для обработки: {len(df_to_process)} строк.")
    else:
        df_to_process = df_all.copy()
        print("Обработанных дат не найдено, обрабатываем все загруженные строки.")

    # 4. Проверка, есть ли что обрабатывать
    if df_to_process.empty:
        print("Нет новых данных для обработки. Завершение работы.")
        exit()

    # 5. Сортировка данных по дате (важно!)
    print("Сортировка данных по дате...")
    try:
        # Попробуем преобразовать в datetime для корректной сортировки, затем вернем в строку если нужно
        df_to_process['date_dt'] = pd.to_datetime(df_to_process['date'])
        df_to_process.sort_values('date_dt', inplace=True)
        # df_to_process.drop(columns=['date_dt'], inplace=True) # Удаляем временную колонку
        # Или оставляем 'date' как datetime, если группировка по datetime работает
        print("Сортировка завершена.")
    except Exception as e:
        print(f"Предупреждение: Не удалось отсортировать по дате как datetime ({e}). Сортировка как строки.")
        df_to_process.sort_values('date', inplace=True)


    # 6. Запуск обработки и сохранения
    compute_and_save_daily_features_incrementally(
        df_to_process,
        OUTPUT_FILENAME,
        SAVE_INTERVAL,
        write_header
    )

    print("Скрипт завершил выполнение.")