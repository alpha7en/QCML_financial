import pandas as pd
import numpy as np
import logging
import os

# --- Конфигурация ---
# Пути к входным файлам
NLP_FEATURES_FILE = "final_qcml_nlp_features_v6_debugged.csv"
ECONOMIC_DATA_FILE = "moex_metrics_final_v4.csv"
# Имя выходного файла
OUTPUT_COMBINED_FILE = "final_dataset.csv"

# Имена колонок даты в исходных файлах
NLP_DATE_COL = 'date'
ECON_DATE_COL = 'TRADEDATE'

# Ожидаемые колонки из NLP файла (кроме даты)
EXPECTED_NLP_COLS = [
    'news_sentiment_accel_1d_z21',
    'news_topic_entropy_daily_z21', # Измененное имя из предыдущего шага
    'news_sentiment_extreme_1d_z21',
    'news_topic_shock_1d_z21',
    'news_topic_intensity_MacroRF_1d_z21',
    'news_topic_intensity_MonPolicy_1d_z21',
    'news_topic_intensity_Geopol_1d_z21',
    'news_sentiment_level_21d_z63',
    'news_topic_entropy_level_21d_z63',
    'news_topic_intensity_OilGas_21d_z63',
    'news_topic_intensity_Fiscal_21d_z63',
    'news_sentiment_level_63d_z1y'
]

# Ожидаемые колонки из Economic файла (кроме даты)
EXPECTED_ECON_COLS = [
    'Z_IMO_return',
    'Z_IMO_vol_real21',
    'Rel_IMO_RTSOG',
    'Z_USD_RUB_trend21',
    'Z_ADX_IMOEX',
    'Z_IMO_trend21',
    'Z_RTSOG_trend63',
    'Z_IMO_trend63',
    'Z_RGBITR_trend63',
    'Z_RUCBTRNS_trend63',
    'IMO_fwd_log_ret_3d'
]

# --- Настройка Логирования ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler_file = logging.FileHandler(f"{OUTPUT_COMBINED_FILE}.log", mode='w')
log_handler_file.setFormatter(log_formatter)
log_handler_stream = logging.StreamHandler()
log_handler_stream.setFormatter(log_formatter)

logger = logging.getLogger()
if logger.hasHandlers(): logger.handlers.clear()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler_file)
logger.addHandler(log_handler_stream)

# --- Основной блок ---
if __name__ == "__main__":
    logging.info("Запуск скрипта объединения NLP и экономических данных...")

    # 1. Загрузка NLP данных
    logging.info(f"Загрузка NLP данных из: {NLP_FEATURES_FILE}")
    if not os.path.exists(NLP_FEATURES_FILE):
        logging.error(f"Файл не найден: {NLP_FEATURES_FILE}"); exit(1)
    try:
        df_nlp = pd.read_csv(NLP_FEATURES_FILE, low_memory=False)
        logging.info(f"Загружено {len(df_nlp)} строк.")
        if df_nlp.empty: logging.warning("NLP файл пуст."); # Не выходим, может быть полезно для outer join
    except Exception as e:
        logging.exception(f"Ошибка при загрузке NLP файла: {e}"); exit(1)

    # 2. Загрузка Economic данных
    logging.info(f"Загрузка экономических данных из: {ECONOMIC_DATA_FILE}")
    if not os.path.exists(ECONOMIC_DATA_FILE):
        logging.error(f"Файл не найден: {ECONOMIC_DATA_FILE}"); exit(1)
    try:
        df_econ = pd.read_csv(ECONOMIC_DATA_FILE, low_memory=False)
        logging.info(f"Загружено {len(df_econ)} строк.")
        if df_econ.empty: logging.warning("Economic файл пуст.");
    except Exception as e:
        logging.exception(f"Ошибка при загрузке Economic файла: {e}"); exit(1)

    # 3. Подготовка Дат и проверка колонок
    logging.info("Подготовка дат и проверка колонок...")

    # NLP DataFrame
    if NLP_DATE_COL not in df_nlp.columns:
        logging.error(f"Колонка даты '{NLP_DATE_COL}' не найдена в NLP файле."); exit(1)
    try:
        df_nlp[NLP_DATE_COL] = pd.to_datetime(df_nlp[NLP_DATE_COL], errors='coerce')
        nlp_rows_before = len(df_nlp)
        df_nlp.dropna(subset=[NLP_DATE_COL], inplace=True)
        if len(df_nlp) < nlp_rows_before: logging.warning(f"Удалено {nlp_rows_before - len(df_nlp)} строк с некорректными датами из NLP файла.")
    except Exception as e:
        logging.exception(f"Ошибка обработки дат в NLP файле: {e}"); exit(1)
    # Проверка NLP колонок
    missing_nlp_cols = [col for col in EXPECTED_NLP_COLS if col not in df_nlp.columns]
    if missing_nlp_cols: logging.warning(f"В NLP файле отсутствуют ожидаемые колонки: {missing_nlp_cols}")

    # Economic DataFrame
    if ECON_DATE_COL not in df_econ.columns:
        logging.error(f"Колонка даты '{ECON_DATE_COL}' не найдена в Economic файле."); exit(1)
    try:
        df_econ[ECON_DATE_COL] = pd.to_datetime(df_econ[ECON_DATE_COL], errors='coerce')
        econ_rows_before = len(df_econ)
        df_econ.dropna(subset=[ECON_DATE_COL], inplace=True)
        if len(df_econ) < econ_rows_before: logging.warning(f"Удалено {econ_rows_before - len(df_econ)} строк с некорректными датами из Economic файла.")
    except Exception as e:
        logging.exception(f"Ошибка обработки дат в Economic файле: {e}"); exit(1)
    # Проверка Economic колонок
    missing_econ_cols = [col for col in EXPECTED_ECON_COLS if col not in df_econ.columns]
    if missing_econ_cols: logging.warning(f"В Economic файле отсутствуют ожидаемые колонки: {missing_econ_cols}")

    # Переименование колонки даты в Economic файле для слияния
    df_econ.rename(columns={ECON_DATE_COL: NLP_DATE_COL}, inplace=True)
    logging.info(f"Колонка '{ECON_DATE_COL}' переименована в '{NLP_DATE_COL}' в Economic данных.")

    # Проверка на пустые DataFrame после обработки дат
    if df_nlp.empty and df_econ.empty:
        logging.error("Оба DataFrame пусты после обработки дат. Нечего объединять."); exit(1)
    elif df_nlp.empty:
        logging.warning("NLP DataFrame пуст после обработки дат.")
    elif df_econ.empty:
        logging.warning("Economic DataFrame пуст после обработки дат.")


    # 4. Слияние данных
    logging.info("Слияние NLP и Economic данных по дате...")
    # Используем inner join, чтобы оставить только даты, присутствующие в обоих файлах
    df_merged = pd.merge(df_nlp, df_econ, on=NLP_DATE_COL, how='inner')
    logging.info(f"Слияние завершено. Получено {len(df_merged)} строк с общими датами.")

    if df_merged.empty:
        logging.error("Результат слияния пуст. Нет общих дат в исходных файлах.")
        exit(1)

    # 5. Формирование итогового DataFrame
    # Создаем финальный порядок колонок: дата + NLP + Economic
    final_column_order = [NLP_DATE_COL] + EXPECTED_NLP_COLS + EXPECTED_ECON_COLS

    # Проверяем, все ли нужные колонки есть в df_merged
    actual_merged_cols = df_merged.columns.tolist()
    missing_final_cols = [col for col in final_column_order if col not in actual_merged_cols]
    if missing_final_cols:
        logging.error(f"После слияния отсутствуют необходимые колонки: {missing_final_cols}")
        # Добавляем недостающие колонки с NaN, чтобы сохранить файл
        for col in missing_final_cols: df_merged[col] = np.nan
        logging.warning(f"Недостающие колонки добавлены со значениями NaN.")
        # Используем исходный список final_column_order, т.к. добавили недостающие
        final_cols_to_save = final_column_order
    else:
        # Выбираем и переупорядочиваем колонки
        final_cols_to_save = final_column_order

    df_final = df_merged[final_cols_to_save].copy()
    logging.info(f"Финальный DataFrame содержит {len(df_final.columns)} колонок.")

    # Сортировка по дате на всякий случай
    df_final.sort_values(NLP_DATE_COL, inplace=True)

    # 6. Сохранение результата
    logging.info(f"Сохранение итогового файла в {OUTPUT_COMBINED_FILE}...")
    try:
        # Форматируем дату перед сохранением
        df_final[NLP_DATE_COL] = pd.to_datetime(df_final[NLP_DATE_COL]).dt.strftime('%Y-%m-%d')
        df_final.to_csv(OUTPUT_COMBINED_FILE, index=False, encoding='utf-8', float_format='%.6f') # Используем 6 знаков после запятой
        logging.info("Файл успешно сохранен.")
    except Exception as e:
        logging.exception(f"Ошибка при сохранении итогового файла: {e}")

    logging.info("Скрипт завершил выполнение.")