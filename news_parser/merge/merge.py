import pandas as pd
import numpy as np
import logging
import os
import ast

# --- Конфигурация ---
# (Оставляем как в v8)
FILE_SOURCE_1 = "raw_features.csv"
FILE_SOURCE_2 = "gemini_daily_headline_features_structured_v2 (1).csv"
OUTPUT_FINAL_FEATURES_FILE = "final_qcml_nlp_features_v10_extreme_z_debug.csv" # Новое имя

BASE_TOPIC_NAMES = [
    "news_topic_intensity_MacroRF_raw", "news_topic_intensity_MonetaryPolicyRF_raw",
    "news_topic_intensity_GeopoliticsSanctions_raw", "news_topic_intensity_OilGasEnergy_raw",
    "news_topic_intensity_FiscalPolicyCorp_raw"
]
TOPIC_KEYS_ORDERED = ["MacroRF", "MonPolicy", "Geopol", "OilGas", "Fiscal"]

COMBINED_SENTIMENT_RAW = 'combined_sentiment_1d_raw'
COMBINED_SENTIMENT_CHG_RAW = 'combined_sentiment_chg_1d_raw'
COMBINED_SENTIMENT_EXTREME_RAW = 'combined_sentiment_extreme_1d_raw' # !!!! ФОКУС ЗДЕСЬ !!!!
COMBINED_TOPIC_SHIFT_RAW = 'combined_topic_focus_shift_1d_raw'
COMBINED_TOPIC_ENTROPY_RAW = 'combined_topic_entropy_1d_raw'
COMBINED_TOPIC_COLS_RAW = {k: f'combined_{bn}' for k, bn in zip(TOPIC_KEYS_ORDERED, BASE_TOPIC_NAMES)}

OUTPUT_COLS = [
    'news_sentiment_accel_1d_z21','news_topic_entropy_daily_z21',
    'news_sentiment_extreme_1d_z21','news_topic_shock_1d_z21',
    'news_topic_intensity_MacroRF_1d_z21','news_topic_intensity_MonPolicy_1d_z21',
    'news_topic_intensity_Geopol_1d_z21','news_sentiment_level_21d_z63',
    'news_topic_entropy_level_21d_z63','news_topic_intensity_OilGas_21d_z63',
    'news_topic_intensity_Fiscal_21d_z63','news_sentiment_level_63d_z1y'
]

Z_WINDOW_21=21; Z_WINDOW_63=63; Z_WINDOW_252=252
ROLLING_WINDOW_21=21; ROLLING_WINDOW_63=63
MIN_PERIOD_FRAC=0.75

# --- Настройка Логирования (INFO + DEBUG для Z-score) ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
log_handler_file = logging.FileHandler(f"{OUTPUT_FINAL_FEATURES_FILE}.log", mode='w')
log_handler_file.setFormatter(log_formatter)
log_handler_stream = logging.StreamHandler()
log_handler_stream.setFormatter(log_formatter)
logger = logging.getLogger()
if logger.hasHandlers(): logger.handlers.clear()
logger.setLevel(logging.DEBUG) # Включаем DEBUG для отладки Z-score
# Фильтр, чтобы показывать DEBUG только из calculate_rolling_zscore
class ZScoreFilter(logging.Filter):
    def filter(self, record):
        # Показываем INFO и выше всегда, DEBUG только из нужной функции
        return record.levelno >= logging.INFO or record.funcName == 'calculate_rolling_zscore'
logger.addFilter(ZScoreFilter())
logger.addHandler(log_handler_file)
logger.addHandler(log_handler_stream)


# --- Функции ---
def parse_vector_string(vector_str: str, expected_len: int) -> list:
    # (без изменений)
    if pd.isna(vector_str): return [np.nan] * expected_len
    try: parsed_list = ast.literal_eval(str(vector_str))
    except (ValueError, SyntaxError, TypeError): return [np.nan] * expected_len
    if isinstance(parsed_list, list) and len(parsed_list) == expected_len:
        return [pd.to_numeric(item, errors='coerce') for item in parsed_list]
    return [np.nan] * expected_len

# --- calculate_rolling_zscore с отладкой mean/std ---
def calculate_rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    func_name = "calculate_rolling_zscore"
    if not isinstance(series, pd.Series): logging.error(f"Ожидался pandas Series"); return None
    series_name = series.name if series.name else "Unnamed Series"
    logging.debug(f"Запуск для '{series_name}' [Win={window}, MinP={min_periods}]") # DEBUG
    if series.isnull().all(): logging.debug(f"Серия '{series_name}' вся NaN."); return series

    try:
        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    except Exception as e:
        logging.error(f"Ошибка rolling mean/std для '{series_name}': {e}")
        return pd.Series(np.nan, index=series.index)

    # Отладка mean и std
    logging.debug(f"Статистика rolling_mean для '{series_name}':\n{rolling_mean.describe().to_string()}")
    logging.debug(f"Статистика rolling_std для '{series_name}':\n{rolling_std.describe().to_string()}")
    logging.debug(f"Количество NaN в rolling_mean: {rolling_mean.isnull().sum()}")
    logging.debug(f"Количество NaN в rolling_std: {rolling_std.isnull().sum()}")
    logging.debug(f"Количество std == 0: {(rolling_std == 0).sum()}")

    epsilon = 1e-9
    rolling_std_adj = rolling_std.fillna(np.nan).replace(0, epsilon)
    z_scores = (series - rolling_mean) / rolling_std_adj
    zero_or_nan_std_mask = rolling_std.isna() | (rolling_std <= epsilon)
    z_scores.loc[zero_or_nan_std_mask & series.notna()] = 0.0
    z_scores.loc[zero_or_nan_std_mask & series.isna()] = np.nan

    output_nan_count = z_scores.isnull().sum()
    logging.debug(f"Итоговый Z-score для '{series_name}': NaN={output_nan_count}/{len(z_scores)}") # DEBUG

    # Если все еще много NaN, выведем примеры, где это случилось
    if output_nan_count > min_periods + 5 : # Порог, чтобы не выводить для начальных NaN
         nan_indices = z_scores[z_scores.isna()].index
         logging.debug(f"Примеры индексов с NaN в Z-score для '{series_name}' (до 10): {nan_indices[:10].tolist()}")
         # Посмотрим на соответствующие значения в исходных данных и rolling метриках
         debug_subset = pd.DataFrame({
             'input': series.loc[nan_indices[:10]],
             'roll_mean': rolling_mean.loc[nan_indices[:10]],
             'roll_std': rolling_std.loc[nan_indices[:10]],
             'z_calc': z_scores.loc[nan_indices[:10]] # Должны быть NaN
         })
         logging.debug(f"Данные для первых 10 NaN Z-score:\n{debug_subset.to_string()}")

    return z_scores

def load_and_prepare_data(filepath: str, source_name: str, is_source1: bool) -> pd.DataFrame | None:
    # (Версия из v8, без изменений)
    func_name = "load_and_prepare_data"
    logging.info(f"[{func_name}] Загрузка {source_name} из {filepath}...")
    if not os.path.exists(filepath): logging.error(f"[{func_name}] Файл не найден: {filepath}"); return None
    try:
        df = pd.read_csv(filepath, low_memory=False); logging.info(f"[{func_name}] Загружен файл {filepath}, строк: {len(df)}")
        if df.empty: logging.warning(f"[{func_name}] Файл {filepath} пуст."); return None
        if 'date' not in df.columns: logging.error(f"[{func_name}] Колонка 'date' отсутствует: {filepath}"); return None
        df['date'] = pd.to_datetime(df['date'], errors='coerce'); initial_rows = len(df)
        df.dropna(subset=['date'], inplace=True); deleted_rows = initial_rows - len(df)
        if deleted_rows > 0: logging.warning(f"[{func_name}] Удалено {deleted_rows} строк с некорректными датами: {filepath}")
        if df.empty: logging.warning(f"[{func_name}] Нет валидных дат: {filepath}"); return None
        df.sort_values('date', inplace=True)

        current_topic_cols = [f"news_topic_intensity_{topic_key.replace('MonPolicy', 'MonetaryPolicyRF').replace('Geopol', 'GeopoliticsSanctions').replace('OilGas', 'OilGasEnergy').replace('Fiscal', 'FiscalPolicyCorp')}_raw" for topic_key in TOPIC_KEYS_ORDERED]

        if is_source1:
            topic_vector_col = 'news_topic_intensities_1d_raw'
            if topic_vector_col in df.columns:
                logging.info(f"[{func_name}] Парсинг '{topic_vector_col}' для {filepath}..."); parsed_vectors = df[topic_vector_col].apply(parse_vector_string, expected_len=len(current_topic_cols))
                df_topics = pd.DataFrame(parsed_vectors.tolist(), index=df.index, columns=current_topic_cols);
                df = df.drop(columns=[topic_vector_col]).join(df_topics); logging.info(f"[{func_name}] Колонка '{topic_vector_col}' распарсена.")
            else:
                logging.warning(f"[{func_name}] Ожидаемая колонка '{topic_vector_col}' отсутствует: {filepath} (Source 1). Создаю колонки тем с NaN."); [df.setdefault(col, np.nan) for col in current_topic_cols]

        extreme_col_name = 'news_sentiment_extreme_1d_raw'
        if is_source1 and extreme_col_name not in df.columns: logging.error(f"!!! {source_name}: Колонка '{extreme_col_name}' ОТСУТСТВУЕТ !!!")
        elif not is_source1 and extreme_col_name in df.columns: logging.warning(f"!!! {source_name}: Найдена колонка '{extreme_col_name}', хотя не ожидалась.")

        other_raw_cols = list(set( ['news_sentiment_1d_raw', 'news_sentiment_chg_1d_raw', 'news_topic_focus_shift_1d_raw', 'news_topic_entropy_1d_raw'] + current_topic_cols ))
        all_cols_to_check = other_raw_cols + [extreme_col_name]

        present_cols = df.columns.tolist()
        missing_in_df = [col for col in all_cols_to_check if col not in present_cols];
        if missing_in_df: logging.warning(f"[{func_name}] В {filepath} ({source_name}) отсутствуют колонки: {missing_in_df}.")
        for col in all_cols_to_check:
            if col in present_cols: df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.add_suffix(f'_{source_name}'); df.rename(columns={f'date_{source_name}': 'date'}, inplace=True)
        logging.info(f"[{func_name}] Подготовка {source_name} завершена.")
        return df
    except Exception as e: logging.exception(f"[{func_name}] Ошибка при обработке файла {filepath}: {e}"); return None


# --- Основной блок ---
if __name__ == "__main__":
    logging.info("Запуск скрипта слияния и расчета v10 (отладка Z-score extreme)...")

    # 1. Загрузка и подготовка данных
    df1 = load_and_prepare_data(FILE_SOURCE_1, "src1", is_source1=True)
    df2 = load_and_prepare_data(FILE_SOURCE_2, "src2", is_source1=False)
    if df1 is None or df2 is None: logging.error("Не удалось загрузить/подготовить файлы."); exit(1)

    # 2. Слияние DataFrames
    logging.info("Слияние данных по дате...")
    df_merged = pd.merge(df1, df2, on='date', how='outer', suffixes=('_src1', '_src2'))
    df_merged.sort_values('date', inplace=True)
    df_merged.set_index('date', inplace=True)
    logging.info(f"Слияние завершено. Строк: {len(df_merged)}.")

    # 3. Создание КОМБИНИРОВАННЫХ СЫРЫХ признаков
    logging.info("Создание комбинированных сырых признаков...")
    def combine_cols(df, col1_name, col2_name, strategy='mean', combined_name=""): # (без изменений)
        cols_present = [col for col in [col1_name, col2_name] if col in df.columns]
        if not cols_present: logging.warning(f"[combine_cols] Колонки {col1_name}, {col2_name} не найдены для '{combined_name}'."); return pd.Series(np.nan, index=df.index)
        if len(cols_present) == 1: return df[cols_present[0]]
        # logging.info(f"[combine_cols] '{combined_name}': используется стратегия '{strategy}' для {cols_present}.") # Убрали INFO лог
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df[cols_present[0]]) and pd.api.types.is_numeric_dtype(df[cols_present[1]]): return df[cols_present].mean(axis=1, skipna=True)
            else: logging.error(f"[combine_cols] Нечисловые колонки {combined_name}: {cols_present}"); return pd.Series(np.nan, index=df.index)
        elif strategy == 'prefer_src1': return df[col1_name].fillna(df[col2_name])
        elif strategy == 'prefer_src2': return df[col2_name].fillna(df[col1_name])
        else: return df[cols_present[0]]

    # --- Применяем стратегии ---
    raw_sentiment='news_sentiment_1d_raw'; raw_sent_chg='news_sentiment_chg_1d_raw'
    raw_sent_extreme='news_sentiment_extreme_1d_raw'; raw_topic_shift='news_topic_focus_shift_1d_raw'
    raw_topic_entropy='news_topic_entropy_1d_raw'

    df_merged[COMBINED_SENTIMENT_RAW] = combine_cols(df_merged, f'{raw_sentiment}_src1', f'{raw_sentiment}_src2', 'mean', COMBINED_SENTIMENT_RAW)
    df_merged[COMBINED_SENTIMENT_CHG_RAW] = combine_cols(df_merged, f'{raw_sent_chg}_src1', f'{raw_sent_chg}_src2', 'prefer_src2', COMBINED_SENTIMENT_CHG_RAW)
    df_merged[COMBINED_SENTIMENT_EXTREME_RAW] = combine_cols(df_merged, f'{raw_sent_extreme}_src1', f'{raw_sent_extreme}_src2', 'prefer_src1', COMBINED_SENTIMENT_EXTREME_RAW)
    df_merged[COMBINED_TOPIC_SHIFT_RAW] = combine_cols(df_merged, f'{raw_topic_shift}_src1', f'{raw_topic_shift}_src2', 'prefer_src2', COMBINED_TOPIC_SHIFT_RAW)
    df_merged[COMBINED_TOPIC_ENTROPY_RAW] = combine_cols(df_merged, f'{raw_topic_entropy}_src1', f'{raw_topic_entropy}_src2', 'mean', COMBINED_TOPIC_ENTROPY_RAW)

    # Темы
    for i, topic_key in enumerate(TOPIC_KEYS_ORDERED):
        combined_col_name = COMBINED_TOPIC_COLS_RAW[topic_key]
        base_col_name = BASE_TOPIC_NAMES[i]
        df_merged[combined_col_name] = combine_cols(df_merged, f'{base_col_name}_src1', f'{base_col_name}_src2', 'mean', combined_col_name)

    logging.info("Комбинированные сырые признаки созданы. Проверка NaN (особое внимание к EXTREME):")
    combined_cols_list = [COMBINED_SENTIMENT_RAW, COMBINED_SENTIMENT_CHG_RAW, COMBINED_SENTIMENT_EXTREME_RAW, COMBINED_TOPIC_SHIFT_RAW, COMBINED_TOPIC_ENTROPY_RAW] + list(COMBINED_TOPIC_COLS_RAW.values())
    for col in combined_cols_list:
        if col in df_merged.columns:
             nan_count = df_merged[col].isnull().sum()
             log_level = logging.WARNING if nan_count == len(df_merged) else logging.INFO
             logger.log(log_level, f"  - {col}: NaN = {nan_count}/{len(df_merged)} ({nan_count/len(df_merged)*100:.1f}%)")
             # --- Детальная проверка для EXTREME ---
             if col == COMBINED_SENTIMENT_EXTREME_RAW and not df_merged[col].isnull().all():
                  logging.info(f"    -> Статистика для {col}:\n{df_merged[col].describe().to_string()}")
                  logging.info(f"    -> Уникальные значения (top 10):\n{df_merged[col].value_counts().head(10).to_string()}")
        else: logging.warning(f"  - {col}: КОЛОНКА НЕ СОЗДАНА.")

    # 4. Расчет Финальных 12 Признаков
    logging.info("Начало расчета финальных 12 признаков...")
    df_results = pd.DataFrame(index=df_merged.index)
    min_p_21=max(1,int(Z_WINDOW_21*MIN_PERIOD_FRAC)); min_p_63=max(1,int(Z_WINDOW_63*MIN_PERIOD_FRAC))
    min_p_252=max(1,int(Z_WINDOW_252*MIN_PERIOD_FRAC)); min_p_roll21=max(1,int(ROLLING_WINDOW_21*MIN_PERIOD_FRAC))
    min_p_roll63=max(1,int(ROLLING_WINDOW_63*MIN_PERIOD_FRAC))

    def calculate_and_log(df_target, df_source, target_col, source_col_name, func, *args, **kwargs):
        # (без изменений)
        source_series = None
        if isinstance(df_source, pd.DataFrame):
            if source_col_name not in df_source.columns: logging.warning(f"  Пропуск: {target_col} (нет колонки '{source_col_name}')"); df_target[target_col] = np.nan; return
            source_series = df_source[source_col_name]
        elif isinstance(df_source, pd.Series): source_series = df_source
        else: logging.error(f"  Ошибка: Неверный тип источника для {target_col}."); df_target[target_col] = np.nan; return
        if source_series.isnull().all(): logging.warning(f"  Пропуск: {target_col} (вход '{source_col_name}' вся NaN)"); df_target[target_col] = np.nan; return

        # Даем имя серии для отладки внутри Z-score функции
        source_series.rename(source_col_name, inplace=True)

        logging.info(f"  Расчет: {target_col} из '{source_col_name}'")
        try:
            result = func(source_series, *args, **kwargs);
            if result is None: raise ValueError("Функция расчета вернула None")
            df_target[target_col] = result
            output_nan_count = result.isnull().sum()
            logging.info(f"    -> Результат для {target_col}: NaN={output_nan_count}/{len(result)}")
        except Exception as e: logging.exception(f"  Ошибка при расчете {target_col}: {e}"); df_target[target_col] = np.nan

    # --- Расчеты ---
    logging.info("--- Расчет признаков Z21 ---")
    sentiment_accel = df_merged[COMBINED_SENTIMENT_CHG_RAW].diff(1); sentiment_accel.rename('sentiment_accel', inplace=True)
    calculate_and_log(df_results, sentiment_accel, OUTPUT_COLS[0], 'sentiment_accel', calculate_rolling_zscore, window=Z_WINDOW_21, min_periods=min_p_21)
    calculate_and_log(df_results, df_merged, OUTPUT_COLS[1], COMBINED_TOPIC_ENTROPY_RAW, calculate_rolling_zscore, window=Z_WINDOW_21, min_periods=min_p_21)
    # --- Особое внимание сюда ---
    calculate_and_log(df_results, df_merged, OUTPUT_COLS[2], COMBINED_SENTIMENT_EXTREME_RAW, calculate_rolling_zscore, window=Z_WINDOW_21, min_periods=min_p_21)
    # --- Конец особого внимания ---
    calculate_and_log(df_results, df_merged, OUTPUT_COLS[3], COMBINED_TOPIC_SHIFT_RAW, calculate_rolling_zscore, window=Z_WINDOW_21, min_periods=min_p_21)
    for i, topic_key in enumerate(TOPIC_KEYS_ORDERED[:3]):
        calculate_and_log(df_results, df_merged, OUTPUT_COLS[4+i], COMBINED_TOPIC_COLS_RAW.get(topic_key), calculate_rolling_zscore, window=Z_WINDOW_21, min_periods=min_p_21)

    logging.info("--- Расчет признаков Z63 ---")
    sentiment_level_21d = df_merged[COMBINED_SENTIMENT_RAW].rolling(window=ROLLING_WINDOW_21, min_periods=min_p_roll21).mean(); sentiment_level_21d.rename('sentiment_level_21d', inplace=True)
    calculate_and_log(df_results, sentiment_level_21d, OUTPUT_COLS[7], 'sentiment_level_21d', calculate_rolling_zscore, window=Z_WINDOW_63, min_periods=min_p_63)
    entropy_level_21d = df_merged[COMBINED_TOPIC_ENTROPY_RAW].rolling(window=ROLLING_WINDOW_21, min_periods=min_p_roll21).mean(); entropy_level_21d.rename('entropy_level_21d', inplace=True)
    calculate_and_log(df_results, entropy_level_21d, OUTPUT_COLS[8], 'entropy_level_21d', calculate_rolling_zscore, window=Z_WINDOW_63, min_periods=min_p_63)
    for i, topic_key in enumerate(TOPIC_KEYS_ORDERED[3:]):
        output_col_name = OUTPUT_COLS[9+i]; combined_topic_col = COMBINED_TOPIC_COLS_RAW.get(topic_key)
        if combined_topic_col and combined_topic_col in df_merged.columns:
            intensity_level_21d = df_merged[combined_topic_col].rolling(window=ROLLING_WINDOW_21, min_periods=min_p_roll21).mean(); intensity_level_21d.rename(f'intensity_{topic_key}_level_21d', inplace=True)
            calculate_and_log(df_results, intensity_level_21d, output_col_name, intensity_level_21d.name, calculate_rolling_zscore, window=Z_WINDOW_63, min_periods=min_p_63)
        else: logging.warning(f"  Пропуск расчета уровня для {output_col_name} (нет '{combined_topic_col}')"); df_results[output_col_name] = np.nan

    logging.info("--- Расчет признаков Z1Y ---")
    sentiment_level_63d = df_merged[COMBINED_SENTIMENT_RAW].rolling(window=ROLLING_WINDOW_63, min_periods=min_p_roll63).mean(); sentiment_level_63d.rename('sentiment_level_63d', inplace=True)
    calculate_and_log(df_results, sentiment_level_63d, OUTPUT_COLS[11], 'sentiment_level_63d', calculate_rolling_zscore, window=Z_WINDOW_252, min_periods=min_p_252)

    logging.info("Расчет финальных признаков завершен. Проверка NaN в результатах:")
    # ... (проверка NaN без изменений) ...
    for col in OUTPUT_COLS:
        if col in df_results.columns: logging.info(f"  - {col}: NaN = {df_results[col].isnull().sum()}/{len(df_results)} ({df_results[col].isnull().sum()/len(df_results)*100:.1f}%)")
        else: logging.warning(f"  - {col}: КОЛОНКА ОТСУТСТВУЕТ!")

    # 5. Финальная проверка и сохранение
    # ... (без изменений) ...
    final_cols_to_save = OUTPUT_COLS
    missing_final = [col for col in final_cols_to_save if col not in df_results.columns]
    if missing_final: logging.error(f"Не удалось создать колонки: {missing_final}. Добавляю с NaN."); [df_results.setdefault(col, np.nan) for col in missing_final]
    df_final_output = df_results[final_cols_to_save].copy()
    if df_final_output.isnull().all().all(): logging.error("Финальный DataFrame ПОЛНОСТЬЮ NaN!"); exit(1)
    elif df_final_output.empty: logging.error("Финальный DataFrame пуст!"); exit(1)
    logging.info(f"Сохранение итогового файла ({len(df_final_output)} строк) в {OUTPUT_FINAL_FEATURES_FILE}...")
    try:
        df_final_output.reset_index(inplace=True)
        if 'date' in df_final_output.columns: df_final_output['date'] = pd.to_datetime(df_final_output['date']).dt.strftime('%Y-%m-%d')
        df_final_output.to_csv(OUTPUT_FINAL_FEATURES_FILE, index=False, encoding='utf-8', float_format='%.5f')
        logging.info("Файл с финальными 12 признаками успешно сохранен.")
    except Exception as e: logging.exception(f"Ошибка при сохранении итогового файла: {e}")

    logging.info("Скрипт завершил выполнение.")