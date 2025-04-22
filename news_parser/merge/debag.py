import pandas as pd
import numpy as np
import os

# --- Конфигурация ---
FILE_SOURCE_1 = "raw_features.csv"
FILE_SOURCE_2 = "gemini_daily_headline_features_structured_v2 (1).csv"
EXTREME_RATE_COL = 'news_sentiment_extreme_rate_1d_raw' # Имя колонки в Source 1
DATE_COL_SRC1 = 'date'
DATE_COL_SRC2 = 'date' # Предполагаем, что во втором файле тоже 'date'

# Окно для Z-score
Z_WINDOW = 21
MIN_PERIODS = max(1, int(Z_WINDOW * 0.75)) # Используем те же min_periods

# --- Функция Z-score (рабочая версия) ---
def calculate_rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    if not isinstance(series, pd.Series): return None
    series_name = series.name if series.name else "DebugSeries"
    if series.isnull().all(): return series
    try:
        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    except Exception as e: print(f"Ошибка rolling: {e}"); return pd.Series(np.nan, index=series.index)
    epsilon = 1e-9
    rolling_std_adj = rolling_std.fillna(np.nan).replace(0, epsilon)
    z_scores = (series - rolling_mean) / rolling_std_adj
    zero_or_nan_std_mask = rolling_std.isna() | (rolling_std <= epsilon)
    z_scores.loc[zero_or_nan_std_mask & series.notna()] = 0.0
    z_scores.loc[zero_or_nan_std_mask & series.isna()] = np.nan
    return z_scores

# --- Основной блок отладки ---
if __name__ == "__main__":
    print("--- Запуск отладки расчета news_sentiment_extreme_1d_z21 ---")

    # 1. Загрузка минимальных данных
    print(f"\n[Шаг 1] Загрузка данных...")
    if not os.path.exists(FILE_SOURCE_1): print(f"ОШИБКА: Файл {FILE_SOURCE_1} не найден!"); exit()
    if not os.path.exists(FILE_SOURCE_2): print(f"ОШИБКА: Файл {FILE_SOURCE_2} не найден!"); exit()

    try:
        # Загружаем ТОЛЬКО нужные колонки из Source 1
        df1 = pd.read_csv(FILE_SOURCE_1, usecols=[DATE_COL_SRC1, EXTREME_RATE_COL], low_memory=False)
        print(f"  - Загружено {len(df1)} строк из {FILE_SOURCE_1} (колонки: {list(df1.columns)})")
        # Загружаем ТОЛЬКО дату из Source 2
        df2 = pd.read_csv(FILE_SOURCE_2, usecols=[DATE_COL_SRC2], low_memory=False)
        print(f"  - Загружено {len(df2)} строк из {FILE_SOURCE_2} (колонки: {list(df2.columns)})")
    except ValueError as e:
        print(f"ОШИБКА: Не удалось загрузить указанные колонки. Проверьте их имена в файлах! ({e})"); exit()
    except Exception as e:
        print(f"ОШИБКА при загрузке файлов: {e}"); exit()

    # 2. Подготовка дат и определение последней даты
    print(f"\n[Шаг 2] Подготовка дат...")
    try:
        df1[DATE_COL_SRC1] = pd.to_datetime(df1[DATE_COL_SRC1], errors='coerce')
        df1.dropna(subset=[DATE_COL_SRC1], inplace=True)
        df1.sort_values(DATE_COL_SRC1, inplace=True)
        df1.set_index(DATE_COL_SRC1, inplace=True)
        print(f"  - Даты в {FILE_SOURCE_1} обработаны. Диапазон: {df1.index.min()} - {df1.index.max()}")

        df2[DATE_COL_SRC2] = pd.to_datetime(df2[DATE_COL_SRC2], errors='coerce')
        df2.dropna(subset=[DATE_COL_SRC2], inplace=True)
        df2.sort_values(DATE_COL_SRC2, inplace=True)
        df2.set_index(DATE_COL_SRC2, inplace=True)
        print(f"  - Даты в {FILE_SOURCE_2} обработаны. Диапазон: {df2.index.min()} - {df2.index.max()}")

        # Находим последнюю общую дату
        last_common_date = min(df1.index.max(), df2.index.max())
        print(f"  - Последняя общая дата для анализа: {last_common_date.strftime('%Y-%m-%d')}")

        # Определяем диапазон дат для окна Z-score
        start_date_needed = last_common_date - pd.Timedelta(days=Z_WINDOW + 5) # Берем с запасом
        df1_filtered = df1[df1.index >= start_date_needed].copy()
        if len(df1_filtered) < MIN_PERIODS:
             print(f"ОШИБКА: Недостаточно данных в {FILE_SOURCE_1} для расчета Z-score (нужно {MIN_PERIODS}, доступно {len(df1_filtered)} в окне до {last_common_date})")
             exit()
        print(f"  - Отобран диапазон дат для анализа Z-score: {df1_filtered.index.min().strftime('%Y-%m-%d')} - {last_common_date.strftime('%Y-%m-%d')} ({len(df1_filtered)} строк)")

    except Exception as e:
        print(f"ОШИБКА при обработке дат: {e}"); exit()

    # 3. Проверка колонки EXTREME_RATE_COL ДО конвертации
    print(f"\n[Шаг 3] Проверка колонки '{EXTREME_RATE_COL}' ДО конвертации...")
    if EXTREME_RATE_COL not in df1_filtered.columns:
        print(f"ОШИБКА: Колонка '{EXTREME_RATE_COL}' ОТСУТСТВУЕТ в загруженных данных {FILE_SOURCE_1}!"); exit()

    last_day_value_before = df1_filtered.loc[last_common_date, EXTREME_RATE_COL]
    print(f"  - Значение для {last_common_date.strftime('%Y-%m-%d')} ДО конвертации: {last_day_value_before} (тип: {type(last_day_value_before)})")
    print(f"  - Статистика колонки ДО (для окна):\n{df1_filtered[EXTREME_RATE_COL].describe().to_string()}")
    print(f"  - Количество NaN ДО: {df1_filtered[EXTREME_RATE_COL].isnull().sum()}")
    print(f"  - Уникальные значения (top 5) ДО:\n{df1_filtered[EXTREME_RATE_COL].value_counts().head(5).to_string()}")

    # 4. Конвертация в numeric
    print(f"\n[Шаг 4] Конвертация '{EXTREME_RATE_COL}' в numeric...")
    try:
        extreme_series_numeric = pd.to_numeric(df1_filtered[EXTREME_RATE_COL], errors='coerce')
        extreme_series_numeric.rename(EXTREME_RATE_COL, inplace=True) # Даем имя серии
        print("  - Конвертация выполнена.")
    except Exception as e:
        print(f"ОШИБКА при конвертации в numeric: {e}"); exit()

    # 5. Проверка колонки ПОСЛЕ конвертации
    print(f"\n[Шаг 5] Проверка данных ПОСЛЕ конвертации...")
    last_day_value_after = extreme_series_numeric.loc[last_common_date]
    print(f"  - Значение для {last_common_date.strftime('%Y-%m-%d')} ПОСЛЕ конвертации: {last_day_value_after} (тип: {type(last_day_value_after)})")
    print(f"  - Статистика колонки ПОСЛЕ (для окна):\n{extreme_series_numeric.describe().to_string()}")
    nan_count_after = extreme_series_numeric.isnull().sum()
    print(f"  - Количество NaN ПОСЛЕ: {nan_count_after} / {len(extreme_series_numeric)}")
    if nan_count_after == len(extreme_series_numeric):
        print("ОШИБКА: ВСЕ значения стали NaN после конвертации! Проверьте формат данных в CSV.")
        exit()

    # 6. Расчет Z-score
    print(f"\n[Шаг 6] Расчет Z-score для '{EXTREME_RATE_COL}'...")
    try:
        # Передаем сконвертированную числовую серию
        z_score_series = calculate_rolling_zscore(extreme_series_numeric, window=Z_WINDOW, min_periods=MIN_PERIODS)
        print("  - Расчет Z-score выполнен.")
    except Exception as e:
        print(f"ОШИБКА при вызове calculate_rolling_zscore: {e}"); exit()

    # 7. Проверка результата Z-score
    print(f"\n[Шаг 7] Проверка результата Z-score...")
    if z_score_series is None:
        print("ОШИБКА: Функция calculate_rolling_zscore вернула None!"); exit()

    last_day_z_score = z_score_series.loc[last_common_date]
    print(f"  - Итоговый Z-score для {last_common_date.strftime('%Y-%m-%d')}: {last_day_z_score} (тип: {type(last_day_z_score)})")
    print(f"  - Статистика Z-score (для окна):\n{z_score_series.describe().to_string()}")
    z_nan_count = z_score_series.isnull().sum()
    print(f"  - Количество NaN в Z-score: {z_nan_count} / {len(z_score_series)}")

    if pd.isna(last_day_z_score):
        print("\n!!! ВЫВОД: Z-score для последней даты получился NaN. Возможные причины:")
        print(f"    1. Исходное значение '{EXTREME_RATE_COL}' для этой даты было NaN (см. Шаг 5).")
        print(f"    2. Не хватило данных для расчета rolling mean/std (нужно {MIN_PERIODS} точек в окне {Z_WINDOW} дней).")
        print(f"    3. Ошибка внутри функции calculate_rolling_zscore (менее вероятно, если нет сообщений об ошибках).")
    elif last_day_z_score == 0.0:
         print("\n!!! ВЫВОД: Z-score для последней даты равен 0.0. Возможные причины:")
         print(f"    1. Стандартное отклонение в окне было равно 0 (все значения одинаковы).")
         print(f"    2. Значение для этой даты точно совпало со скользящим средним.")
    else:
        print("\nВЫВОД: Z-score для последней даты рассчитан и не является NaN или 0.")

    print("\n--- Отладка завершена ---")