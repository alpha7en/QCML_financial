import pandas as pd
import numpy as np
import time
import os

# --- Конфигурация для Шага 4 ---
INPUT_FILE_STEP3 = 'intermediate_market_data_step3_final.parquet'
OUTPUT_FILE_STEP4 = 'intermediate_market_data_step4_final_v5_debug.parquet' # Новое имя
BETA_WINDOW = 252
MIN_BETA_PERIODS = int(BETA_WINDOW * 0.8) # 201
# --- Отладка ---
DEBUG_MAX_GROUPS = 5 # Детальная отладка для первых N групп
DEBUG_PRINT_WINDOW_DETAIL = True # Печатать детализацию по окнам?

# ==============================================================================
# --- Загрузка Данных из Шага 3 ---
# ==============================================================================
print(f"\n--- Изолированный Шаг 4: Расчет Beta (Чистый Индекс, Глубокая Отладка) ---")
print(f"Загрузка данных из файла Шага 3: {INPUT_FILE_STEP3}...")
NEWS_FEATURES = []
df_unindexed = None # Для расчета индекса до установки MultiIndex
try:
    if INPUT_FILE_STEP3.endswith('.parquet'): df_unindexed = pd.read_parquet(INPUT_FILE_STEP3)
    elif INPUT_FILE_STEP3.endswith('.csv'): df_unindexed = pd.read_csv(INPUT_FILE_STEP3, parse_dates=['TRADEDATE'])
    else: raise ValueError("Неподдерживаемый формат.")

    # --- Базовые Проверки ---
    print("Проверка базовых колонок ПОСЛЕ ЗАГРУЗКИ...")
    required_cols = ['TRADEDATE', 'SECID', 'CLOSE', 'IMOEX_CLOSE'] # Проверяем наличие IMOEX_CLOSE сразу
    for col in required_cols:
        if col not in df_unindexed.columns: raise ValueError(f"Отсутствует необходимая колонка: {col}")
    if not pd.api.types.is_datetime64_any_dtype(df_unindexed['TRADEDATE']):
        df_unindexed['TRADEDATE'] = pd.to_datetime(df_unindexed['TRADEDATE'])

    CALCULATE_BETA = not df_unindexed['IMOEX_CLOSE'].isnull().all()
    if not CALCULATE_BETA: print("WARNING: Колонка IMOEX_CLOSE пуста, Beta не будет рассчитана.")

    NEWS_FEATURES = [col for col in df_unindexed.columns if col.startswith('news_')]
    if NEWS_FEATURES: print(f"Обнаружены новостные колонки ({len(NEWS_FEATURES)}).")

except FileNotFoundError: print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл {INPUT_FILE_STEP3} не найден."); exit()
except Exception as e: print(f"КРИТИЧЕСКАЯ ОШИБКА загрузки: {e}"); exit()

# ==============================================================================
# --- Шаг 4: Расчет Beta (Чистый Индекс, Итерация, Глубокая Отладка) ---
# ==============================================================================

# --- 1. ПРАВИЛЬНЫЙ Расчет Доходностей ---
print("\nРасчет лог-доходностей...")
# Доходность Акций (рассчитаем после установки индекса)
df = df_unindexed.set_index(['TRADEDATE', 'SECID']).sort_index()
if not df.index.is_unique: print("WARNING: Индекс не уникален!")

df['log_ret'] = df.groupby(level='SECID')['CLOSE'].pct_change(fill_method=None)
df['log_ret'] = np.log1p(df['log_ret'])
print(f"Доходности акций рассчитаны. Не-NaN: {df['log_ret'].notna().sum()}")

# Доходность Индекса (по "чистой" серии до MultiIndex)
# Создаем Series: Дата -> IMOEX_CLOSE (убираем дубликаты дат)
imoex_series = df_unindexed[['TRADEDATE', 'IMOEX_CLOSE']].drop_duplicates().set_index('TRADEDATE')['IMOEX_CLOSE'].sort_index()
log_ret_idx_series = np.log1p(imoex_series.pct_change(fill_method=None))
log_ret_idx_series.name = 'log_ret_idx'
print(f"Доходности индекса рассчитаны. Не-NaN: {log_ret_idx_series.notna().sum()}")
print(f"ОТЛАДКА: Количество NaN в log_ret_idx_series: {log_ret_idx_series.isnull().sum()}")
if log_ret_idx_series.isnull().all():
     print("ОШИБКА ОТЛАДКИ: Все значения log_ret_idx_series являются NaN!")
     CALCULATE_BETA = False # Не можем считать бету

# Присоединяем "чистую" доходность индекса к основному DF
df = df.join(log_ret_idx_series, on='TRADEDATE')
print(f"Чистая доходность индекса присоединена. Не-NaN в df['log_ret_idx']: {df['log_ret_idx'].notna().sum()}")

# Инициализируем колонку Beta как NaN
df['Beta'] = np.nan

if CALCULATE_BETA:
    print(f"\n--- Расчет Beta (окно {BETA_WINDOW}, мин. {MIN_BETA_PERIODS}) ---")
    start_time = time.time()

    # 2. Итерация по группам с детальной отладкой окна
    print("Расчет скользящей беты через итерацию по группам...")
    all_beta_results = []
    groups = df[['log_ret', 'log_ret_idx']].groupby(level='SECID')
    num_groups = len(groups)
    processed_groups = 0
    calculated_betas_count_total = 0

    for name, group_df in groups:
        processed_groups += 1
        is_debug_group = processed_groups <= DEBUG_MAX_GROUPS

        log_ret = group_df['log_ret'] # Оригинальные с NaN
        log_ret_idx = group_df['log_ret_idx'] # Оригинальные с NaN

        if is_debug_group:
            print(f"\n--- ОТЛАДКА ГРУППЫ: {name} ({processed_groups}/{num_groups}) ---")
            print(f"  Размер группы: {len(group_df)}")
            print(f"  Не-NaN log_ret/log_ret_idx: {log_ret.notna().sum()}/{log_ret_idx.notna().sum()}")

        # Рассчитываем статистики (Pandas должен обработать NaN попарно для cov)
        try:
            # Rolling объекты с min_periods
            rolling_ret = log_ret.rolling(window=BETA_WINDOW, min_periods=MIN_BETA_PERIODS)
            rolling_ret_idx = log_ret_idx.rolling(window=BETA_WINDOW, min_periods=MIN_BETA_PERIODS)

            # Рассчитываем ковариацию и дисперсию
            rolling_cov = rolling_ret.cov(log_ret_idx)
            rolling_var = rolling_ret_idx.var()

            # --- ОТЛАДКА: Детальная проверка первых валидных окон ---
            if is_debug_group and DEBUG_PRINT_WINDOW_DETAIL:
                print(f"  ОТЛАДКА ОКНА (Первые 5 валидных Cov/Var):")
                # Индексы, где оба расчета не NaN
                potential_beta_idx = rolling_cov.notna() & rolling_var.notna()
                valid_beta_indices = potential_beta_idx[potential_beta_idx].index[:5]

                if not valid_beta_indices.empty:
                    print("    Дата        | Var_Idx  | Cov      | Beta_Raw | ValidPairs")
                    for idx_val in valid_beta_indices:
                        var_val = rolling_var.loc[idx_val]
                        cov_val = rolling_cov.loc[idx_val]
                        beta_raw = cov_val / var_val if var_val != 0 else np.nan

                        # Оценка количества пар в окне
                        current_pos = group_df.index.get_loc(idx_val)
                        start_pos = max(0, current_pos - BETA_WINDOW + 1)
                        window_data = group_df.iloc[start_pos : current_pos + 1]
                        valid_pairs_in_window = (window_data['log_ret'].notna() & window_data['log_ret_idx'].notna()).sum()

                        date_str = idx_val[0].strftime('%Y-%m-%d')
                        # --- ВЫВОДИМ ЯВНО, ВЫПОЛНЯЕТСЯ ЛИ УСЛОВИЕ MIN_PERIODS ---
                        sufficient_pairs = valid_pairs_in_window >= MIN_BETA_PERIODS
                        print(f"    {date_str} | {var_val:8.4f} | {cov_val:8.4f} | {beta_raw:8.4f} | {valid_pairs_in_window:^10} | MinPeriodsMet? {sufficient_pairs}")
                        if not sufficient_pairs and not np.isnan(beta_raw):
                            print("      !!! WARNING: Beta рассчитана, но пар < min_periods? !!!")

                else:
                    print("    Не найдено валидных окон (где Var и Cov не NaN) для отладки.")

            # Рассчитываем бету для группы
            epsilon = 1e-10
            beta_group = rolling_cov / rolling_var.where(rolling_var.abs() > epsilon, np.nan)
            beta_group = beta_group.replace([np.inf, -np.inf], np.nan)

            calculated_betas_group = beta_group.notna().sum()
            calculated_betas_count_total += calculated_betas_group
            if is_debug_group:
                 print(f"  Рассчитано {calculated_betas_group} не-NaN значений Beta для группы.")

        except Exception as group_e:
            print(f"  ОШИБКА при расчете Beta для группы {name}: {group_e}")
            beta_group = pd.Series(np.nan, index=group_df.index)

        all_beta_results.append(beta_group)

    # 3. Сборка результатов
    print("\nСборка результатов расчета беты...")
    if all_beta_results:
        beta_final_series = pd.concat(all_beta_results)
        print(f"ОТЛАДКА: Проверка индекса beta_final_series после concat: Совпадает с df.index? {beta_final_series.index.equals(df.index)}, Длина: {len(beta_final_series)}")
        # Используем .reindex для безопасности перед присвоением
        print("Выравнивание индекса beta_final_series по df.index перед присвоением...")
        try:
            df['Beta'] = beta_final_series.reindex(df.index)
            print("Присвоение Beta завершено.")
        except Exception as e:
            print(f"ОШИБКА при выравнивании/присвоении Beta: {e}. Beta останется NaN.")
    else:
        print("Не удалось собрать результаты беты.")

    # 4. Очистка
    print("Удаление временных колонок доходностей...")
    df.drop(columns=['log_ret', 'log_ret_idx'], inplace=True, errors='ignore')

    # 5. Итоговая проверка
    print(f"Beta рассчитана ({time.time() - start_time:.2f} сек).")
    # ... (Блок проверки Beta как раньше) ...
    beta_total_non_nan = df['Beta'].notna().sum()
    print(f"  Итоговое количество НЕ-NaN Beta: {beta_total_non_nan}")
    if beta_total_non_nan != calculated_betas_count_total:
         print(f"  !!! WARNING: Расхождение в подсчете не-NaN Beta ({calculated_betas_count_total} в цикле vs {beta_total_non_nan} в итоге) !!!")
    if beta_total_non_nan > 0:
        print("  Статистика Beta (describe):"); print(df['Beta'].describe())
    else: print("  Все значения Beta являются NaN.")

else:
    print("\nРасчет Beta пропущен.")

# --- GICS Dummies ---
print("\nСоздание GICS Dummies пропущено.")

# ==============================================================================
# --- Финальная Проверка DataFrame В ПАМЯТИ перед сохранением ---
# ==============================================================================
print("\n--- Финальная проверка DataFrame В ПАМЯТИ (Перед Сохранением) ---")
# ... (Блок финальной проверки как раньше) ...
print("Колонки:", df.columns.tolist())
print("\nСтатистика Beta (в памяти):")
if 'Beta' in df.columns and df['Beta'].notna().any(): print(df['Beta'].describe())
else: print("Beta NaN/отсутствует.")

# ==============================================================================
# --- Сохранение Результата Шага 4 ---
# ==============================================================================
print(f"\n--- Сохранение данных после Шага 4 в {OUTPUT_FILE_STEP4} ---")
# ... (Код сохранения без изменений) ...
try:
    df_to_save = df.reset_index()
    print("Колонки для сохранения:", df_to_save.columns.tolist())
    if OUTPUT_FILE_STEP4.endswith('.parquet'): df_to_save.to_parquet(OUTPUT_FILE_STEP4)
    elif OUTPUT_FILE_STEP4.endswith('.csv'): df_to_save.to_csv(OUTPUT_FILE_STEP4, index=False)
    print(f"Файл {OUTPUT_FILE_STEP4} сохранен.")
except Exception as e: print(f"\nОШИБКА при сохранении файла: {e}")

# ==============================================================================
# --- Верификация Сохраненного Файла ---
# ==============================================================================
print(f"\n--- Верификация сохраненного файла {OUTPUT_FILE_STEP4} ---")
# ... (Код верификации без изменений) ...
try:
    if OUTPUT_FILE_STEP4.endswith('.parquet'): df_check = pd.read_parquet(OUTPUT_FILE_STEP4)
    elif OUTPUT_FILE_STEP4.endswith('.csv'): df_check = pd.read_csv(OUTPUT_FILE_STEP4, parse_dates=['TRADEDATE'])
    print("Файл успешно прочитан.")
    # ... (проверки Beta, Momentum, Size, Новостей из файла) ...
    print("\nСтатистика Beta из файла:")
    if 'Beta' in df_check.columns and df_check['Beta'].notna().any(): print(df_check['Beta'].describe())
    else: print("Beta NaN/отсутствует.")
except Exception as e: print(f"ОШИБКА при верификации: {e}")

print("\n--- Завершение Шага 4 ---")