import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
import os
import warnings

warnings.filterwarnings('ignore', category=sm.tools.sm_exceptions.EstimationWarning)
warnings.filterwarnings('ignore', category=sm.tools.sm_exceptions.ValueWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Конфигурация для Шага 6 ---
INPUT_FILE_STEP5 = '5intermediate_market_data_step5_normalized.parquet'
OUTPUT_FILE_STEP6 = '6intermediate_market_data_step6_target_reduced_reg.parquet' # Новое имя
FWD_RETURN_DAYS = 15

# ==============================================================================
# --- Загрузка Данных из Шага 5 ---
# ==============================================================================
print(f"\n--- Шаг 6: Расчет Целевой Переменной (Уменьшенные Регрессоры) ---")
print(f"Загрузка данных из файла Шага 5: {INPUT_FILE_STEP5}...")
NEWS_FEATURES = []
df = None
try:
    # ... (Загрузка и проверка файла) ...
    if INPUT_FILE_STEP5.endswith('.parquet'): df = pd.read_parquet(INPUT_FILE_STEP5)
    elif INPUT_FILE_STEP5.endswith('.csv'): df = pd.read_csv(INPUT_FILE_STEP5, parse_dates=['TRADEDATE'])
    else: raise ValueError("Неподдерживаемый формат.")
    # Проверки...
    required_cols = ['TRADEDATE', 'SECID', 'CLOSE']
    norm_cols = [col for col in df.columns if col.startswith('norm_')]
    required_cols.extend(norm_cols)
    NEWS_FEATURES = [col for col in df.columns if col.startswith('news_')]
    required_cols.extend(NEWS_FEATURES)
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required: raise ValueError(f"Отсутствуют колонки: {missing_required}")
    if not pd.api.types.is_datetime64_any_dtype(df['TRADEDATE']): df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    # Установка Индекса
    print("Установка индекса...")
    df.set_index(['TRADEDATE', 'SECID'], inplace=True)
    df.sort_index(inplace=True)
    if not df.index.is_unique: print("WARNING: Индекс не уникален!")
    print(f"Данные загружены. Строк: {len(df)}.")
except Exception as e: print(f"КРИТИЧЕСКАЯ ОШИБКА загрузки: {e}"); exit()

# ==============================================================================
# --- Шаг 6.1: Расчет "Сырой" Будущей Доходности ---
# ==============================================================================
print(f"\nРасчет RawFwdReturn ({FWD_RETURN_DAYS} дней)...")
# ... (Код расчета RawFwdReturn без изменений) ...
df['Close_fwd'] = df.groupby(level='SECID')['CLOSE'].shift(-FWD_RETURN_DAYS)
df['RawFwdReturn'] = np.log(df['Close_fwd'] / df['CLOSE'])
df['RawFwdReturn'] = df['RawFwdReturn'].replace([np.inf, -np.inf], np.nan)
df.drop(columns=['Close_fwd'], inplace=True)
print(f"RawFwdReturn рассчитан. Не-NaN: {df['RawFwdReturn'].notna().sum()}.")

# ==============================================================================
# --- Шаг 6.2-6.4: Кросс-секционная Регрессия (Уменьшенные Регрессоры) ---
# ==============================================================================
print("\nВыполнение кросс-секционной регрессии...")
start_time = time.time()

# --- Определяем список регрессоров X ---
# Берем ТОЛЬКО те norm_ колонки, которые НЕ являются полностью NaN
norm_features_all = [col for col in df.columns if col.startswith('norm_')]
# Отфильтровываем те, где все значения NaN
norm_features_valid = df[norm_features_all].dropna(axis=1, how='all').columns.tolist()

regressors_X = norm_features_valid[:] # Копируем валидные нормализованные
regressors_X.extend(NEWS_FEATURES) # Добавляем новости

print(f"Регрессоры (X) для очистки ({len(regressors_X)}): {regressors_X}")
# Ожидаем увидеть здесь только norm_Momentum, norm_Size, norm_Beta + новости

# --- Функция OLS (без изменений) ---
def get_ols_residuals(data_for_date, y_col, x_cols):
    # --- ОТЛАДКА: Сколько строк ДО dropna? ---
    n_before_drop = len(data_for_date)
    data_clean = data_for_date[[y_col] + x_cols].dropna()
    n_after_drop = len(data_clean)

    Y = data_clean[y_col]
    X = data_clean[x_cols]
    X = sm.add_constant(X, prepend=True, has_constant='skip')

    min_obs_for_ols = max(len(X.columns) + 1, 10)

    # --- ОТЛАДКА: Печатаем информацию, если данных не хватает ---
    if n_after_drop < min_obs_for_ols:
        # Печатаем ТОЛЬКО если раньше было достаточно строк
        # if n_before_drop >= min_obs_for_ols:
        #     print(f"  INFO: Дата {data_for_date.name.strftime('%Y-%m-%d')}: Недостаточно данных после dropna ({n_after_drop} < {min_obs_for_ols}). Пропущено {n_before_drop - n_after_drop} строк с NaN.")
        return pd.Series(np.nan, index=data_for_date.index, dtype=float)

    try:
        model = sm.OLS(Y, X).fit()
        # --- ОТЛАДКА: Печатаем R^2 для диагностики ---
        # print(f"  DEBUG: Дата {data_for_date.name.strftime('%Y-%m-%d')}: R^2 = {model.rsquared:.4f}, Obs = {model.nobs}")
        residuals = pd.Series(np.nan, index=data_for_date.index)
        residuals.loc[model.resid.index] = model.resid
        return residuals
    except Exception as e:
        # print(f"  Ошибка OLS на дату {data_for_date.name.strftime('%Y-%m-%d')}: {e}")
        return pd.Series(np.nan, index=data_for_date.index, dtype=float)


# --- Ручная Сборка Результатов Apply (без изменений) ---
print("Группировка по дате и применение OLS (ручная сборка)...")
all_residuals_list = []
grouped_by_date_for_ols = df.groupby(level='TRADEDATE')
num_dates = len(grouped_by_date_for_ols)
processed_dates = 0
dates_with_residuals = 0 # Счетчик дат, где удалось рассчитать остатки

for date_key, group_data in grouped_by_date_for_ols:
    processed_dates += 1
    if processed_dates % 100 == 0: print(f"  Обработано дат: {processed_dates}/{num_dates}...")

    residuals_for_date = get_ols_residuals(group_data, y_col='RawFwdReturn', x_cols=regressors_X)
    all_residuals_list.append(residuals_for_date)
    if residuals_for_date.notna().any(): # Если хоть один остаток рассчитан
        dates_with_residuals += 1


print(f"\nOLS применен ко всем датам ({time.time() - start_time:.2f} сек). Сборка результатов...")
print(f"Дат, для которых удалось рассчитать остатки: {dates_with_residuals} из {num_dates}")

if all_residuals_list:
    residuals_combined = pd.concat(all_residuals_list)
    residuals_combined.sort_index(inplace=True)
    # Проверка индекса не так важна, присваиваем через reindex
    print("Присвоение ResidualReturn (с выравниванием)...")
    df['ResidualReturn'] = residuals_combined.reindex(df.index) # Используем reindex для безопасности
else:
    df['ResidualReturn'] = np.nan

residual_non_nan = df['ResidualReturn'].notna().sum()
print(f"Итоговое количество Не-NaN ResidualReturn: {residual_non_nan}")
if residual_non_nan == 0 and dates_with_residuals > 0:
     print("WARNING: Остатки рассчитывались для некоторых дат, но итоговый подсчет 0. Проблема с присвоением/reindex?")


# ==============================================================================
# --- Шаг 6.5: Нормализация Остатков ---
# ==============================================================================
print("\nНормализация остатков (ResidualReturn) -> FinalTarget...")
# ... (Код нормализации без изменений) ...
start_time = time.time()
if df['ResidualReturn'].notna().any():
    grouped_residuals = df['ResidualReturn'].groupby(level='TRADEDATE')
    final_target = grouped_residuals.transform(lambda x: (x - x.mean()) / x.std())
    df['FinalTarget'] = final_target.replace([np.inf, -np.inf], np.nan)
    print(f"FinalTarget рассчитан. Не-NaN: {df['FinalTarget'].notna().sum()} ({time.time() - start_time:.2f} сек).")
else:
    print("Нормализация остатков пропущена, так как ResidualReturn пуст (NaN).")
    df['FinalTarget'] = np.nan


# ==============================================================================
# --- Проверка Результата Шага 6 ---
# ==============================================================================
print("\n--- Проверка DataFrame после Шага 6 ---")
# ... (Блок проверки как раньше) ...
print("Новые колонки: ResidualReturn, FinalTarget")
pd.set_option('display.max_rows', 100)
print(df.info(verbose=True, show_counts=True))
pd.reset_option('display.max_rows')
print("\nСтатистика для ResidualReturn:")
if df['ResidualReturn'].notna().any(): print(df['ResidualReturn'].describe())
else: print("ResidualReturn содержит только NaN.")
print("\nСтатистика для FinalTarget:")
if df['FinalTarget'].notna().any(): print(df['FinalTarget'].describe())
else: print("FinalTarget содержит только NaN.")


# ==============================================================================
# --- Сохранение Результата Шага 6 ---
# ==============================================================================
print(f"\n--- Сохранение данных после Шага 6 в {OUTPUT_FILE_STEP6} ---")
# ... (Код сохранения) ...
try:
    df_to_save = df.reset_index()
    print("Колонки для сохранения:", df_to_save.columns.tolist())
    if OUTPUT_FILE_STEP6.endswith('.parquet'): df_to_save.to_parquet(OUTPUT_FILE_STEP6)
    elif OUTPUT_FILE_STEP6.endswith('.csv'): df_to_save.to_csv(OUTPUT_FILE_STEP6, index=False)
    print(f"Файл {OUTPUT_FILE_STEP6} сохранен.")
except Exception as e: print(f"\nОШИБКА при сохранении файла: {e}")


# ==============================================================================
# --- Верификация Сохраненного Файла (Опционально) ---
# ==============================================================================
print(f"\n--- Верификация сохраненного файла {OUTPUT_FILE_STEP6} ---")
# ... (Код верификации) ...
try:
    if OUTPUT_FILE_STEP6.endswith('.parquet'): df_check = pd.read_parquet(OUTPUT_FILE_STEP6)
    elif OUTPUT_FILE_STEP6.endswith('.csv'): df_check = pd.read_csv(OUTPUT_FILE_STEP6, parse_dates=['TRADEDATE'])
    print("Файл успешно прочитан.")
    if 'FinalTarget' in df_check.columns:
         print("\nСтатистика FinalTarget из файла:")
         if df_check['FinalTarget'].notna().any(): print(df_check['FinalTarget'].describe())
         else: print("  Все значения FinalTarget в файле NaN.")
    else: print("ОШИБКА: Колонка FinalTarget отсутствует!")
except Exception as e: print(f"ОШИБКА при верификации: {e}")


print("\n--- Завершение Шага 6 ---")