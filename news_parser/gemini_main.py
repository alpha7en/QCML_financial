# Импорты остаются прежними
import itertools

from google import genai
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from datetime import datetime, timedelta
import time
import logging
import os
import re
import json
from collections import deque
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional

# --- Конфигурация (остается прежней) ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY не найден.")

GEMINI_MODEL_NAME = "gemini-2.0-flash" # Используем рекомендованный Flash
GEMINI_MODEL_LIST = [
    "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash-preview-04-17" # Используем рекомендованный Flash первым
    # Добавьте другие модели, если нужно, например "gemini-1.0-pro"
    # Убедитесь, что у вас есть доступ к этим моделям
]


# Имена файлов
INPUT_HEADLINES_CSV = "lenta_economics_last_5_years.csv"
OUTPUT_FEATURES_CSV = "gemini_daily_headline_features_structured_v2.csv" # Новое имя

# Параметры
SAVE_INTERVAL_DAYS = 10
API_RETRY_DELAY = 3
API_MAX_RETRIES = 3
HISTORY_DAYS = 2

# Темы
TOPIC_LABELS_LIST = [
    "MacroRF", "MonetaryPolicyRF", "GeopoliticsSanctions",
    "OilGasEnergy", "FiscalPolicyCorp"
]
TOPIC_DESCRIPTIONS_DICT = {
    "MacroRF": "Макроэкономика России: ВВП, инфляция, безработица, промышленное производство, розничные продажи, Росстат, прогнозы Минэкономразвития, экономический рост или спад.",
    "MonetaryPolicyRF": "Денежно-кредитная политика ЦБ РФ: Банк России, ключевая ставка, решения ЦБ, заявления Эльвиры Набиуллиной, инфляционные ожидания, валютные интервенции, банковское регулирование.",
    "GeopoliticsSanctions": "Геополитика и Санкции: Внешняя политика РФ, международные отношения, геополитическая напряженность, санкции против России, торговые войны, СВО и ее экономические последствия.",
    "OilGasEnergy": "Нефть, Газ, Энергетика: Цены на нефть (Brent, Urals) и газ, рынок энергоносителей, ОПЕК+, решения ОПЕК, экспорт и добыча нефти/газа в РФ, Газпром, Роснефть, Новатэк.",
    "FiscalPolicyCorp": "Бюджетная политика и Корпорации РФ: Бюджет России, Минфин, налоги (НДС, НДПИ, налог на прибыль), госрасходы, госдолг, крупные госкомпании, приватизация, корпоративные новости крупнейших компаний (Сбербанк, Лукойл и др.) в контексте влияния на рынок/экономику."
}

# --- Pydantic Модели (остаются прежними) ---
class TopicIntensities(BaseModel):
    MacroRF: float = Field(..., ge=0.0, le=1.0)
    MonetaryPolicyRF: float = Field(..., ge=0.0, le=1.0)
    GeopoliticsSanctions: float = Field(..., ge=0.0, le=1.0)
    OilGasEnergy: float = Field(..., ge=0.0, le=1.0)
    FiscalPolicyCorp: float = Field(..., ge=0.0, le=1.0)

class DailyAnalysisResult(BaseModel):
    sentiment: float = Field(..., ge=-1.0, le=1.0)
    topic_intensities: TopicIntensities

# --- Настройка Логирования (остается прежней) ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(f"{OUTPUT_FEATURES_CSV}.log"), logging.StreamHandler()])

# --- Инициализация НОВОГО КЛИЕНТА Gemini ---
try:
    # Используем новый способ инициализации клиента
    client = genai.Client(api_key=GEMINI_API_KEY)
    logging.info(f"Gemini Client инициализирован.")
    # Модель будет указана при вызове generate_content
except Exception as e:
    logging.error(f"Ошибка инициализации Gemini Client: {e}")
    exit(1)

# --- Функции ---

# Функция build_gemini_prompt остается БЕЗ ИЗМЕНЕНИЙ (она готовит текстовый промпт)
def build_gemini_prompt(current_day_headlines: list[str], history: deque) -> str:
    prompt = """
Проанализируй СОВОКУПНОСТЬ следующих новостных заголовков за ОДИН день.
Твоя задача - определить:
1.  Общую усредненную тональность (сентимент) всех заголовков дня (число от -1.0 до 1.0).
2.  Общую интенсивность (долю внимания) для КАЖДОЙ из 5 заданных тем во всех заголовках дня (число от 0.0 до 1.0 для каждой темы). ТЫ ДОЛЖЕН ПОСЧИТАТЬ РЕЗУЛЬТАТ ДЛЯ КАЖДОЙ, А ТОЛЬКО ЗАТЕМ ВЫДАТЬ УСРЕДНЕНИЕ. Каждую новость оценивай вне контекста остальных, это важно

Темы:
"""
    for label, desc in TOPIC_DESCRIPTIONS_DICT.items():
        prompt += f"- {label}: {desc}\n"
    prompt += """
Контекст предыдущих дней (для согласованности):
--- НАЧАЛО ИСТОРИИ ---
"""
    if not history: prompt += "(Нет доступной истории за предыдущие дни)\n"
    else:
        for entry in history:
             result_str = "(Нет данных)"
             if entry.get('parsed_result') and isinstance(entry['parsed_result'], DailyAnalysisResult):
                 try: result_str = json.dumps(entry['parsed_result'].model_dump(), ensure_ascii=False, default=str) # Pydantic V2+
                 except AttributeError: result_str = json.dumps(entry['parsed_result'].dict(), ensure_ascii=False, default=str) # Pydantic V1
             prompt += f"Дата: {entry['date']}\n"
             prompt += "Заголовки (примеры):\n"
             for h in entry['headlines'][:5]: prompt += f"- {h}\n"
             if len(entry['headlines']) > 5: prompt += "- ... и другие\n"
             prompt += f"Результат анализа (предыдущий): {result_str}\n---\n"
    prompt += "--- КОНЕЦ ИСТОРИИ ---\n\n"


    for headline in current_day_headlines: prompt += f"- {headline}\n"
    prompt += "--- КОНЕЦ ЗАГОЛОВКОВ ТЕКУЩЕГО ДНЯ ---\n\n"
    prompt += "Предоставь результат анализа."

    return prompt


# ИЗМЕНЕННАЯ функция вызова API с использованием НОВОГО клиента и `config`
def get_structured_gemini_analysis(prompt: str, model_name_iterator: itertools.cycle) -> Optional[DailyAnalysisResult]:
    """
    Вызывает Gemini API с использованием нового клиента и запросом
    на структурированный вывод по схеме DailyAnalysisResult.
    Возвращает инстанс Pydantic модели или None в случае ошибки.
    """

    retries = 0

    current_model_name = next(model_name_iterator)

    while retries < API_MAX_RETRIES:
        try:
            # Определяем конфигурацию запроса со схемой
            # Используем 'generation_config' как более общее название,
            # но содержимое как в примере с 'config'
            generation_config = {
                'response_mime_type': 'application/json',
                'response_schema': DailyAnalysisResult, # Передаем Pydantic модель/схему
                'candidate_count': 1,
                'temperature': 0.2 # Можно добавить
            }

            # Настройки безопасности


            # Вызов с использованием client.models.generate_content
            response = client.models.generate_content(
                model=f'{current_model_name}', # Префикс models/ может быть необходим
                contents=prompt,
                config=generation_config, # Передаем конфигурацию

            )

            logging.debug(f"Gemini Raw Response Object: {response}")

            # Проверяем наличие ошибки или блокировки ДО попытки парсинга
            # Обратите внимание: структура response может отличаться в разных версиях API/SDK
            # Пробуем универсальный доступ к кандидатам, если он есть
            candidates = getattr(response, 'candidates', [])
            if not candidates:
                 logging.warning(f"Gemini API не вернул кандидатов (возможно, из-за safety filters или ошибки).")
                 # Проверяем prompt_feedback если он есть
                 if hasattr(response, 'prompt_feedback') and getattr(response.prompt_feedback, 'block_reason', None):
                     logging.warning(f"Prompt feedback block reason: {response.prompt_feedback.block_reason}")
                 else:
                      logging.warning("Причина отсутствия кандидатов неизвестна.")

                 time.sleep(API_RETRY_DELAY / 2)
                 retries += 1
                 continue # К следующей попытке

            # --- ПАРСИНГ ---
            # 1. Пытаемся использовать response.parsed (если SDK его предоставляет)
            parsed_result = None
            if hasattr(response, 'parsed'):
                 try:
                      if isinstance(response.parsed, DailyAnalysisResult):
                           parsed_result = response.parsed
                           logging.debug("Успешно получено через response.parsed")
                      else:
                           logging.warning(f"Атрибут response.parsed имеет неожиданный тип: {type(response.parsed)}")
                 except Exception as e:
                      logging.warning(f"Ошибка при доступе к response.parsed: {e}")

            # 2. Если response.parsed не сработал, пытаемся парсить response.text
            if parsed_result is None:
                 if hasattr(response, 'text') and response.text:
                      cleaned_text = response.text.strip()
                      cleaned_text = re.sub(r'^```json\s*|\s*```$', '', cleaned_text).strip()
                      logging.debug(f"Парсинг response.text: {cleaned_text}")
                      if not cleaned_text:
                           logging.warning("response.text пуст после очистки.")
                      else:
                           try:
                               # Для Pydantic V2+
                               # parsed_result = DailyAnalysisResult.model_validate_json(cleaned_text)
                               # Для Pydantic V1
                               parsed_result = DailyAnalysisResult.parse_raw(cleaned_text)
                               logging.debug("Успешно распарсено из response.text")
                           except ValidationError as e:
                                logging.error(f"Ошибка валидации Pydantic при парсинге response.text: {e}. Ответ: {cleaned_text}")
                           except json.JSONDecodeError as e:
                                logging.error(f"Ошибка декодирования JSON из response.text: {e}. Ответ: {cleaned_text}")
                           except Exception as e:
                                logging.error(f"Неизвестная ошибка парсинга response.text: {e}. Ответ: {cleaned_text}")
                 else:
                      logging.warning("Атрибут response.text отсутствует или пуст.")

            # Возвращаем результат, если он был успешно получен одним из способов
            if parsed_result:
                 return parsed_result
            else:
                 # Если не удалось получить результат ни одним способом
                 logging.error("Не удалось получить или распарсить результат от Gemini.")
                 # Увеличиваем счетчик и повторяем попытку, если возможно
                 time.sleep(API_RETRY_DELAY / 2)
                 retries += 1
                 continue

        except Exception as e:
            # Обработка других ошибок API
            retries += 1
            logging.error(f"Ошибка вызова Gemini API (Попытка {retries}/{API_MAX_RETRIES}): {e}")
            # Можно добавить проверку на конкретные типы ошибок API, если они известны
            if retries < API_MAX_RETRIES:
                logging.info(f"Ожидание {API_RETRY_DELAY} секунд перед повторной попыткой...")
                time.sleep(API_RETRY_DELAY)
            else:
                logging.error("Достигнуто максимальное количество попыток API.")
                return None # Все попытки исчерпаны

    # Если цикл завершился без успешного результата
    logging.error("Не удалось получить структурированный ответ от Gemini после всех попыток.")
    return None


# Функция calculate_daily_features_from_gemini остается БЕЗ ИЗМЕНЕНИЙ
# Она уже принимает Optional[DailyAnalysisResult]
def calculate_daily_features_from_gemini(
    date_str: str,
    gemini_structured_result: Optional[DailyAnalysisResult],
    prev_day_features: Optional[dict]
) -> Optional[dict]:
    if gemini_structured_result is None:
        nan_vec = np.array([np.nan] * len(TOPIC_LABELS_LIST))
        features = {"date": date_str,"news_sentiment_1d_raw": np.nan,"news_sentiment_chg_1d_raw": np.nan,
                    **{f"news_topic_intensity_{label}_raw": np.nan for label in TOPIC_LABELS_LIST},
                    "news_topic_focus_shift_1d_raw": np.nan,"news_topic_entropy_1d_raw": np.nan,
                    "_topic_vector_internal": list(nan_vec)}
        return features
    current_sentiment_avg = gemini_structured_result.sentiment
    sentiment_chg = np.nan
    if prev_day_features and pd.notna(prev_day_features.get("sentiment")) and pd.notna(current_sentiment_avg):
        sentiment_chg = current_sentiment_avg - prev_day_features["sentiment"]
    intensities_obj = gemini_structured_result.topic_intensities
    current_topic_vector = np.array([getattr(intensities_obj, label, np.nan) for label in TOPIC_LABELS_LIST])
    topic_focus_shift = np.nan
    prev_topic_vector = prev_day_features.get("topics_vector") if prev_day_features else None
    if prev_topic_vector is not None and not np.isnan(prev_topic_vector).any() and \
       current_topic_vector is not None and not np.isnan(current_topic_vector).any():
        if np.linalg.norm(current_topic_vector) > 1e-6 and np.linalg.norm(prev_topic_vector) > 1e-6:
            dist = cosine(current_topic_vector, prev_topic_vector)
            topic_focus_shift = np.clip(dist, 0.0, 2.0)
            if np.isnan(topic_focus_shift): topic_focus_shift = 0.0
        else: topic_focus_shift = 0.0
    topic_entropy = np.nan
    if current_topic_vector is not None and not np.isnan(current_topic_vector).any() and np.all(current_topic_vector >= 0):
        vec_sum = current_topic_vector.sum()
        if vec_sum > 1e-6:
            normalized_vector = current_topic_vector / vec_sum
            epsilon = 1e-12
            topic_entropy = entropy(normalized_vector + epsilon, base=2)
    features = {"date": date_str,
                "news_sentiment_1d_raw": round(current_sentiment_avg, 5) if pd.notna(current_sentiment_avg) else np.nan,
                "news_sentiment_chg_1d_raw": round(sentiment_chg, 5) if pd.notna(sentiment_chg) else np.nan,
                **{f"news_topic_intensity_{label}_raw": round(getattr(intensities_obj, label, np.nan), 5) if pd.notna(getattr(intensities_obj, label, np.nan)) else np.nan for label in TOPIC_LABELS_LIST},
                "news_topic_focus_shift_1d_raw": round(topic_focus_shift, 5) if pd.notna(topic_focus_shift) else np.nan,
                "news_topic_entropy_1d_raw": round(topic_entropy, 5) if pd.notna(topic_entropy) else np.nan,
                "_topic_vector_internal": list(current_topic_vector)}
    return features


# Функции load_processed_dates_and_last_data и save_results_batch остаются БЕЗ ИЗМЕНЕНИЙ
def load_processed_dates_and_last_data(filepath: str) -> tuple[set, dict | None]:
    processed_dates = set()
    last_day_data = None
    if not os.path.exists(filepath): return processed_dates, last_day_data
    try:
        last_row_dict = None
        usecols = ['date','news_sentiment_1d_raw','_topic_vector_internal'] + [f'news_topic_intensity_{label}_raw' for label in TOPIC_LABELS_LIST]
        reader = pd.read_csv(filepath, chunksize=5000, parse_dates=['date'], usecols=lambda c: c in usecols + ['date'], low_memory=False)
        for chunk in reader:
            processed_dates.update(chunk['date'].dt.strftime('%Y-%m-%d').astype(str))
            if not chunk.empty: last_row_dict = chunk.iloc[-1].to_dict()
        if last_row_dict:
            last_date_obj = last_row_dict.get('date'); logging.info(f"Последняя обработанная дата: {last_date_obj.strftime('%Y-%m-%d') if last_date_obj else 'Не найдена'}")
            last_sentiment = last_row_dict.get('news_sentiment_1d_raw', np.nan)
            last_topic_vector = None
            internal_vec_str = last_row_dict.get('_topic_vector_internal')
            if pd.notna(internal_vec_str):
                try:
                    vec_str = str(internal_vec_str).strip('[] ');
                    if vec_str:
                        parsed_list = [np.nan if x.strip().lower() == 'nan' else float(x.strip()) for x in vec_str.split(',')]
                        if len(parsed_list) == len(TOPIC_LABELS_LIST): last_topic_vector = np.array(parsed_list)
                except Exception as e: pass
            if last_topic_vector is None:
                vec_from_cols = []; valid = True
                for label in TOPIC_LABELS_LIST:
                    col_name = f"news_topic_intensity_{label}_raw"; val = last_row_dict.get(col_name, np.nan)
                    if pd.notna(val) and isinstance(val, (int, float, np.number)): vec_from_cols.append(float(val))
                    elif pd.isna(val): vec_from_cols.append(np.nan)
                    else: valid = False; break
                if valid and len(vec_from_cols) == len(TOPIC_LABELS_LIST): last_topic_vector = np.array(vec_from_cols)
            if last_topic_vector is not None:
                last_day_data = {"sentiment": float(last_sentiment) if pd.notna(last_sentiment) else np.nan, "topics_vector": last_topic_vector}
                logging.debug(f"Восстановлены данные за последний день: {last_day_data}")
            else: logging.warning("Не удалось восстановить вектор тем за последний день.")
    except pd.errors.EmptyDataError: logging.info(f"Файл '{filepath}' пуст.")
    except Exception as e: logging.error(f"Ошибка при чтении '{filepath}': {e}."); processed_dates = set(); last_day_data = None
    return processed_dates, last_day_data

def save_results_batch(results_batch: list[dict], filepath: str, write_header: bool):
    if not results_batch: return
    df_batch = pd.DataFrame(results_batch)
    if '_topic_vector_internal' in df_batch.columns: df_batch['_topic_vector_internal'] = df_batch['_topic_vector_internal'].apply(lambda x: str(list(x)) if isinstance(x, np.ndarray) else str(x))
    cols_order = ['date','news_sentiment_1d_raw','news_sentiment_chg_1d_raw'] + \
                 [f'news_topic_intensity_{label}_raw' for label in TOPIC_LABELS_LIST] + \
                 ['news_topic_focus_shift_1d_raw','news_topic_entropy_1d_raw', '_topic_vector_internal']
    df_batch = df_batch.reindex(columns=cols_order)
    try:
        df_batch.to_csv(filepath, mode='a' if not write_header else 'w', header=write_header, index=False, encoding='utf-8', float_format='%.5f')
        logging.info(f"Сохранен батч из {len(results_batch)} дней в {filepath}")
    except Exception as e: logging.error(f"Критическая ошибка при сохранении батча в {filepath}: {e}")

# --- Основной блок (логика остается прежней, но использует новый вызов API) ---
if __name__ == "__main__":
    logging.info("Запуск Gemini Headline Analyzer (Structured Output v2)...")

    # 1. Загрузка состояния
    processed_dates, last_day_data = load_processed_dates_and_last_data(OUTPUT_FEATURES_CSV)
    write_header = not os.path.exists(OUTPUT_FEATURES_CSV) or os.path.getsize(OUTPUT_FEATURES_CSV) == 0
    if processed_dates: logging.info(f"Найдено {len(processed_dates)} обработанных дат.")

    # 2. Загрузка заголовков
    try:
        df_headlines = pd.read_csv(INPUT_HEADLINES_CSV, usecols=['date', 'headline'], dtype={'date': str, 'headline': str})
        df_headlines.dropna(subset=['date', 'headline'], inplace=True); df_headlines = df_headlines[df_headlines['headline'].str.strip() != '']
        logging.info(f"Загружено {len(df_headlines)} валидных строк заголовков.")
        df_headlines['date'] = pd.to_datetime(df_headlines['date'])
    except FileNotFoundError: logging.error(f"Файл '{INPUT_HEADLINES_CSV}' не найден."); exit(1)
    except Exception as e: logging.error(f"Ошибка загрузки файла заголовков: {e}"); exit(1)

    # 3. Фильтрация и сортировка
    df_headlines['date_str'] = df_headlines['date'].dt.strftime('%Y-%m-%d')
    if processed_dates: df_to_process = df_headlines[~df_headlines['date_str'].isin(processed_dates)].copy()
    else: df_to_process = df_headlines.copy()
    if df_to_process.empty: logging.info("Нет новых дат для обработки."); exit(0)
    logging.info(f"Осталось для обработки {len(df_to_process)} строк ({df_to_process['date_str'].nunique()} дат).")
    df_to_process.sort_values('date', inplace=True)

    # 4. Группировка и обработка
    grouped = df_to_process.groupby('date_str')
    daily_batch_results = []
    days_processed_since_last_save = 0
    is_first_batch = write_header
    history_queue = deque(maxlen=HISTORY_DAYS)

    # Инициализация истории
    if last_day_data:
         last_date_str = sorted(list(processed_dates))[-1] if processed_dates else None
         if last_date_str:
             last_parsed_result = None
             try:
                  intensities = TopicIntensities(**{label: val for label, val in zip(TOPIC_LABELS_LIST, last_day_data["topics_vector"])})
                  last_parsed_result = DailyAnalysisResult(sentiment=last_day_data["sentiment"], topic_intensities=intensities)
             except Exception as e: logging.warning(f"Не удалось создать Pydantic объект из данных за {last_date_str}: {e}")
             history_queue.append({"date": last_date_str,"headlines": ["(Заголовки не сохранены)"],"parsed_result": last_parsed_result})
             logging.info(f"Инициализирована история данными за {last_date_str}")

    model_cycler = itertools.cycle(GEMINI_MODEL_LIST)


    total_days_to_process = len(grouped)
    logging.info(f"Начинаем обработку {total_days_to_process} дней...")

    for date_str, group in tqdm(grouped, desc="Обработка дней"):
        current_day_headlines = group['headline'].tolist()
        prompt = build_gemini_prompt(current_day_headlines, history_queue)
        # Вызов НОВОЙ функции API
        gemini_structured_result = get_structured_gemini_analysis(prompt, model_cycler)

        prev_day_calc_features = None
        if history_queue:
            last_hist_entry = history_queue[-1]
            if last_hist_entry.get('parsed_result') and isinstance(last_hist_entry['parsed_result'], DailyAnalysisResult):
                 prev_parsed = last_hist_entry['parsed_result']
                 prev_topic_vector = np.array([getattr(prev_parsed.topic_intensities, label, np.nan) for label in TOPIC_LABELS_LIST])
                 prev_sentiment = prev_parsed.sentiment
                 prev_day_calc_features = {"sentiment": prev_sentiment, "topics_vector": prev_topic_vector}

        daily_features = calculate_daily_features_from_gemini(date_str, gemini_structured_result, prev_day_calc_features)

        if daily_features:
            daily_batch_results.append(daily_features)
            days_processed_since_last_save += 1
            history_queue.append({"date": date_str, "headlines": current_day_headlines, "parsed_result": gemini_structured_result})
        else: logging.error(f"Не удалось рассчитать фичи для даты {date_str}")

        if days_processed_since_last_save >= SAVE_INTERVAL_DAYS:
            save_results_batch(daily_batch_results, OUTPUT_FEATURES_CSV, is_first_batch)
            daily_batch_results = []; days_processed_since_last_save = 0; is_first_batch = False

    save_results_batch(daily_batch_results, OUTPUT_FEATURES_CSV, is_first_batch)
    logging.info("Анализ заголовков с помощью Gemini (Structured Output v2) завершен.")