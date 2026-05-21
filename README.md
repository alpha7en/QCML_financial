# QCML_financial

Проект экспериментального исследования квантовой модели для предсказания биржевой
динамики на базе новостных и рыночных факторов. Репозиторий содержит исходный пайплайн
подготовки данных, классические и квантовые модели, а также артефакты промежуточных
датасетов. Полное описание исследования — в `Отчёт_QML.pdf`.

## Краткое содержание отчёта
- Цель: предсказать динамику акций по рыночным факторам и новостным признакам
  (по мотивам FinBERT/нулевого шота), которых не было в исходной статье Qognitive.
- Данные: ~1500 акций Мосбиржи за 5 лет, дневные параметры (close, объём, капитализация,
  индекс IMOEX), глобальные новостные признаки (Lenta/RBC/Коммерсант).
- Новости: глобальные признаки объединяются с акциями через эмбеддинги компаний (3D).
- Подход: очистка линейной компоненты (кросс-секционная OLS) и обучение моделей на
  остатках (FinalTarget).
- Результаты: базовый CNN‑GRU давал R² ≈ 0.034; более сильная классическая модель — R² ≈ 0.2.
  Квантовая модель обучалась медленно, но показывала снижение loss и признаки нелинейных
  зависимостей; альтернативная квантовая модель лучше ловила экстремальные значения.
- Ограничения: нет полноценных финансовых отчётностей, часть пайплайна строилась
  вручную и с отладкой; полный backtest стратегии не реализован.

## Структура репозитория
- `QCML/quantum/` — квантовая модель (PennyLane): `qcml_model.py`, `main_train.py`,
  конфиг `config.json`, датасет `0moex_qcml_final_dataset_with_embeddings.parquet`.
- `QCML/YaCloude/` — классическая CNN‑GRU/TCN‑логика и job‑конфиг (`train_tcn_job.py`,
  `config.json`, `config.yaml`).
- `QCML/classic/` — ноутбуки с классическими экспериментами.
- `news_parser/` — сбор новостей, извлечение признаков, сбор/слияние данных,
  построение эмбеддингов компаний.
- `Отчёт_QML.pdf` — полный исследовательский отчёт.

## Основные артефакты данных
- `news_parser/lenta_economics_last_5_years.csv` — заголовки Lenta.ru.
- `news_parser/gemini_daily_headline_features_structured_v2.csv` — дневные признаки
  из Gemini.
- `news_parser/final_dataset.csv` — промежуточный датасет новостей для слияния.
- `news_parser/dataset2version/*intermediate*.parquet` — промежуточные шаги сборки.
- `news_parser/dataset2version/moex_qcml_final_dataset.parquet` — финальный датасет
  (до очистки).
- `news_parser/dataset2version/moex_qcml_final_dataset_cleaned_with_news.parquet` —
  очищенный датасет с новостями.
- `news_parser/dataset2version/0moex_qcml_final_dataset_with_embeddings.parquet` —
  финальный датасет с эмбеддингами (вход для моделей).

## Восстановленный пайплайн данных
1. **Сбор новостей**
   - Lenta.ru: `news_parser/main_rbc.py` (скрапинг заголовков).
   - RBC: `news_parser/settings.py` + `news_parser/scrape_rbc_articles.py`,
     а также `news_parser/analysis/rbc/` для разведки рубрик.
   - Коммерсант: `news_parser/analysis/parser.py` (скрапинг архивов).
2. **Нормализация новостей в признаки**
   - Yandex Cloud (тональность + темы): `news_parser/analysis/main.py`,
     `news_parser/analysis/smart_main.py` → `raw_features.csv`.
   - Gemini (дневные признаки по заголовкам): `news_parser/gemini_main.py`
     → `gemini_daily_headline_features_structured_v2.csv`.
3. **Сбор рыночных данных и базовое объединение**
   - `news_parser/dataset2version/main.py` — получение тикеров, истории торгов,
     IMOEX и слияние с новостями.
4. **Изолированные шаги подготовки**
   - `stage2.py` — Momentum/Size + placeholder‑финансы.
   - `stage4.py` — rolling‑beta (IMOEX).
   - `stage5.py` — кросс‑секционная нормализация (Z‑score).
   - `stage6.py` — OLS‑регрессия, ResidualReturn, `FinalTarget`.
   - `stage7.py` — сборка финального датасета.
5. **Финальная очистка**
   - `final_filter.py` — выброс NaN и сокращение до нужных признаков.
6. **Эмбеддинги компаний**
   - `embeding/gemini.py` или `embeding/main_yandex.py` — генерация описаний/эмбеддингов.
   - `embeding/compression.py` — UMAP/сжатие до 3D.
7. **Слияние эмбеддингов**
   - `merge.py` — добавление `umap_1..3` и удаление `SECID`.

## Модели
- **QCML (квантовая)**: `QCML/quantum/qcml_model.py`, обучение в
  `QCML/quantum/main_train.py`, параметры — `QCML/quantum/config.json`.
- **Классические**: ноутбуки `QCML/classic/` (CNN‑GRU/MLP эксперименты).
- **CNN‑GRU/TCN‑job**: `QCML/YaCloude/train_tcn_job.py` + `QCML/YaCloude/config.json`.

## Минимальный порядок запуска
1. Скрапинг новостей (Lenta/RBC/Коммерсант).
2. Вычисление новостных признаков (Yandex/Gemini).
3. Сбор рыночных данных и шаги `stage2 → stage7`.
4. Финальная очистка и эмбеддинги компаний.
5. Обучение моделей (классика / QCML).

## Переменные окружения и секреты
Создайте `.env` (см. `.env.example`) или экспортируйте переменные:
- `GEMINI_API_KEY`
- `YANDEX_FOLDER_ID`
- `YANDEX_AUTH_TOKEN`

## Статус и пробелы
Часть пайплайна отлаживалась вручную, финансовые отчётности не собирались,
а полный backtest стратегии не реализован. Репозиторий отражает восстановленную
цепочку исследования и оставляет пространство для повторного обучения.
