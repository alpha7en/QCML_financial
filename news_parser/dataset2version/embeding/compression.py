import pandas as pd
import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE # Для визуализации, если нужно
import matplotlib.pyplot as plt

try:
    import umap
    umap_available = True
except ImportError:
    print("Библиотека umap-learn не найдена. Установите: pip install umap-learn")
    umap_available = False

# --- ЗАГРУЗКА ЭМБЕДДИНГОВ ---
base_path = './'
embeddings_yandex_cache_file = os.path.join(base_path, 'ticker_embeddings_yandex_cache.pkl')
ticker_to_embedding_map = {}
embedding_dimension = None

if os.path.exists(embeddings_yandex_cache_file):
    try:
        with open(embeddings_yandex_cache_file, 'rb') as f:
            ticker_to_embedding_map = pickle.load(f)
        if ticker_to_embedding_map:
            # Удаляем тикеры с пустыми эмбеддингами
            tickers_to_remove = [t for t, emb in ticker_to_embedding_map.items() if not isinstance(emb, np.ndarray) or emb.size == 0]
            if tickers_to_remove:
                print(f"Удаление {len(tickers_to_remove)} тикеров с пустыми эмбеддингами из карты.")
                for t in tickers_to_remove: del ticker_to_embedding_map
            # Определяем размерность
            if ticker_to_embedding_map:
                 embedding_dimension = len(next(iter(ticker_to_embedding_map.values())))
                 print(f"Загружены эмбеддинги ({len(ticker_to_embedding_map)} шт., размерность {embedding_dimension})")
            else: print("Кэш эмбеддингов пуст после удаления.")
        else: print("Кэш эмбеддингов пуст.")
    except Exception as e: print(f"Ошибка загрузки кэша: {e}")

if not ticker_to_embedding_map:
    raise SystemExit("Нет эмбеддингов для обработки.")

# --- ПОДГОТОВКА ДАННЫХ ДЛЯ СЖАТИЯ ---
# Получаем список тикеров и матрицу эмбеддингов в согласованном порядке
tickers_list = list(ticker_to_embedding_map.keys())
embedding_matrix = np.array([ticker_to_embedding_map[t] for t in tickers_list])
print(f"Матрица эмбеддингов для сжатия: {embedding_matrix.shape}") # Должно быть (N_tickers, embedding_dimension)

# --- СЖАТИЕ ---
N_COMPONENTS = 3 # <<< Установите желаемое количество компонент (3 или 4)

# 1. PCA
print(f"\n--- Сжатие с помощью PCA до {N_COMPONENTS} компонент ---")
pca = PCA(n_components=N_COMPONENTS, random_state=42)
embeddings_pca = pca.fit_transform(embedding_matrix)
print(f"Объясненная дисперсия по компонентам: {pca.explained_variance_ratio_}")
print(f"Суммарная объясненная дисперсия: {np.sum(pca.explained_variance_ratio_):.4f}")
print(f"Размер сжатых эмбеддингов PCA: {embeddings_pca.shape}")

# Создаем словарь: тикер -> сжатый эмбеддинг PCA
ticker_to_pca_map = {ticker: emb for ticker, emb in zip(tickers_list, embeddings_pca)}

# 2. UMAP (если доступен)
ticker_to_umap_map = {}
embeddings_umap = None
if umap_available:
    print(f"\n--- Сжатие с помощью UMAP до {N_COMPONENTS} компонент ---")
    # Параметры UMAP можно настраивать
    reducer = umap.UMAP(n_components=N_COMPONENTS,
                        n_neighbors=15, # default=15, можно уменьшить (5-10) для локальной структуры
                        min_dist=0.1,   # default=0.1, можно увеличить для более разреженных кластеров
                        metric='cosine', # Косинусное расстояние часто хорошо для текстовых эмбеддингов
                        random_state=42,
                        n_jobs=1) # n_jobs=-1 может ускорить, но иногда вызывает проблемы
    embeddings_umap = reducer.fit_transform(embedding_matrix)
    print(f"Размер сжатых эмбеддингов UMAP: {embeddings_umap.shape}")
    # Создаем словарь: тикер -> сжатый эмбеддинг UMAP
    ticker_to_umap_map = {ticker: emb for ticker, emb in zip(tickers_list, embeddings_umap)}
else:
    print("\nUMAP недоступен, пропускаем.")


# --- ИСПОЛЬЗОВАНИЕ СЖАТЫХ ЭМБЕДДИНГОВ ---
# Теперь вы можете использовать `ticker_to_pca_map` или `ticker_to_umap_map`
# для добавления 3-4 новых признаков к вашим данным перед созданием последовательностей
# и обучением основной модели.

# Пример добавления PCA признаков к вашему df_cleaned
# (Предполагается, что индекс df_cleaned содержит тикеры или есть колонка 'ticker')

# Создаем DataFrame из сжатых эмбеддингов
pca_df = pd.DataFrame(embeddings_pca, index=tickers_list, columns=[f'pca_{i+1}' for i in range(N_COMPONENTS)])
umap_df = pd.DataFrame(embeddings_umap, index=tickers_list, columns=[f'umap_{i+1}' for i in range(N_COMPONENTS)]) if embeddings_umap is not None else None

print("\nПример сжатых PCA эмбеддингов:")
print(pca_df.head())
if umap_df is not None:
    # Сохраняем PCA и UMAP в CSV
    pca_csv_path = 'pca_embeddings.csv'
    umap_csv_path = 'umap_embeddings.csv'
    pca_df.to_csv(pca_csv_path)
    print(f"PCA embeddings saved to: {pca_csv_path}")
    umap_df.to_csv(umap_csv_path)
    print(f"UMAP embeddings saved to: {umap_csv_path}")

    print("\nПример сжатых UMAP эмбеддингов:")
    print(umap_df.head())



# >>> ДАЛЬНЕЙШИЕ ШАГИ (в вашем основном скрипте): <<<
# 1. Загрузить нужный DataFrame сжатых эмбеддингов (pca_df или umap_df).
# 2. Объединить (merge) его с вашим df_cleaned по тикеру.
#    Убедитесь, что у df_cleaned есть столбец с тикером или используйте индекс.
#    Пример: df_cleaned = df_cleaned.merge(pca_df, left_on='ticker_column', right_index=True, how='left')
# 3. Добавить имена новых столбцов ('pca_1', 'pca_2', ...) в `feature_cols`.
# 4. Продолжить выполнение скрипта (масштабирование, создание последовательностей и т.д.)
#    с обновленным набором признаков.
# 5. Не забудьте обновить `input_dim` для вашей нейросети!

# ... (весь предыдущий код до секции ИСПОЛЬЗОВАНИЕ СЖАТЫХ ЭМБЕДДИНГОВ) ...

# --- ВИЗУАЛИЗАЦИЯ СЖАТЫХ ЭМБЕДДИНГОВ (добавлено) ---
print("\n--- Визуализация сжатых эмбеддингов ---")

# Создаем DataFrame для удобства работы с результатами
results_df = pd.DataFrame(index=tickers_list)
results_df[[f'pca_{i+1}' for i in range(N_COMPONENTS)]] = embeddings_pca
if embeddings_umap is not None:
    results_df[[f'umap_{i+1}' for i in range(N_COMPONENTS)]] = embeddings_umap

# --- Опционально: Попробуем получить сектора для раскраски ---
# Это потребует от вас наличия файла или словаря с классификацией
# Пример: ticker_to_sector = {'SBER': 'Финансы', 'GAZP': 'Нефть/Газ', ...}
ticker_to_sector = {'SBER': 'Финансы', 'GAZP': 'Нефть/Газ', "AFLT":"авиа", "AKRN":"удобрения", "ARSA":"комунальные", } # ЗАГЛУШКА - Замените на ваши реальные данные секторов
# Пример загрузки из CSV:
# sector_mapping_file = os.path.join(base_path, 'ticker_sectors.csv')
# try:
#     sector_df = pd.read_csv(sector_mapping_file, index_col=0) # Предполагаем столбец с тикером как индекс
#     if 'sector' in sector_df.columns: # Ищем столбец 'sector'
#          ticker_to_sector = sector_df['sector'].to_dict()
#          print(f"Загружена информация о секторах для {len(ticker_to_sector)} тикеров.")
#     else: print("Столбец 'sector' не найден в файле.")
# except FileNotFoundError:
#     print(f"Файл с секторами '{sector_mapping_file}' не найден. Графики будут без раскраски по секторам.")
# except Exception as e:
#     print(f"Ошибка загрузки секторов: {e}")

# Добавляем сектор в DataFrame, если информация доступна
if ticker_to_sector:
    results_df['sector'] = results_df.index.map(ticker_to_sector).fillna('Unknown')
    sector_colors = plt.cm.tab10(np.linspace(0, 1, results_df['sector'].nunique()))
    sector_color_map = {sector: color for sector, color in zip(results_df['sector'].unique(), sector_colors)}
    colors = results_df['sector'].map(sector_color_map)
else:
    results_df['sector'] = 'N/A'
    colors = 'blue' # Используем один цвет, если секторов нет

# --- Строим графики ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns # Для лучшей палитры и стилей

    # Используем стиль seaborn для красоты
    sns.set(style='whitegrid')

    num_plots = 1 + (1 if embeddings_umap is not None else 0)
    plt.figure(figsize=(8 * num_plots, 7)) # Размер фигуры зависит от кол-ва графиков

    # График PCA
    ax1 = plt.subplot(1, num_plots, 1)
    scatter1 = ax1.scatter(results_df['pca_1'], results_df['pca_2'], # Берем первые 2 компоненты
                           c=colors, alpha=0.7, s=50, cmap='tab10') # Используем цвета секторов
    ax1.set_title(f'PCA сжатие эмбеддингов (Компоненты 1 и 2)\nОбъясн. дисп.={np.sum(pca.explained_variance_ratio_[:2]):.3f}')
    ax1.set_xlabel('Главная Компонента 1')
    ax1.set_ylabel('Главная Компонента 2')
    ax1.grid(True)
    # Добавляем легенду для секторов, если они есть
    if ticker_to_sector and results_df['sector'].nunique() <= 10 : # Ограничим кол-во для читаемости
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=sector,
                              markerfacecolor=color, markersize=8) for sector, color in sector_color_map.items()]
        ax1.legend(handles=handles, title="Сектора", bbox_to_anchor=(1.05, 1), loc='upper left')


    # График UMAP (если есть)
    if embeddings_umap is not None:
        ax2 = plt.subplot(1, num_plots, 2)
        scatter2 = ax2.scatter(results_df['umap_1'], results_df['umap_2'], # Берем первые 2 компоненты
                               c=colors, alpha=0.7, s=50, cmap='tab10')
        ax2.set_title('UMAP сжатие эмбеддингов (Компоненты 1 и 2)')
        ax2.set_xlabel('UMAP Компонента 1')
        ax2.set_ylabel('UMAP Компонента 2')
        ax2.grid(True)
        # Добавляем легенду для секторов
        if ticker_to_sector and results_df['sector'].nunique() <= 10:
             handles = [plt.Line2D([0], [0], marker='o', color='w', label=sector,
                                   markerfacecolor=color, markersize=8) for sector, color in sector_color_map.items()]
             ax2.legend(handles=handles, title="Сектора", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Оставляем место справа для легенды, если она есть

    # Сохранение или отображение
    output_viz_path = os.path.join(base_path, 'embeddings_visualization.png')
    try:
        plt.savefig(output_viz_path, dpi=150, bbox_inches='tight')
        print(f"\nГрафик визуализации эмбеддингов сохранен: {output_viz_path}")
    except Exception as e:
        print(f"Ошибка сохранения графика визуализации: {e}")

    # В Colab можно и показать
    # plt.show()
    plt.close() # Закрываем фигуру

except ImportError:
    print("\nБиблиотеки Matplotlib или Seaborn не найдены. Визуализация невозможна.")
    print("Установите: pip install matplotlib seaborn")
except Exception as e:
    print(f"\nОшибка при построении графика визуализации: {e}")
