from yandex_cloud_ml_sdk import YCloudML
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import time
import numpy as np


RATE_LIMIT_DELAY = 0.2  # 200 мс
# Инициализация клиента (замените на ваши параметры)
sdk = YCloudML(
    folder_id="b1gu0dd26bpgkokh9fsk",
    auth="AQVNyz-waLh_zQFPyVCl-LhOUQdwCMNc2bJtfCud"
)


# Инициализация Zero-shot классификатора
classifier = (
  sdk.models.text_classifiers("yandexgpt-lite")
)

# Функция отправки запроса классификации
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
    return {pred.label: pred.confidence for pred in resp.run(text)}

# TaskDescription для тональности
sentiment_task = (
    "Определите тональность новости в диапазоне "
    "двух классов: positive, neutral и negative."
)

# TaskDescription для тем с ключевыми словами
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
    # Классификация тональности
    sent_conf = classify_text(text, sentiment_task, sentiment_labels)
    score = sent_conf["positive"] - sent_conf["negative"]
    # Классификация тем
    topic_conf = classify_text(text, topic_task, topic_labels)
    return score, topic_conf


def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    # df: колонки ['date','title','content']
    daily = []
    grouped = df.groupby("date")
    for date, group in grouped:
        scores = []
        topic_vecs = []
        extreme_count = 0

        for _, row in group.iterrows():
            text = row["title"]
            score, topics = analyze_article(text)
            scores.append(score)
            topic_vecs.append(list(topics.values()))
            if abs(score) > 0.75:
                extreme_count += 1

        avg_sent = sum(scores) / len(scores)
        # news_sentiment_chg будет считаться снаружи, здесь только сырое значение.
        avg_vec = np.mean(topic_vecs, axis=0).tolist()

        # Косинусное расстояние к вектору предыдущего дня — вычисляется снаружи.
        ent = entropy(avg_vec, base=2)
        extreme_rate = extreme_count / len(scores)

        daily.append({
            "date": date,
            "news_sentiment_1d_raw": avg_sent,
            "news_topic_intensities_1d_raw": avg_vec,
            "news_topic_entropy_1d_raw": ent,
            "news_sentiment_extreme_1d_raw": extreme_rate
        })

    return pd.DataFrame(daily)

#Пример использования:
df_all = pd.read_csv("kommersant_business_politics_archive_last_5_years.csv")
features_df = compute_daily_features(df_all)
features_df.to_csv("raw_features.csv", index=False, sep=",", encoding="utf-8")
