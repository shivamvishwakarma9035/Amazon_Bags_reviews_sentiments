"""
STEP 2 — Sentiment Analysis
Reads data/reviews.csv → adds sentiment scores + themes
Output: data/reviews_sentiment.csv, data/brand_summary.csv
Run: python src/sentiment.py
"""

import pandas as pd
import re
from collections import Counter
from textblob import TextBlob 


ASPECTS = {
    "wheels":    ["wheel", "wheels", "rolling", "rolls", "spinner"],
    "handle":    ["handle", "handles", "trolley handle", "telescopic"],
    "material":  ["material", "fabric", "polycarbonate", "hardshell", "soft"],
    "zipper":    ["zipper", "zip", "zippers", "lock"],
    "size":      ["size", "capacity", "spacious", "compact", "fitting"],
    "durability":["durable", "durability", "sturdy", "strong", "broke", "damaged", "cracked"],
    "price":     ["price", "value", "worth", "money", "expensive", "cheap", "affordable"],
    "looks":     ["look", "looks", "design", "color", "colour", "stylish", "appearance"],
}

POSITIVE_WORDS = [
    "good","great","excellent","best","love","perfect","amazing","awesome",
    "fantastic","happy","quality","smooth","durable","sturdy","recommend",
    "satisfied","nice","beautiful","lightweight","spacious","easy"
]
NEGATIVE_WORDS = [
    "bad","worst","poor","broke","damaged","cheap","disappointing","waste",
    "useless","defective","broken","crack","cracked","issue","problem",
    "complaint","return","refund","pathetic","terrible","horrible"
]


def get_sentiment_score(text):
    """Returns polarity score between -1 (negative) and +1 (positive)."""
    if not text or not isinstance(text, str):
        return 0.0
    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 4)

def get_sentiment_label(score):
    if score >= 0.1:  return "Positive"
    if score <= -0.1: return "Negative"
    return "Neutral"

def extract_themes(texts, word_list, top_n=5):
    """Count how many reviews mention words from a list."""
    counter = Counter()
    for text in texts:
        if not isinstance(text, str): continue
        lower = text.lower()
        for word in word_list:
            if word in lower:
                counter[word] += 1
    return [w for w, _ in counter.most_common(top_n)]

def aspect_sentiment(text, keywords):
    """Check sentiment of sentences containing aspect keywords."""
    if not isinstance(text, str): return None
    scores = []
    for sentence in text.split("."):
        if any(kw in sentence.lower() for kw in keywords):
            s = TextBlob(sentence).sentiment.polarity
            scores.append(s)
    return round(sum(scores)/len(scores), 3) if scores else None

# MAIN ANALYSIS

def analyze_reviews():
    print("[+] Loading reviews...")
    df = pd.read_csv("data/reviews.csv")
    df["body"] = df["body"].fillna("")

    # ── Per-review sentiment ──
    print("[+] Scoring sentiment per review...")
    df["sentiment_score"] = df["body"].apply(get_sentiment_score)
    df["sentiment_label"] = df["sentiment_score"].apply(get_sentiment_label)

    # ── Aspect-level sentiment ──
    print("[+] Aspect-level sentiment...")
    for aspect, keywords in ASPECTS.items():
        df[f"aspect_{aspect}"] = df["body"].apply(
            lambda x: aspect_sentiment(x, keywords)
        )

    df.to_csv("data/reviews_sentiment.csv", index=False)
    print("✓ Saved → data/reviews_sentiment.csv")

    print("[+] Building brand summary...")
    summaries = []
    for brand, group in df.groupby("brand"):
        texts = group["body"].tolist()

        avg_sentiment   = round(group["sentiment_score"].mean(), 3)
        pos_pct         = round((group["sentiment_label"] == "Positive").mean() * 100, 1)
        neg_pct         = round((group["sentiment_label"] == "Negative").mean() * 100, 1)
        top_praises     = extract_themes(texts, POSITIVE_WORDS)
        top_complaints  = extract_themes(texts, NEGATIVE_WORDS)

        aspect_avgs = {}
        for aspect in ASPECTS:
            col = f"aspect_{aspect}"
            aspect_avgs[aspect] = round(group[col].dropna().mean(), 3) if col in group else None

        summaries.append({
            "brand":           brand,
            "total_reviews":   len(group),
            "avg_sentiment":   avg_sentiment,
            "positive_pct":    pos_pct,
            "negative_pct":    neg_pct,
            "top_praises":     ", ".join(top_praises),
            "top_complaints":  ", ".join(top_complaints),
            **{f"aspect_{k}": v for k, v in aspect_avgs.items()},
        })

    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv("data/brand_summary.csv", index=False)
    print("✓ Saved → data/brand_summary.csv")
    print("\n── Brand Sentiment Summary ──")
    print(df_summary[["brand","avg_sentiment","positive_pct","negative_pct"]].to_string(index=False))

if __name__ == "__main__":
    analyze_reviews()
    print("\n✓ Done! Run dashboard.py next.")
