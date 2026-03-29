# Luggage Brand Intelligence Dashboard
**Moonshot AI Agent Internship Assignment**

Competitive intelligence dashboard for luggage brands on Amazon India.
Scrapes product + review data, runs sentiment analysis, and presents insights via an interactive Streamlit dashboard.

---

# Project Structure

```
luggage_intel/
├── data/                     ← Auto-generated CSVs go here
│   ├── products.csv
│   ├── reviews.csv
│   ├── reviews_sentiment.csv
│   └── brand_summary.csv
├── src/
│   ├── scraper.py            ← Step 1: Scrape Amazon India
│   ├── sentiment.py          ← Step 2: Sentiment analysis
│   └── dashboard.py          ← Step 3: Streamlit dashboard
├── assets/                   ← Optional logo/images
├── requirements.txt
└── README.md
```

---

##  Setup

```bash
# 1. Clone / download the project
cd luggage_intel

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download TextBlob corpora
python -m textblob.download_corpora
```

---

## Run Order

### Step 1 — Scrape Data
```bash
python src/scraper.py
```
Outputs: `data/products.csv`, `data/reviews.csv`

### Step 2 — Sentiment Analysis
```bash
python src/sentiment.py
```
Outputs: `data/reviews_sentiment.csv`, `data/brand_summary.csv`

### Step 3 — Launch Dashboard
```bash
streamlit run src/dashboard.py
```
Opens at: `http://localhost:8501`

---

##  Brands Covered
- Safari
- Skybags
- American Tourister
- VIP

---

##  Dashboard Features
- Overview metrics (brands, products, reviews, avg sentiment, avg price)
- Brand comparison (price, discount, rating, sentiment)
- Sentiment breakdown (positive/negative/neutral per brand)
- Pricing insights (price vs rating scatter, price spread)
- Aspect-level sentiment radar (wheels, handle, zipper, durability etc.)
- Product drilldown with top positive/negative reviews
- Agent Insights — 5 auto-generated non-obvious conclusions

---

##  Sentiment Methodology
- **Library:** TextBlob (polarity scoring -1 to +1)
- **Labels:** Positive (>0.1), Neutral (-0.1 to 0.1), Negative (<-0.1)
- **Aspect-level:** Sentence-level polarity for 8 product aspects

---

##  Limitations
- Amazon may block scraping with CAPTCHA; re-run or use a VPN if blocked
- Review count limited to 10 per product for speed; increase `REVIEWS_PER_PRODUCT` in scraper.py
- TextBlob sentiment is lexicon-based; for higher accuracy, swap with an LLM

---

##  Future Improvements
- Use GPT/Claude API for richer theme extraction
- Add time-series sentiment trend
- Detect fake/suspicious reviews
- Deploy on Streamlit Cloud or Render
