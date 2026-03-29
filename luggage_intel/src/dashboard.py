"""
STEP 3 — Streamlit Dashboard with LLM Sentiment Analysis
 LLM-powered sentiment detection (Groq + LangGraph)
 Sarcasm detection and nuanced sentiment understanding
 Dynamic filters on all charts
 Data-driven insights from LLM analysis
Run: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Luggage Brand Intelligence",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# LOAD DATA
@st.cache_data
def load_data():
    products = pd.read_csv("data/products.csv")
    reviews  = pd.read_csv("data/reviews_sentiment.csv")
    summary  = pd.read_csv("data/brand_summary.csv")

    # Normalize column names
    products.columns = products.columns.str.strip().str.lower()
    reviews.columns = reviews.columns.str.strip().str.lower()
    summary.columns = summary.columns.str.strip().str.lower()

    # Fix column name inconsistencies
    if "body" in reviews.columns and "review_body" not in reviews.columns:
        reviews.rename(columns={"body": "review_body"}, inplace=True)
    if "cbrand" in products.columns and "brand" not in products.columns:
        products.rename(columns={"cbrand": "brand"}, inplace=True)

    # Convert numeric columns
    for col in ["price", "mrp", "discount_pct", "rating", "review_count"]:
        if col in products.columns:
            products[col] = pd.to_numeric(products[col], errors="coerce")
    for col in ["rating", "sentiment_score"]:
        if col in reviews.columns:
            reviews[col] = pd.to_numeric(reviews[col], errors="coerce")

    # Fill missing titles
    if "title" in products.columns:
        products["title"] = products.apply(
            lambda r: f"{r['brand']} Product ({r['asin']})" if pd.isna(r["title"]) or r["title"] == "N/A" else r["title"],
            axis=1
        )

    return products, reviews, summary

products, reviews, summary = load_data()
BRANDS = sorted(products["brand"].unique().tolist())
COLORS = px.colors.qualitative.Set2


st.sidebar.title("🧳 Filters")

selected_brands = st.sidebar.multiselect(
    "Select Brands", BRANDS, default=BRANDS
)

price_min = int(products["price"].min(skipna=True) or 0)
price_max = int(products["price"].max(skipna=True) or 20000)
price_range = st.sidebar.slider("Price Range (₹)", price_min, price_max, (price_min, price_max))

min_rating = st.sidebar.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.5)

sentiment_filter = st.sidebar.selectbox(
    "Sentiment Filter (Reviews)", ["All", "Positive", "Neutral", "Negative"]
)

# ── APPLY FILTERS TO ALL DATA ──
fp = products[
    (products["brand"].isin(selected_brands)) &
    (products["price"].between(*price_range)) &
    (products["rating"].fillna(0) >= min_rating)
].copy()

fr = reviews[reviews["brand"].isin(selected_brands)].copy()
if sentiment_filter != "All":
    fr = fr[fr["sentiment_label"] == sentiment_filter]

fs = summary[summary["brand"].isin(selected_brands)].copy()


# HEADER
st.title("🧳 Luggage Brand Intelligence Dashboard")
st.caption("Amazon India — Competitive Analysis | Safari · Skybags · American Tourister · VIP")
st.divider()


# 1. OVERVIEW METRICS

st.subheader("📊 Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Brands Tracked",    len(selected_brands))
c2.metric("Products Analyzed", len(fp))
c3.metric("Reviews Analyzed",  len(fr))
avg_sent = fr["sentiment_score"].mean() if not fr.empty else 0
c4.metric("Avg Sentiment", f"{avg_sent:.2f}")
avg_price = fp["price"].mean() if not fp.empty else 0
c5.metric("Avg Price (₹)", f"₹{avg_price:,.0f}")
st.divider()


# 2. BRAND COMPARISON (fully dynamic)

st.subheader("🏆 Brand Comparison")

col1, col2 = st.columns(2)
with col1:
    avg_price_df = fp.groupby("brand")["price"].mean().reset_index()
    fig = px.bar(avg_price_df, x="brand", y="price", color="brand",
                 color_discrete_sequence=COLORS,
                 title="Average Selling Price by Brand (₹)",
                 labels={"price": "Avg Price (₹)", "brand": "Brand"})
    fig.update_layout(showlegend=False, xaxis_title="Brand", yaxis_title="Avg Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    avg_disc_df = fp.groupby("brand")["discount_pct"].mean().reset_index()
    fig2 = px.bar(avg_disc_df, x="brand", y="discount_pct", color="brand",
                  color_discrete_sequence=COLORS,
                  title="Average Discount % by Brand",
                  labels={"discount_pct": "Avg Discount (%)", "brand": "Brand"})
    fig2.update_layout(showlegend=False, xaxis_title="Brand", yaxis_title="Avg Discount (%)")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    avg_rating_df = fp.groupby("brand")["rating"].mean().reset_index()
    fig3 = px.bar(avg_rating_df, x="brand", y="rating", color="brand",
                  color_discrete_sequence=COLORS,
                  title="Average Star Rating by Brand",
                  labels={"rating": "Avg Rating", "brand": "Brand"},
                  range_y=[0, 5])
    fig3.update_layout(showlegend=False, xaxis_title="Brand", yaxis_title="Avg Rating")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    all_reviews_unfiltered = reviews[reviews["brand"].isin(selected_brands)]
    avg_sent_df = all_reviews_unfiltered.groupby("brand")["sentiment_score"].mean().reset_index()
    fig4 = px.bar(avg_sent_df, x="brand", y="sentiment_score", color="brand",
                  color_discrete_sequence=COLORS,
                  title="Average Sentiment Score by Brand",
                  labels={"sentiment_score": "Sentiment (-1 to 1)", "brand": "Brand"})
    fig4.update_layout(showlegend=False, xaxis_title="Brand", yaxis_title="Sentiment Score")
    st.plotly_chart(fig4, use_container_width=True)

# ── Scorecard Table ──
st.subheader("📋 Side-by-Side Brand Scorecard")
brand_stats = fp.groupby("brand").agg(
    avg_price=("price", "mean"),
    avg_discount=("discount_pct", "mean"),
    avg_rating=("rating", "mean"),
    product_count=("asin", "count"),
).reset_index()

sent_agg = all_reviews_unfiltered.groupby("brand")["sentiment_score"].mean().reset_index()
sent_agg.columns = ["brand", "avg_sentiment"]
brand_stats = brand_stats.merge(sent_agg, on="brand", how="left")

if not fs.empty and "top_praises" in fs.columns:
    brand_stats = brand_stats.merge(fs[["brand", "top_praises", "top_complaints"]], on="brand", how="left")

display_stats = brand_stats.copy()
display_stats["avg_price"]     = display_stats["avg_price"].map("₹{:,.0f}".format)
display_stats["avg_discount"]  = display_stats["avg_discount"].map("{:.1f}%".format)
display_stats["avg_rating"]    = display_stats["avg_rating"].map("{:.2f} ⭐".format)
display_stats["avg_sentiment"] = display_stats["avg_sentiment"].map("{:.3f}".format)
display_stats = display_stats.rename(columns={
    "brand": "Brand", "avg_price": "Avg Price", "avg_discount": "Avg Discount",
    "avg_rating": "Avg Rating", "product_count": "Products",
    "avg_sentiment": "Sentiment", "top_praises": "Top Praises",
    "top_complaints": "Top Complaints"
})
st.dataframe(display_stats, use_container_width=True)
st.divider()


# 3. WHY IS A BRAND WINNING?
st.subheader("🥇 Why Is A Brand Winning?")

if not brand_stats.empty:
    # Compute a composite win score
    norm = brand_stats.copy()
    for col in ["avg_rating", "avg_sentiment"]:
        rng = norm[col].max() - norm[col].min()
        norm[col + "_n"] = (norm[col] - norm[col].min()) / rng if rng > 0 else 0.5

    # Lower price = better value (invert)
    rng = norm["avg_price"].max() - norm["avg_price"].min()
    norm["price_n"] = 1 - ((norm["avg_price"] - norm["avg_price"].min()) / rng) if rng > 0 else 0.5

    # Lower discount dependency = more organic demand
    rng = norm["avg_discount"].max() - norm["avg_discount"].min()
    norm["disc_n"] = 1 - ((norm["avg_discount"] - norm["avg_discount"].min()) / rng) if rng > 0 else 0.5

    norm["win_score"] = (
        norm["avg_rating_n"]    * 0.35 +
        norm["avg_sentiment_n"] * 0.35 +
        norm["price_n"]         * 0.15 +
        norm["disc_n"]          * 0.15
    )

    winner = norm.loc[norm["win_score"].idxmax()]
    winner_brand = winner["brand"]

    # Winner card
    wc1, wc2, wc3 = st.columns([1, 2, 1])
    with wc2:
        st.success(f"### 🏆 {winner_brand} is currently WINNING")

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Avg Rating",    f"{winner['avg_rating']:.2f} ⭐")
    w2.metric("Avg Sentiment", f"{winner['avg_sentiment']:.3f}")
    w3.metric("Avg Price",     f"₹{winner['avg_price']:,.0f}")
    w4.metric("Avg Discount",  f"{winner['avg_discount']:.1f}%")

    # Explanation
    st.markdown("#### 📝 Why?")
    reasons = []

    # Highest rating?
    if norm.loc[norm["avg_rating"].idxmax(), "brand"] == winner_brand:
        reasons.append(f"⭐ **Highest average star rating** ({winner['avg_rating']:.2f}/5) among all brands — customers consistently rate it best.")

    # Highest sentiment?
    if norm.loc[norm["avg_sentiment"].idxmax(), "brand"] == winner_brand:
        reasons.append(f"💚 **Best customer sentiment** ({winner['avg_sentiment']:.3f}) — reviews are overwhelmingly positive.")

    # Lowest discount dependency?
    if norm.loc[norm["avg_discount"].idxmin(), "brand"] == winner_brand:
        reasons.append(f"💰 **Least discount-dependent** ({winner['avg_discount']:.1f}% avg discount) — demand is organic, not driven by heavy price cuts.")

    # Best value (rating/price)?
    norm["value_score"] = norm["avg_rating"] / (norm["avg_price"] / 1000 + 0.1)
    if norm.loc[norm["value_score"].idxmax(), "brand"] == winner_brand:
        reasons.append(f"🎯 **Best value-for-money** — highest rating relative to its price point.")

    if not reasons:
        reasons.append(f"📊 **{winner_brand}** leads on a composite score of rating, sentiment, pricing, and discount dependency.")

    for r in reasons:
        st.markdown(f"- {r}")

    # All brands ranked
    st.markdown("####  Brand Win Score Ranking")
    rank_df = norm[["brand", "win_score", "avg_rating", "avg_sentiment", "avg_price", "avg_discount"]].sort_values("win_score", ascending=False).reset_index(drop=True)
    rank_df.index += 1
    rank_df.columns = ["Brand", "Win Score", "Avg Rating", "Avg Sentiment", "Avg Price (₹)", "Avg Discount (%)"]
    rank_df["Win Score"] = rank_df["Win Score"].map("{:.3f}".format)
    rank_df["Avg Rating"] = rank_df["Avg Rating"].map("{:.2f}".format)
    rank_df["Avg Sentiment"] = rank_df["Avg Sentiment"].map("{:.3f}".format)
    rank_df["Avg Price (₹)"] = rank_df["Avg Price (₹)"].map("₹{:,.0f}".format)
    rank_df["Avg Discount (%)"] = rank_df["Avg Discount (%)"].map("{:.1f}%".format)
    st.dataframe(rank_df, use_container_width=True)

st.divider()

# 4. SENTIMENT ANALYSIS

st.subheader(" Sentiment Breakdown")

col5, col6 = st.columns(2)
with col5:
    sent_counts = fr.groupby(["brand", "sentiment_label"]).size().reset_index(name="count")
    fig5 = px.bar(sent_counts, x="brand", y="count", color="sentiment_label",
                  barmode="group", title="Sentiment Distribution by Brand",
                  color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"})
    fig5.update_layout(xaxis_title="Brand", yaxis_title="Review Count")
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    fig6 = px.box(fr, x="brand", y="sentiment_score", color="brand",
                  color_discrete_sequence=COLORS,
                  title="Sentiment Score Distribution by Brand")
    fig6.update_layout(showlegend=False, xaxis_title="Brand", yaxis_title="Sentiment Score")
    st.plotly_chart(fig6, use_container_width=True)

st.divider()


# 4.5. LLM-POWERED SENTIMENT INSIGHTS

st.subheader(" LLM-Powered Sentiment Analysis")
st.caption("Powered by Groq + LangGraph — Detects sarcasm, nuance, and context-aware sentiment")

lm_col1, lm_col2, lm_col3 = st.columns(3)

# Check if sarcasm_detected column exists
has_sarcasm = "sarcasm_detected" in fr.columns

with lm_col1:
    total_reviews = len(fr)
    sarcastic = (fr["sarcasm_detected"] == True).sum() if has_sarcasm else 0
    sarcasm_pct = (sarcastic / total_reviews * 100) if total_reviews > 0 else 0
    lm_col1.metric("Sarcasm Detected", f"{sarcastic} ({sarcasm_pct:.1f}%)")

with lm_col2:
    avg_sentiment = fr["sentiment_score"].mean() if not fr.empty else 0
    lm_col2.metric("Avg LLM Sentiment", f"{avg_sentiment:.3f}")

with lm_col3:
    mixed_sentiment = (
        ((fr["sentiment_score"] > -0.2) & (fr["sentiment_score"] < 0.2)).sum()
        if not fr.empty else 0
    )
    mixed_pct = (mixed_sentiment / total_reviews * 100) if total_reviews > 0 else 0
    lm_col3.metric("Nuanced/Mixed", f"{mixed_sentiment} ({mixed_pct:.1f}%)")

st.write("")

# Sarcasm detection by brand
if has_sarcasm and not fr.empty:
    sarcasm_by_brand = fr.groupby("brand")["sarcasm_detected"].apply(
        lambda x: (x == True).sum()
    ).reset_index(name="sarcasm_count")
    total_by_brand = fr.groupby("brand").size().reset_index(name="total")
    sarcasm_by_brand = sarcasm_by_brand.merge(total_by_brand, on="brand")
    sarcasm_by_brand["sarcasm_pct"] = (sarcasm_by_brand["sarcasm_count"] / sarcasm_by_brand["total"] * 100).round(1)

    col_sarc1, col_sarc2 = st.columns(2)
    
    with col_sarc1:
        fig_sarc = px.bar(sarcasm_by_brand, x="brand", y="sarcasm_count", color="brand",
                         color_discrete_sequence=COLORS,
                         title="Sarcasm Detected by Brand (Count)",
                         labels={"sarcasm_count": "Sarcastic Reviews", "brand": "Brand"})
        fig_sarc.update_layout(showlegend=False, xaxis_title="Brand", yaxis_title="Count")
        st.plotly_chart(fig_sarc, use_container_width=True)
    
    with col_sarc2:
        fig_sarc_pct = px.bar(sarcasm_by_brand, x="brand", y="sarcasm_pct", color="brand",
                             color_discrete_sequence=COLORS,
                             title="Sarcasm % of Reviews",
                             labels={"sarcasm_pct": "Sarcasm %", "brand": "Brand"})
        fig_sarc_pct.update_layout(showlegend=False, xaxis_title="Brand", yaxis_title="% Sarcastic")
        st.plotly_chart(fig_sarc_pct, use_container_width=True)

# Show example sarcastic reviews if available
if has_sarcasm:
    sarcastic_reviews = fr[fr["sarcasm_detected"] == True].head(5)
    if not sarcastic_reviews.empty:
        with st.expander("📌 Example Sarcastic Reviews Detected"):
            for idx, rev in sarcastic_reviews.iterrows():
                brand = rev.get("brand", "Unknown")
                score = rev.get("sentiment_score", 0)
                label = rev.get("sentiment_label", "N/A")
                text = rev.get("review_body", rev.get("body", "No text"))[:300]
                
                col = "🟡" if label == "Neutral" else ("🟢" if label == "Positive" else "🔴")
                st.write(f"**{col} {brand}** | Score: {score:.3f} | Label: {label}")
                st.info(f"_{text}_")

st.divider()


# 5. PRICING INSIGHTS

st.subheader(" Pricing Insights")

col7, col8 = st.columns(2)
with col7:
    fig7 = px.scatter(fp, x="price", y="rating", color="brand",
                      size="discount_pct", hover_data=["title", "discount_pct"],
                      color_discrete_sequence=COLORS,
                      title="Price vs Rating (bubble size = discount %)",
                      labels={"price": "Price (₹)", "rating": "Rating"})
    st.plotly_chart(fig7, use_container_width=True)

with col8:
    fig8 = px.box(fp, x="brand", y="price", color="brand",
                  color_discrete_sequence=COLORS,
                  title="Price Spread by Brand (₹)",
                  labels={"price": "Price (₹)", "brand": "Brand"})
    fig8.update_layout(showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# 6. ASPECT-LEVEL SENTIMENT RADAR
# ─────────────────────────────────────────────
aspect_cols = [c for c in summary.columns if c.startswith("aspect_")]
if aspect_cols:
    st.subheader("🔍 Aspect-Level Sentiment Radar")
    radar_fs = summary[summary["brand"].isin(selected_brands)]
    aspects = [c.replace("aspect_", "") for c in aspect_cols]
    fig9 = go.Figure()
    for _, row in radar_fs.iterrows():
        vals = [float(row[c]) if pd.notna(row[c]) else 0 for c in aspect_cols]
        fig9.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=aspects + [aspects[0]],
            fill="toself",
            name=row["brand"]
        ))
    fig9.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
        title="Aspect Sentiment Comparison (wheels, zipper, durability etc.)"
    )
    st.plotly_chart(fig9, use_container_width=True)
    st.divider()

# ─────────────────────────────────────────────
# 7. ANOMALY DETECTION
# ─────────────────────────────────────────────
st.subheader("⚠️ Anomaly Detection")
st.caption("Brands with high ratings but negative sentiment on key aspects — possible hidden quality issues")

anomalies = []
dur_col = "aspect_durability"
if dur_col in summary.columns:
    for _, row in summary.iterrows():
        if row["brand"] not in selected_brands:
            continue
        brand_prods = products[products["brand"] == row["brand"]]
        avg_star = brand_prods["rating"].mean()
        dur_sent = row.get(dur_col, None)
        if pd.notna(avg_star) and pd.notna(dur_sent):
            if avg_star >= 4.0 and dur_sent < 0:
                anomalies.append({
                    "Brand": row["brand"],
                    "Avg Star Rating": f"{avg_star:.2f} ⭐",
                    "Durability Sentiment": f"{dur_sent:.3f} ❌",
                    "Flag": "High rating but negative durability reviews — investigate!"
                })

if anomalies:
    st.warning(f"⚠️ {len(anomalies)} anomaly(ies) detected!")
    st.dataframe(pd.DataFrame(anomalies), use_container_width=True)
else:
    st.success("✅ No anomalies detected — ratings and sentiment are consistent.")

st.divider()


# 8. PRODUCT DRILLDOWN

st.subheader("🔎 Product Drilldown")

drill_brand = st.selectbox("Pick a brand", selected_brands if selected_brands else BRANDS)
brand_prods = fp[fp["brand"] == drill_brand].sort_values("rating", ascending=False)

if not brand_prods.empty:
    selected_product = st.selectbox("Pick a product", brand_prods["title"].tolist())
    prod_row = brand_prods[brand_prods["title"] == selected_product].iloc[0]

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Price",    f"₹{prod_row['price']:,.0f}" if pd.notna(prod_row["price"]) else "N/A")
    pc2.metric("MRP",      f"₹{prod_row['mrp']:,.0f}"   if pd.notna(prod_row["mrp"])   else "N/A")
    pc3.metric("Discount", f"{prod_row['discount_pct']:.0f}%" if pd.notna(prod_row["discount_pct"]) else "N/A")
    pc4.metric("Rating",   f"{prod_row['rating']} ⭐"   if pd.notna(prod_row["rating"]) else "N/A")

    prod_reviews = reviews[reviews["asin"] == prod_row["asin"]]
    if not prod_reviews.empty:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("✅ **Top Positive Reviews**")
            pos = prod_reviews[prod_reviews["sentiment_label"] == "Positive"]["review_body"].head(3).tolist()
            for r in pos:
                st.success(str(r)[:300])
            if not pos:
                st.info("No positive reviews found.")
        with r2:
            st.markdown("❌ **Top Negative Reviews**")
            neg = prod_reviews[prod_reviews["sentiment_label"] == "Negative"]["review_body"].head(3).tolist()
            for r in neg:
                st.error(str(r)[:300])
            if not neg:
                st.info("No negative reviews found.")
    else:
        st.info("No reviews found for this product.")

st.divider()


# 9. AGENT INSIGHTS (fully data-driven)

st.subheader("🤖 Agent Insights — 5 Non-Obvious Conclusions")
st.caption("Auto-generated from scraped data")

insights = []

if not brand_stats.empty:
    # 1. Value-for-money winner
    brand_stats["value_score"] = brand_stats["avg_rating"] / (brand_stats["avg_price"] / 1000 + 0.1)
    best_val   = brand_stats.loc[brand_stats["value_score"].idxmax()]
    worst_val  = brand_stats.loc[brand_stats["value_score"].idxmin()]
    insights.append(
        f"💡 **{best_val['brand']}** offers the best value-for-money "
        f"(rating {best_val['avg_rating']:.2f} at ₹{best_val['avg_price']:,.0f}) "
        f"while **{worst_val['brand']}** is the weakest value proposition."
    )

    # 2. Discount dependency
    most_disc  = brand_stats.loc[brand_stats["avg_discount"].idxmax()]
    least_disc = brand_stats.loc[brand_stats["avg_discount"].idxmin()]
    insights.append(
        f"📉 **{most_disc['brand']}** relies most heavily on discounting "
        f"({most_disc['avg_discount']:.1f}% avg) to drive sales — this may signal "
        f"weak organic demand. **{least_disc['brand']}** needs the least discounting "
        f"({least_disc['avg_discount']:.1f}%)."
    )

    # 3. Sentiment vs rating mismatch
    merged_check = brand_stats.merge(sent_agg, on="brand", how="left")
    if "avg_sentiment" in merged_check.columns:
        merged_check["mismatch"] = merged_check["avg_rating"] - (merged_check["avg_sentiment"] * 5)
        mismatch_brand = merged_check.loc[merged_check["mismatch"].abs().idxmax(), "brand"]
        insights.append(
            f"⚠️ **{mismatch_brand}** shows the biggest gap between star ratings and "
            f"actual review sentiment — customers may be rating out of habit rather than satisfaction. "
            f"Deep-dive into reviews recommended."
        )

    # 4. Most reviewed = market leader
    rcount = reviews.groupby("brand").size().reset_index(name="total_reviews")
    if not rcount.empty:
        most_rev = rcount.loc[rcount["total_reviews"].idxmax()]
        insights.append(
            f"📣 **{most_rev['brand']}** has the highest review volume "
            f"({most_rev['total_reviews']} reviews scraped) — indicating stronger market "
            f"penetration and higher customer engagement than competitors."
        )

    # 5. Premium vs budget positioning
    sorted_price = brand_stats.sort_values("avg_price", ascending=False)
    premium = sorted_price.iloc[0]["brand"]
    budget  = sorted_price.iloc[-1]["brand"]
    premium_price = sorted_price.iloc[0]["avg_price"]
    budget_price  = sorted_price.iloc[-1]["avg_price"]
    insights.append(
        f"🏷️ Clear market segmentation: **{premium}** is positioned as the premium brand "
        f"(avg ₹{premium_price:,.0f}) while **{budget}** targets the budget segment "
        f"(avg ₹{budget_price:,.0f}) — a price gap of ₹{premium_price - budget_price:,.0f}. "
        f"These brands are not directly competing on price."
    )

for i, insight in enumerate(insights, 1):
    st.markdown(f"**{i}.** {insight}")

st.divider()
st.caption("Built for Moonshot AI Agent Internship Assignment · Amazon India Luggage Intelligence · 2025")