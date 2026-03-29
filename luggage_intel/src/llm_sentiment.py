"""
STEP 2 — LLM-Powered Sentiment Analysis with Groq & LangGraph
Reads data/reviews.csv → LLM sentiment scores + aspect analysis
Output: data/reviews_sentiment.csv, data/brand_summary.csv
Handles sarcasm, nuance, and context-aware sentiment detection
Run: python src/llm_sentiment.py
"""

import pandas as pd
import json
import os
from typing import TypedDict, Annotated
from collections import Counter
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.types import Command

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("⚠️  GROQ_API_KEY environment variable not set. Please set it and try again.")

# Fast & capable for sentiment analysis with Groq
MODEL = "llama-3.3-70b-versatile"


class ReviewState(TypedDict):
    """Graph state for processing a single review."""
    review_text: str
    brand: str
    asin: str
    overall_sentiment: dict  
    aspect_sentiments: dict 
    sarcasm_detected: bool
    overall_failed: bool
    error: str | None



llm = ChatGroq(
    model=MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.3,
    timeout=30
)

def node_extract_overall_sentiment(state: ReviewState) -> Command[ReviewState]:
    """
    Use LLM to extract:
    - Overall sentiment score (-1 to 1)
    - Sentiment label (Positive/Neutral/Negative)
    - Sarcasm detection
    """
    prompt = PromptTemplate.from_template("""
You are an expert sentiment analyzer for product reviews. Analyze the following review and detect sarcasm or nuanced sentiment.

Review: "{review_text}"
Brand: {brand}

Respond in JSON format ONLY:
{{
    "sentiment_score": <float between -1.0 (very negative) and 1.0 (very positive)>,
    "sentiment_label": "<Positive|Neutral|Negative>",
    "sarcasm_detected": <boolean>,
    "reasoning": "<brief explanation of sentiment decision>"
}}

Focus on:
1. Detect subtle sarcasm (e.g., "Great product!" said about a broken item)
2. Consider context and tone, not just keywords
3. Handle mixed sentiments (e.g., "Good design but poor quality" = mixed/neutral)
    """)
    
    try:
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({
            "review_text": state["review_text"][:2000],  # Limit token usage
            "brand": state["brand"]
        })
       
        raw_score = result.get("sentiment_score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        score = max(-1.0, min(1.0, score))

        raw_label = str(result.get("sentiment_label", "Neutral")).strip().lower()
        label_map = {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
        }
        label = label_map.get(raw_label)
        if label is None:
            if score >= 0.1:
                label = "Positive"
            elif score <= -0.1:
                label = "Negative"
            else:
                label = "Neutral"
        
        state["overall_sentiment"] = {
            "score": score,
            "label": label,
        }
        state["sarcasm_detected"] = result.get("sarcasm_detected", False)
        state["overall_failed"] = False
        
    except Exception as e:
        state["error"] = f"Overall sentiment extraction failed: {str(e)}"
        state["overall_sentiment"] = {"score": 0.0, "label": "Neutral"}
        state["sarcasm_detected"] = False
        state["overall_failed"] = True
    
    return state


def node_extract_aspect_sentiments(state: ReviewState) -> Command[ReviewState]:
    """
    Extract sentiment scores for specific aspects:
    wheels, handle, material, zipper, size, durability, price, looks
    """
    aspects = ["wheels", "handle", "material", "zipper", "size", "durability", "price", "looks"]
    
    prompt = PromptTemplate.from_template("""
Analyze the following product review for luggage and extract sentiment for each aspect.
If an aspect is not mentioned, return null.

Review: "{review_text}"

Respond in JSON format ONLY. For each aspect, provide a sentiment score from -1.0 to 1.0:
{{
    "aspects": {{
        "wheels": <null or float>,
        "handle": <null or float>,
        "material": <null or float>,
        "zipper": <null or float>,
        "size": <null or float>,
        "durability": <null or float>,
        "price": <null or float>,
        "looks": <null or float>
    }}
}}

Guidelines:
- Only score aspects explicitly mentioned or clearly implied in the review
- Return null for unmentioned aspects
- Consider sarcasm and context
    """)
    
    try:
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"review_text": state["review_text"][:2000]})
        
        aspect_scores = result.get("aspects", {})
        # Clean up nulls and round to 3 decimals
        state["aspect_sentiments"] = {
            k: round(v, 3) if v is not None else None 
            for k, v in aspect_scores.items()
        }
        
    except Exception as e:
        existing_error = state.get("error")
        aspect_error = f"Aspect extraction failed: {str(e)}"
        state["error"] = f"{existing_error} | {aspect_error}" if existing_error else aspect_error
        state["aspect_sentiments"] = {aspect: None for aspect in aspects}
    
    return state


def build_sentiment_graph():
    """Construct the LangGraph workflow for sentiment analysis."""
    graph = StateGraph(ReviewState)
    
    # Add nodes
    graph.add_node("extract_overall", node_extract_overall_sentiment)
    graph.add_node("extract_aspects", node_extract_aspect_sentiments)
    
    # Define edges: sequential processing
    graph.add_edge("extract_overall", "extract_aspects")
    graph.add_edge("extract_aspects", END)
    
    # Set entry point
    graph.set_entry_point("extract_overall")
    
    return graph.compile()


def analyze_reviews():
    """
    Main pipeline:
    1. Load reviews
    2. Process each review through LangGraph sentiment workflow
    3. Save sentiment-enriched reviews
    4. Generate brand-level summary
    """
    
    print("[+] Loading reviews...")
    df = pd.read_csv("data/reviews.csv")
    df["body"] = df["body"].fillna("")
    
    print(f"[+] Loaded {len(df)} reviews")
    print("[+] Building sentiment analysis graph...")
    
    sentiment_graph = build_sentiment_graph()
    
    # Initialize result columns
    df["sentiment_score"] = 0.0
    df["sentiment_label"] = "Neutral"
    df["sarcasm_detected"] = False
    
    aspect_cols = ["wheels", "handle", "material", "zipper", "size", "durability", "price", "looks"]
    for aspect in aspect_cols:
        df[f"aspect_{aspect}"] = None
    
    print("\n[+] Processing reviews by brand...")
    total_processed = 0
    overall_failures = 0
    sample_errors = []

    for brand, group_df in df.groupby("brand"):
        print(f"\n  Processing brand: {brand} ({len(group_df)} reviews)")
        
        for idx, row in group_df.iterrows():
            review_text = str(row["body"]) if pd.notna(row["body"]) else ""
            
            # Skip empty reviews
            if not review_text.strip():
                continue
            
            # Create state for this review
            state = ReviewState(
                review_text=review_text,
                brand=brand,
                asin=row.get("asin", ""),
                overall_sentiment={},
                aspect_sentiments={},
                sarcasm_detected=False,
                overall_failed=False,
                error=None
            )
            
            # Run through graph
            result = sentiment_graph.invoke(state)
            
            # Update dataframe with results
            df.at[idx, "sentiment_score"] = round(result["overall_sentiment"].get("score", 0.0), 4)
            df.at[idx, "sentiment_label"] = result["overall_sentiment"].get("label", "Neutral")
            df.at[idx, "sarcasm_detected"] = result.get("sarcasm_detected", False)

            total_processed += 1
            if result.get("overall_failed"):
                overall_failures += 1
                if result.get("error") and len(sample_errors) < 3:
                    sample_errors.append(result["error"])
            
            # Update aspect sentiments
            for aspect in aspect_cols:
                aspect_score = result["aspect_sentiments"].get(aspect)
                df.at[idx, f"aspect_{aspect}"] = aspect_score
            
            # Progress indicator
            if (group_df.index.tolist().index(idx) + 1) % 10 == 0:
                print(f"    Processed {group_df.index.tolist().index(idx) + 1}/{len(group_df)} reviews")

    if total_processed > 0 and overall_failures == total_processed:
        error_preview = "\n".join(sample_errors) if sample_errors else "No error details captured."
        raise RuntimeError(
            "All overall sentiment API calls failed, so results would be all Neutral/0. "
            "Please check GROQ_MODEL/API availability. Sample errors:\n" + error_preview
        )
    
    # Save enriched reviews
    print("\n[+] Saving enriched reviews...")
    df.to_csv("data/reviews_sentiment.csv", index=False)
    print("✓ Saved → data/reviews_sentiment.csv")
    
    # Generate brand-level summary
    print("\n[+] Building brand summary...")
    summaries = []
    
    for brand, group in df.groupby("brand"):
        texts = group["body"].tolist()
        
        avg_sentiment = round(group["sentiment_score"].mean(), 3)
        pos_pct = round((group["sentiment_label"] == "Positive").mean() * 100, 1)
        neg_pct = round((group["sentiment_label"] == "Negative").mean() * 100, 1)
        sarcasm_pct = round((group["sarcasm_detected"] == True).mean() * 100, 1)
        
        # Extract most mentioned positive/negative aspects using LLM
        positive_reviews = group[group["sentiment_label"] == "Positive"]["body"].tolist()
        negative_reviews = group[group["sentiment_label"] == "Negative"]["body"].tolist()
        
        top_praises = extract_themes_llm(positive_reviews[:20], "positive") if positive_reviews else "N/A"
        top_complaints = extract_themes_llm(negative_reviews[:20], "negative") if negative_reviews else "N/A"
        
        # Aspect averages
        aspect_avgs = {}
        for aspect in aspect_cols:
            col = f"aspect_{aspect}"
            aspect_avgs[aspect] = round(group[col].dropna().mean(), 3) if col in group.columns else None
        
        summaries.append({
            "brand": brand,
            "total_reviews": len(group),
            "avg_sentiment": avg_sentiment,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "sarcasm_detected_pct": sarcasm_pct,
            "top_praises": top_praises,
            "top_complaints": top_complaints,
            **{f"aspect_{k}": v for k, v in aspect_avgs.items()},
        })
    
    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv("data/brand_summary.csv", index=False)
    print("✓ Saved → data/brand_summary.csv")
    
    # Display summary
    print("\n── Brand Sentiment Summary (LLM-Powered) ──")
    print(df_summary[["brand", "avg_sentiment", "positive_pct", "negative_pct", "sarcasm_detected_pct"]].to_string(index=False))
    print("\n✓ Done! Run dashboard.py next.")


def extract_themes_llm(review_texts, sentiment_type):
    """
    Use LLM to extract top themes/phrases from reviews.
    sentiment_type: 'positive' or 'negative'
    """
    if not review_texts:
        return "N/A"
    
    combined_text = " ".join(review_texts[:20])[:3000]  # Limit input
    
    prompt = PromptTemplate.from_template("""
Extract the top 5 most common positive or negative themes/phrases from these product reviews.
Type: {sentiment_type}

Reviews: {reviews}

Respond in JSON format ONLY:
{{"themes": ["theme1", "theme2", "theme3", "theme4", "theme5"]}}

Be concise and specific to luggage/travel products.
    """)
    
    try:
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({
            "reviews": combined_text,
            "sentiment_type": sentiment_type
        })
        themes = result.get("themes", [])
        return ", ".join(themes[:5]) if themes else "N/A"
    except Exception as e:
        return f"Error: {str(e)[:50]}"


if __name__ == "__main__":
    print("=" * 60)
    print("LLM-Powered Sentiment Analysis with Groq & LangGraph")
    print("=" * 60)
    analyze_reviews()
