import ast
import base64
import html
import math
import os
import re
from io import BytesIO
from collections import Counter
from pathlib import Path
from urllib.parse import quote_plus

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib-cache")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud


st.set_page_config(
    page_title="E-Commerce Review Intelligence System",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


ASPECT_KEYWORDS = {
    "Quality": ["quality", "material", "durable", "defect", "poor", "broken", "good"],
    "Delivery": ["delivery", "shipping", "arrived", "courier", "slow", "late", "fast"],
    "Packaging": ["packaging", "package", "box", "wrapped", "damaged", "sealed"],
    "Price": ["price", "cheap", "expensive", "worth", "value", "affordable"],
    "Customer Service": ["service", "seller", "support", "reply", "refund", "helped"],
    "Function": ["work", "works", "function", "useful", "battery", "performance"],
}

ASPECT_HIGHLIGHT_TERMS = {
    "Quality": ["quality", "material", "durable", "defect", "poor", "broken", "good", "great"],
    "Delivery": ["delivery", "shipping", "arrived", "courier", "slow", "late", "fast"],
    "Packaging": ["packaging", "package", "box", "wrapped", "damaged", "sealed"],
    "Price": ["price", "cheap", "expensive", "worth", "value", "affordable"],
    "Customer Service": ["customer service", "service", "seller", "support", "reply", "refund", "helped"],
    "Function": ["function", "work", "works", "useful", "battery", "performance"],
    "Size and Fit": ["size", "fit", "small", "large", "tight", "loose"],
    "Taste and Flavor": ["taste", "flavor", "flavour", "tasty", "sweet", "bitter"],
    "Design and Appearance": ["design", "appearance", "look", "looks", "color", "style"],
    "Usability": ["use", "using", "easy", "difficult", "simple", "convenient"],
    "Comfort": ["comfort", "comfortable", "soft", "hard"],
}

POSITIVE_WORDS = {
    "good", "great", "excellent", "useful", "worth", "fast", "perfect", "love",
    "helpful", "satisfied", "durable", "affordable", "recommend", "comfortable",
    "amazing", "quick", "secure", "nice", "easy",
}

NEGATIVE_WORDS = {
    "bad", "poor", "slow", "damaged", "broken", "late", "expensive", "refund",
    "terrible", "worst", "disappointed", "defect", "defective", "missing", "leak",
    "fake", "reply", "failed", "weak",
}

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "was", "were", "are", "but",
    "not", "you", "your", "have", "has", "had", "very", "product", "item",
    "customer", "review", "seller", "from", "they", "them", "too", "than",
    "then", "will", "would", "could", "should", "about", "after", "before",
}

SENTIMENT_COLORS = {
    "Positive": "#0f9f6e",
    "Neutral": "#d6a400",
    "Negative": "#df4b4b",
}


st.markdown(
    """
<style>
:root {
    --ink: #111827;
    --muted: #64748b;
    --line: #e2e8f0;
    --panel: #ffffff;
    --soft: #f8fafc;
    --green: #15803d;
    --red: #dc2626;
    --purple: #7c3aed;
    --amber: #b45309;
    --mint: #0f766e;
}

#MainMenu, footer, header, section[data-testid="stSidebar"] {visibility: hidden;}

[data-testid="collapsedControl"] {display: none;}

.stApp {
    background:
        radial-gradient(circle at 6% 8%, rgba(20, 184, 166, 0.18), transparent 28%),
        radial-gradient(circle at 92% 6%, rgba(250, 204, 21, 0.14), transparent 24%),
        linear-gradient(135deg, #eef7f1 0%, #f8fafc 46%, #fff8df 100%);
    color: var(--ink);
}

.block-container {
    max-width: 1280px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.app-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    background: rgba(255,255,255,0.84);
    border: 1px solid rgba(255,255,255,0.92);
    border-radius: 24px;
    padding: 12px 16px 12px 18px;
    box-shadow: 0 20px 55px rgba(15,23,42,0.08);
    margin-bottom: 18px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 260px;
}

.brand-mark {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    color: white;
    font-weight: 900;
    background: linear-gradient(135deg, #16a34a, #064e3b);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.32), 0 10px 22px rgba(15,118,110,0.24);
}

.brand-name {
    font-size: 16px;
    font-weight: 900;
    color: #111827;
}

.brand-sub {
    font-size: 11px;
    color: #64748b;
    margin-top: -2px;
}

.nav-right {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #64748b;
    font-size: 12px;
    font-weight: 750;
}

.nav-pill {
    border-radius: 999px;
    background: #111827;
    color: white;
    padding: 8px 12px;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(15,118,110,0.98) 0%, rgba(19,78,74,0.98) 100%);
    border-right: 0;
}

section[data-testid="stSidebar"] * {
    color: #f9fafb !important;
}

.side-title {
    font-size: 21px;
    font-weight: 800;
    line-height: 1.25;
    margin: 8px 0 2px 0;
}

.side-subtitle {
    color: #cbd5e1 !important;
    font-size: 13px;
    margin-bottom: 18px;
}

.topbar {
    background:
        linear-gradient(120deg, rgba(255,255,255,0.96), rgba(255,255,255,0.82)),
        radial-gradient(circle at 88% 20%, rgba(163, 230, 53, 0.24), transparent 34%);
    border: 1px solid rgba(255,255,255,0.88);
    border-radius: 22px;
    padding: 32px 34px;
    margin-bottom: 18px;
    box-shadow: 0 24px 70px rgba(15,23,42,0.08);
}

.topbar h1 {
    font-size: 42px;
    line-height: 1.16;
    margin: 0 0 8px 0;
    letter-spacing: 0;
}

.topbar p {
    color: var(--muted);
    font-size: 15px;
    margin: 0;
    max-width: 780px;
}

.section-title {
    font-size: 19px;
    font-weight: 800;
    color: var(--ink);
    margin: 18px 0 10px 0;
}

.panel {
    background: rgba(255,255,255,0.86);
    border: 1px solid rgba(255,255,255,0.82);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 18px 42px rgba(15,23,42,0.06);
    margin-bottom: 14px;
}

.panel-title {
    font-size: 14px;
    font-weight: 800;
    color: var(--ink);
    margin-bottom: 10px;
}

.kpi {
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 22px;
    padding: 20px 20px;
    min-height: 124px;
    box-shadow: 0 22px 52px rgba(15,23,42,0.075);
}

.kpi-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}

.kpi-value {
    color: var(--ink);
    font-size: 33px;
    line-height: 1.15;
    font-weight: 850;
    margin-top: 8px;
}

.chart-title {
    font-size: 16px;
    font-weight: 850;
    color: #111827;
    margin-bottom: 3px;
}

.chart-caption {
    color: #64748b;
    font-size: 12px;
    margin-bottom: 14px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 26px;
    border: 1px solid rgba(255,255,255,0.92);
    box-shadow: 0 22px 52px rgba(15,23,42,0.075);
    background: rgba(255,255,255,0.88);
}

.kpi-note {
    color: var(--muted);
    font-size: 12px;
    margin-top: 8px;
}

.tag {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 750;
    margin: 3px 5px 3px 0;
    border: 1px solid transparent;
}

.tag-green {background: #dcfce7; color: #166534; border-color: #bbf7d0;}
.tag-red {background: #fee2e2; color: #991b1b; border-color: #fecaca;}
.tag-purple {background: #ede9fe; color: #5b21b6; border-color: #ddd6fe;}
.tag-amber {background: #fef3c7; color: #92400e; border-color: #fde68a;}
.tag-gray {background: #f1f5f9; color: #334155; border-color: #e2e8f0;}

.mini-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.mini-table th {
    text-align: left;
    padding: 10px;
    border-bottom: 1px solid var(--line);
    color: var(--muted);
}

.mini-table td {
    padding: 10px;
    border-bottom: 1px solid var(--line);
    color: #334155;
}

.insight {
    background: #111827;
    color: #f9fafb;
    border-radius: 8px;
    padding: 16px;
    line-height: 1.7;
    margin-bottom: 14px;
}

.review {
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(226,232,240,0.9);
    border-left: 5px solid #64748b;
    border-radius: 14px;
    padding: 13px 14px;
    margin-bottom: 10px;
    line-height: 1.6;
    color: #334155;
}

.review-meta {
    color: var(--muted);
    font-size: 12px;
    margin-top: 8px;
}

.recommendation-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(226,232,240,0.9);
    border-left: 5px solid #0f766e;
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
}

.recommendation-aspect {
    font-weight: 850;
    color: #111827;
    margin-bottom: 4px;
}

.recommendation-body {
    color: #475569;
    font-size: 13px;
    line-height: 1.55;
}

.highlight-word {
    background: #fef08a;
    color: #713f12;
    border-radius: 6px;
    padding: 0 3px;
    font-weight: 900;
}

.recommendation-detail {
    min-height: 360px;
    border-radius: 22px;
    background:
        radial-gradient(circle at 92% 8%, rgba(187,247,208,0.48), transparent 28%),
        linear-gradient(135deg, rgba(255,255,255,0.96), rgba(248,250,252,0.86));
    border: 1px solid rgba(226,232,240,0.92);
    padding: 22px 24px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.78);
}

.recommendation-kicker {
    color: #64748b;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.recommendation-heading {
    color: #111827;
    font-size: 26px;
    line-height: 1.1;
    font-weight: 950;
    margin-bottom: 12px;
}

.recommendation-stats {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin: 14px 0;
}

.recommendation-stat {
    border-radius: 16px;
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(226,232,240,0.9);
    padding: 10px 12px;
}

.recommendation-stat-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
}

.recommendation-stat-value {
    color: #111827;
    font-size: 18px;
    font-weight: 900;
    margin-top: 4px;
}

.recommendation-copy {
    color: #334155;
    font-size: 14px;
    line-height: 1.65;
    margin-top: 12px;
}

.donut-panel {
    position: relative;
    min-height: 380px;
    display: grid;
    place-items: center;
}

.donut-svg {
    width: min(100%, 390px);
    height: auto;
    display: block;
}

.donut-segment {
    cursor: pointer;
    filter: drop-shadow(0 12px 18px rgba(15, 23, 42, 0.08));
    transition: opacity 0.15s ease, transform 0.15s ease;
    transform-origin: 160px 160px;
}

.donut-segment:hover {
    opacity: 0.82;
    transform: scale(1.015);
}

.donut-segment.selected {
    stroke: #111827;
    stroke-width: 3.5;
}

.donut-center-title {
    font-size: 12px;
    fill: #64748b;
    font-weight: 800;
    text-anchor: middle;
}

.donut-center-value {
    font-size: 24px;
    fill: #111827;
    font-weight: 950;
    text-anchor: middle;
}

.donut-label {
    font-size: 11px;
    fill: #334155;
    font-weight: 850;
    text-anchor: middle;
    pointer-events: none;
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid #0f766e;
    background: #0f766e;
    color: white;
    font-weight: 800;
}

.stButton > button:hover {
    border: 1px solid #115e59;
    background: #115e59;
    color: white;
}

[data-testid="stFileUploader"], [data-testid="stDataFrame"] {
    border-radius: 16px;
}

textarea {
    border-radius: 16px !important;
}

.click-cloud {
    padding: 10px 0 0 0;
    border-radius: 18px;
    background: transparent;
    border: 0;
    line-height: 1.55;
}

.wordcloud-frame {
    border-radius: 24px;
    background:
        radial-gradient(circle at 15% 18%, rgba(187,247,208,0.44), transparent 30%),
        linear-gradient(135deg, rgba(255,255,255,0.92), rgba(248,250,252,0.78));
    padding: 8px 10px 0 10px;
    min-height: 455px;
}

.wordcloud-img {
    display: block;
    width: 100%;
    min-height: 360px;
    max-height: 430px;
    object-fit: contain;
    border-radius: 22px;
    background: white;
    box-shadow: inset 0 0 0 1px rgba(226,232,240,0.8);
}

.cloud-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    padding-top: 8px;
}

.cloud-actions a {
    display: inline-flex;
    align-items: center;
    height: 30px;
    padding: 0 10px;
    border-radius: 999px;
    border: 1px solid rgba(15,118,110,0.16);
    background: rgba(255,255,255,0.74);
    text-decoration: none;
    font-size: 12px;
    font-weight: 800;
    transition: transform 0.12s ease, opacity 0.12s ease;
}

.cloud-actions a:hover {
    transform: translateY(-2px);
    opacity: 0.82;
}

.cloud-hint {
    color: #64748b;
    font-size: 12px;
    margin-top: 6px;
}

.pill-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 4px 0 8px 0;
}

.sentiment-switch {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 4px 0 10px 0;
}

.sentiment-switch a {
    display: inline-flex;
    align-items: center;
    min-height: 32px;
    padding: 0 12px;
    border-radius: 999px;
    border: 1px solid rgba(226,232,240,0.95);
    background: rgba(255,255,255,0.76);
    color: #334155;
    text-decoration: none;
    font-size: 12px;
    font-weight: 850;
}

.sentiment-switch a.active {
    color: #ffffff;
    border-color: transparent;
    box-shadow: 0 10px 22px rgba(15,23,42,0.12);
}

.sentiment-switch a.positive.active {background: #15803d;}
.sentiment-switch a.negative.active {background: #dc2626;}
.sentiment-switch a.neutral.active {background: #ca8a04;}

div[data-testid="stRadio"] > div {
    gap: 6px;
}

div[data-testid="stRadio"] label {
    border-radius: 999px;
    background: rgba(255,255,255,0.74);
    border: 1px solid rgba(226,232,240,0.9);
    padding: 7px 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


def esc(value):
    return html.escape(str(value))


def title_block(title, subtitle):
    st.markdown(
        f"""
        <div class="topbar">
            <h1>{esc(title)}</h1>
            <p>{esc(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label, value, note=""):
    note_html = f"<div class='kpi-note'>{esc(note)}</div>" if note else ""
    return f"""
    <div class="kpi">
        <div class="kpi-label">{esc(label)}</div>
        <div class="kpi-value">{esc(value)}</div>
        {note_html}
    </div>
    """


def panel(title, body):
    return f"<div class='panel'><div class='panel-title'>{esc(title)}</div>{body}</div>"


def chart_heading(title, caption=""):
    caption_html = f"<div class='chart-caption'>{esc(caption)}</div>" if caption else ""
    st.markdown(
        f"<div class='chart-title'>{esc(title)}</div>{caption_html}",
        unsafe_allow_html=True,
    )


def tag_html(text, sentiment=None):
    sentiment = normalize_sentiment(sentiment or text)
    class_name = {
        "Positive": "tag-green",
        "Negative": "tag-red",
        "Neutral": "tag-purple",
    }.get(sentiment, "tag-gray")
    return f"<span class='tag {class_name}'>{esc(text)}</span>"


def tags(items, color="gray"):
    items = [str(item).strip() for item in items if str(item).strip()]
    if not items:
        return "<span class='tag tag-gray'>No result</span>"
    class_name = {
        "green": "tag-green",
        "red": "tag-red",
        "purple": "tag-purple",
        "amber": "tag-amber",
        "gray": "tag-gray",
    }.get(color, "tag-gray")
    return "".join(f"<span class='tag {class_name}'>{esc(item)}</span>" for item in items)


def section(title):
    st.markdown(f"<div class='section-title'>{esc(title)}</div>", unsafe_allow_html=True)


def normalize_sentiment(value):
    text = str(value).strip().lower()
    if text in {"pos", "positive", "1", "good"}:
        return "Positive"
    if text in {"neg", "negative", "-1", "bad"}:
        return "Negative"
    if text in {"neu", "neutral", "0", "mixed"}:
        return "Neutral"
    return str(value).strip().title() if str(value).strip() else "Neutral"


def split_multi(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (SyntaxError, ValueError):
            pass
    return [part.strip() for part in re.split(r"\s*(?:;|\||,)\s*", text) if part.strip()]


def split_column(series):
    values = []
    for value in series.dropna():
        values.extend(split_multi(value))
    return values


def parse_aspect_sentiment_value(value):
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return {str(k).strip(): normalize_sentiment(v) for k, v in parsed.items()}
        if isinstance(parsed, list):
            result = {}
            for item in parsed:
                if isinstance(item, dict):
                    aspect = item.get("aspect") or item.get("Aspect")
                    sentiment = item.get("sentiment") or item.get("Sentiment")
                    if aspect and sentiment:
                        result[str(aspect).strip()] = normalize_sentiment(sentiment)
            if result:
                return result
    except (SyntaxError, ValueError):
        pass

    result = {}
    for pair in re.split(r"\s*(?:;|\|)\s*", text):
        if ":" in pair:
            aspect, sentiment = pair.split(":", 1)
        elif "=" in pair:
            aspect, sentiment = pair.split("=", 1)
        else:
            continue
        aspect = aspect.strip()
        if aspect:
            result[aspect] = normalize_sentiment(sentiment)
    return result


def aspect_sentiment_frame(df):
    rows = []
    for _, row in df.iterrows():
        parsed = parse_aspect_sentiment_value(row.get("aspect_sentiment", ""))
        if not parsed:
            for aspect in split_multi(row.get("aspects", "")):
                parsed[aspect] = normalize_sentiment(row.get("predicted_sentiment", "Neutral"))
        for aspect, sentiment in parsed.items():
            rows.append({"Aspect": aspect, "Sentiment": normalize_sentiment(sentiment)})
    return pd.DataFrame(rows)


def clean_keywords(text, limit=None):
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", str(text).lower())
    cleaned = [word.strip("-'") for word in words if word not in STOPWORDS]
    return cleaned[:limit] if limit else cleaned


def simple_sentiment_predict(review):
    words = clean_keywords(review)
    positive = sum(word in POSITIVE_WORDS for word in words)
    negative = sum(word in NEGATIVE_WORDS for word in words)
    if negative > positive:
        return "Negative", min(0.74 + 0.06 * negative, 0.96)
    if positive > negative:
        return "Positive", min(0.74 + 0.06 * positive, 0.96)
    return "Neutral", 0.70


def simple_aspect_extract(review):
    text = str(review).lower()
    aspects = [
        aspect for aspect, keywords in ASPECT_KEYWORDS.items()
        if any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords)
    ]
    return aspects or ["General Experience"]


def aspect_level_sentiment(review, aspects, fallback):
    text = str(review).lower()
    result = {aspect: fallback for aspect in aspects}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if aspect not in result:
            continue
        window_hits = [kw for kw in keywords if kw in text]
        if not window_hits:
            continue
        positive = sum(word in text for word in POSITIVE_WORDS)
        negative = sum(word in text for word in NEGATIVE_WORDS)
        if aspect == "Delivery":
            positive = sum(word in text for word in ["fast", "quick", "arrived"])
            negative = sum(word in text for word in ["slow", "late", "delayed"])
        elif aspect == "Packaging":
            positive = sum(word in text for word in ["secure", "good", "sealed"])
            negative = sum(word in text for word in ["damaged", "broken", "leak"])
        elif aspect == "Quality":
            positive = sum(word in text for word in ["good", "excellent", "durable", "nice"])
            negative = sum(word in text for word in ["poor", "bad", "defect", "broken"])
        elif aspect == "Price":
            positive = sum(word in text for word in ["worth", "cheap", "value", "affordable"])
            negative = sum(word in text for word in ["expensive", "overpriced"])
        elif aspect == "Customer Service":
            positive = sum(word in text for word in ["helped", "helpful", "quick", "reply"])
            negative = sum(word in text for word in ["no reply", "refund", "rude", "ignored"])

        if negative > positive:
            result[aspect] = "Negative"
        elif positive > negative:
            result[aspect] = "Positive"
    return result


def generate_business_insight(sentiment, aspect_sentiments):
    negative_aspects = [a for a, s in aspect_sentiments.items() if s == "Negative"]
    positive_aspects = [a for a, s in aspect_sentiments.items() if s == "Positive"]
    if sentiment == "Negative":
        focus = ", ".join(negative_aspects or aspect_sentiments.keys())
        return f"This review indicates customer dissatisfaction. The main issue is related to {focus}. The seller should prioritise these areas for service recovery and quality improvement."
    if sentiment == "Positive":
        focus = ", ".join(positive_aspects or aspect_sentiments.keys())
        return f"This review shows customer satisfaction. The strongest customer experience signals are related to {focus}, which can be used in product positioning and seller performance monitoring."
    mixed = ", ".join(aspect_sentiments.keys())
    return f"This review contains mixed or moderate feedback. The seller should inspect the aspect-level sentiment for {mixed} before deciding whether follow-up action is needed."


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    lower_map = {col.lower().strip().replace(" ", "_"): col for col in df.columns}

    aliases = {
        "original_review": ["original_review", "review", "review_text", "text", "comment", "customer_review"],
        "predicted_sentiment": ["predicted_sentiment", "sentiment", "prediction", "sentiment_label", "label"],
        "confidence": ["confidence", "confidence_score", "score", "probability", "sentiment_confidence"],
        "aspects": ["aspects", "detected_aspects", "aspect", "aspect_terms"],
        "aspect_sentiment": ["aspect_sentiment", "aspect_level_sentiment", "absa", "aspect_sentiments"],
        "keywords": ["keywords", "keyword", "extracted_keywords", "key_terms"],
        "rating": ["rating", "stars", "star_rating"],
        "product_category": ["product_category", "category", "product_type"],
    }

    for target, names in aliases.items():
        if target in df.columns:
            continue
        for name in names:
            if name in lower_map:
                df[target] = df[lower_map[name]]
                break

    if "original_review" not in df.columns:
        df["original_review"] = ""
    if "predicted_sentiment" not in df.columns:
        df["predicted_sentiment"] = df["original_review"].apply(lambda x: simple_sentiment_predict(x)[0])
    if "confidence" not in df.columns:
        df["confidence"] = 0.0
    if "aspects" not in df.columns:
        df["aspects"] = df["original_review"].apply(lambda x: ";".join(simple_aspect_extract(x)))
    if "keywords" not in df.columns:
        df["keywords"] = df["original_review"].apply(lambda x: ";".join(clean_keywords(x, 8)))
    else:
        keyword_fallback_source = df.get("cleaned_review", df["original_review"])
        empty_keywords = df["keywords"].isna() | df["keywords"].astype(str).str.strip().eq("")
        df.loc[empty_keywords, "keywords"] = keyword_fallback_source[empty_keywords].apply(
            lambda x: ";".join(clean_keywords(x, 8))
        )
    if "aspect_sentiment" not in df.columns:
        values = []
        for _, row in df.iterrows():
            sentiment = normalize_sentiment(row.get("predicted_sentiment", "Neutral"))
            aspects = split_multi(row.get("aspects", ""))
            values.append(";".join(f"{aspect}:{sentiment}" for aspect in aspects))
        df["aspect_sentiment"] = values

    df["predicted_sentiment"] = df["predicted_sentiment"].apply(normalize_sentiment)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
    if df["confidence"].max() > 1:
        df["confidence"] = df["confidence"] / 100
    df["confidence"] = df["confidence"].clip(0, 1)
    return df


def corpus_candidates():
    roots = [
        Path("sentiment_analysis"),
        Path("sentiment analysis"),
        Path("E-Commerce-Review-Analysis-System") / "sentiment_analysis",
        Path("E-Commerce-Review-Analysis-System") / "sentiment analysis",
        Path("review_source_repo") / "sentiment_analysis",
        Path("review_source_repo") / "sentiment analysis",
    ]
    files = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.csv"))

    def score(path):
        name = path.name.lower()
        score_value = 0
        if name == "sentiment_results.csv":
            score_value += 100
        if name == "final_corpus.csv":
            score_value += 60
        for index, word in enumerate(("sentiment", "result", "corpus", "review", "analysis", "dataset")):
            if word in name:
                score_value += 20 - index
        if "clean" in name or "processed" in name or "final" in name:
            score_value += 8
        return score_value

    return sorted(files, key=lambda path: (-score(path), len(path.parts), str(path)))


def repository_root_for(path):
    parts = list(path.parts)
    if "sentiment_analysis" in parts:
        return Path(*parts[: parts.index("sentiment_analysis")])
    if "sentiment analysis" in parts:
        return Path(*parts[: parts.index("sentiment analysis")])
    return path.parent


def merge_aspect_outputs(df, source_path):
    if "review_id" not in df.columns:
        return df

    repo_root = repository_root_for(source_path)
    aspect_path = repo_root / "absb" / "outputs" / "aspect_results.csv"
    if not aspect_path.exists():
        return df

    try:
        aspects = pd.read_csv(aspect_path)
    except Exception:
        return df

    wanted = [col for col in ["review_id", "aspects", "aspect_sentiment", "keywords"] if col in aspects.columns]
    if "review_id" not in wanted:
        return df

    aspects = aspects[wanted].copy()
    merged = df.merge(aspects, on="review_id", how="left", suffixes=("", "_aspect"))
    for col in ["aspects", "aspect_sentiment", "keywords"]:
        aspect_col = f"{col}_aspect"
        if aspect_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[aspect_col]
        else:
            missing = merged[col].isna() | merged[col].astype(str).str.strip().eq("")
            merged.loc[missing, col] = merged.loc[missing, aspect_col]
        merged = merged.drop(columns=[aspect_col])
    return merged


@st.cache_data
def load_project_recommendations(source_path_text):
    if not source_path_text or source_path_text.startswith("Uploaded") or source_path_text.startswith("Demo"):
        return pd.DataFrame()
    source_path = Path(source_path_text)
    repo_root = repository_root_for(source_path)
    recommendations_path = repo_root / "absb" / "outputs" / "recommendations.csv"
    if not recommendations_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(recommendations_path)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_project_aspect_summary(source_path_text):
    if not source_path_text or source_path_text.startswith("Uploaded") or source_path_text.startswith("Demo"):
        return pd.DataFrame()
    source_path = Path(source_path_text)
    repo_root = repository_root_for(source_path)
    summary_path = repo_root / "absb" / "outputs" / "aspect_sentiment_table.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        return pd.DataFrame()

    required = {"aspect", "Positive", "Neutral", "Negative"}
    if not required.issubset(summary.columns):
        return pd.DataFrame()
    for col in ["Positive", "Neutral", "Negative"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)
    summary["Total"] = summary[["Positive", "Neutral", "Negative"]].sum(axis=1)
    summary["negative_ratio"] = summary["Negative"] / summary["Total"].replace(0, pd.NA)
    summary["negative_ratio"] = summary["negative_ratio"].fillna(0)
    return summary


def recommendation_pie_frame(summary):
    focus_aspects = ["Customer Service", "Packaging", "Quality", "Delivery", "Size and Fit"]
    rows = []
    for aspect in focus_aspects:
        match = summary[summary["aspect"] == aspect]
        if match.empty:
            rows.append({"aspect": aspect, "negative_count": 0})
        else:
            rows.append({"aspect": aspect, "negative_count": int(match.iloc[0]["Negative"])})

    other = summary[~summary["aspect"].isin(focus_aspects)]
    other_count = int(other["Negative"].sum()) if not other.empty else 0
    rows.append({"aspect": "Other", "negative_count": other_count})
    return pd.DataFrame(rows)


def recommendation_donut_chart(summary):
    pie_df = recommendation_pie_frame(summary)
    color_map = {
        "Customer Service": "#0f766e",
        "Packaging": "#c9a227",
        "Quality": "#3f7f72",
        "Delivery": "#83b38b",
        "Size and Fit": "#b86b6b",
        "Other": "#94a3b8",
    }
    fig = px.pie(
        pie_df,
        names="aspect",
        values="negative_count",
        hole=0.56,
        color="aspect",
        color_discrete_map=color_map,
    )
    fig.update_traces(
        textinfo="label+percent",
        textposition="inside",
        marker=dict(line=dict(color="rgba(255,255,255,0.9)", width=3)),
        hovertemplate="%{label}<br>Negative reviews: %{value}<extra></extra>",
    )
    fig.update_layout(
        height=390,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", size=12),
        showlegend=False,
    )
    return fig


def selected_recommendation_aspect():
    query_aspect = st.query_params.get("recommendation_aspect", "")
    if isinstance(query_aspect, list):
        query_aspect = query_aspect[0] if query_aspect else ""
    if query_aspect:
        st.session_state["selected_recommendation_aspect"] = str(query_aspect)
    return st.session_state.get("selected_recommendation_aspect", "Customer Service")


def polar_to_cartesian(cx, cy, radius, angle_degrees):
    angle_radians = (angle_degrees - 90) * 3.141592653589793 / 180.0
    return cx + radius * math.cos(angle_radians), cy + radius * math.sin(angle_radians)


def donut_slice_path(cx, cy, outer_radius, inner_radius, start_angle, end_angle):
    outer_start = polar_to_cartesian(cx, cy, outer_radius, end_angle)
    outer_end = polar_to_cartesian(cx, cy, outer_radius, start_angle)
    inner_start = polar_to_cartesian(cx, cy, inner_radius, start_angle)
    inner_end = polar_to_cartesian(cx, cy, inner_radius, end_angle)
    large_arc = 1 if end_angle - start_angle > 180 else 0
    return (
        f"M {outer_start[0]:.3f} {outer_start[1]:.3f} "
        f"A {outer_radius} {outer_radius} 0 {large_arc} 0 {outer_end[0]:.3f} {outer_end[1]:.3f} "
        f"L {inner_start[0]:.3f} {inner_start[1]:.3f} "
        f"A {inner_radius} {inner_radius} 0 {large_arc} 1 {inner_end[0]:.3f} {inner_end[1]:.3f} Z"
    )


def render_clickable_recommendation_donut(summary, selected_aspect):
    pie_df = recommendation_pie_frame(summary)
    total = max(float(pie_df["negative_count"].sum()), 1)
    colors = {
        "Customer Service": "#0f766e",
        "Packaging": "#c9a227",
        "Quality": "#3f7f72",
        "Delivery": "#83b38b",
        "Size and Fit": "#b86b6b",
        "Other": "#94a3b8",
    }
    cloud_mode, cloud_word = selected_cloud_query()
    cloud_sentiment = selected_cloud_sentiment()
    start = 0
    paths = []
    labels = []
    for _, row in pie_df.iterrows():
        aspect = str(row["aspect"])
        value = float(row["negative_count"])
        span = 360 * value / total if value > 0 else 2
        end = start + span
        path = donut_slice_path(160, 160, 128, 72, start, end)
        mid = start + span / 2
        label_x, label_y = polar_to_cartesian(160, 160, 104, mid)
        href = (
            f"?cloud_mode={quote_plus(cloud_mode)}"
            f"&cloud_word={quote_plus(cloud_word)}"
            f"&cloud_sentiment={quote_plus(cloud_sentiment)}"
            f"&recommendation_aspect={quote_plus(aspect)}"
        )
        selected_class = " selected" if aspect == selected_aspect else ""
        paths.append(
            f"<a href='{href}'><path class='donut-segment{selected_class}' d='{path}' "
            f"fill='{colors.get(aspect, '#94a3b8')}' stroke='rgba(255,255,255,0.92)' stroke-width='3'>"
            f"<title>{esc(aspect)}: {int(value):,} negative reviews</title></path></a>"
        )
        label = aspect.replace("Customer Service", "Service").replace("Size and Fit", "Fit")
        labels.append(f"<text class='donut-label' x='{label_x:.1f}' y='{label_y:.1f}'>{esc(label)}</text>")
        start = end

    selected_negative = int(pie_df.loc[pie_df["aspect"] == selected_aspect, "negative_count"].sum())
    center_label = selected_aspect.replace("Customer Service", "Service").replace("Size and Fit", "Fit")
    return (
        "<div class='donut-panel'>"
        "<svg class='donut-svg' viewBox='0 0 320 320' role='img' aria-label='Business recommendation aspect donut chart'>"
        + "".join(paths)
        + "".join(labels)
        + "<circle cx='160' cy='160' r='58' fill='rgba(255,255,255,0.96)' />"
        + "<text class='donut-center-title' x='160' y='150'>Selected</text>"
        + f"<text class='donut-center-value' x='160' y='176'>{esc(center_label)}</text>"
        + f"<text class='donut-center-title' x='160' y='198'>{selected_negative:,} negative</text>"
        + "</svg></div>"
    )


def recommendation_for_aspect(recommendations, aspect):
    if recommendations.empty:
        return ""
    match = recommendations[recommendations["aspect"] == aspect]
    if match.empty and aspect == "Other":
        match = recommendations[~recommendations["aspect"].isin(["Customer Service", "Packaging", "Quality", "Delivery", "Size and Fit"])]
    if match.empty:
        return "No direct recommendation is available for this aspect yet. Monitor related reviews and inspect recurring complaint themes."
    return str(match.iloc[0].get("recommendation", ""))


def render_recommendation_detail(summary, recommendations, aspect):
    focus_aspects = ["Customer Service", "Packaging", "Quality", "Delivery", "Size and Fit"]
    if aspect == "Other":
        row = {
            "aspect": "Other",
            "Positive": int(summary[~summary["aspect"].isin(focus_aspects)]["Positive"].sum()),
            "Neutral": int(summary[~summary["aspect"].isin(focus_aspects)]["Neutral"].sum()),
            "Negative": int(summary[~summary["aspect"].isin(focus_aspects)]["Negative"].sum()),
        }
        row["Total"] = row["Positive"] + row["Neutral"] + row["Negative"]
        row["negative_ratio"] = row["Negative"] / row["Total"] if row["Total"] else 0
        top_other = summary[~summary["aspect"].isin(focus_aspects)].sort_values("Negative", ascending=False).head(3)
        recommendation = "Other complaints are mainly concentrated in " + ", ".join(top_other["aspect"].astype(str).tolist()) + ". Review these themes together to identify secondary improvement opportunities."
    else:
        match = summary[summary["aspect"] == aspect]
        if match.empty:
            row = {"aspect": aspect, "Positive": 0, "Neutral": 0, "Negative": 0, "Total": 0, "negative_ratio": 0}
        else:
            row = match.iloc[0].to_dict()
        recommendation = recommendation_for_aspect(recommendations, aspect)

    ratio = float(row.get("negative_ratio", 0) or 0)
    st.markdown(
        f"""
        <div class="recommendation-detail">
            <div class="recommendation-kicker">Selected Aspect</div>
            <div class="recommendation-heading">{esc(aspect)}</div>
            <div class="pill-row">
                <span class="tag tag-red">{ratio * 100:.1f}% negative</span>
                <span class="tag tag-gray">{int(row.get("Total", 0)):,} aspect mentions</span>
            </div>
            <div class="recommendation-stats">
                <div class="recommendation-stat">
                    <div class="recommendation-stat-label">Positive</div>
                    <div class="recommendation-stat-value">{int(row.get("Positive", 0)):,}</div>
                </div>
                <div class="recommendation-stat">
                    <div class="recommendation-stat-label">Neutral</div>
                    <div class="recommendation-stat-value">{int(row.get("Neutral", 0)):,}</div>
                </div>
                <div class="recommendation-stat">
                    <div class="recommendation-stat-label">Negative</div>
                    <div class="recommendation-stat-value">{int(row.get("Negative", 0)):,}</div>
                </div>
            </div>
            <div class="recommendation-copy">{esc(recommendation)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_local_project_corpus():
    for path in corpus_candidates():
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df = merge_aspect_outputs(df, path)
        normalized = normalize_columns(df)
        has_review_text = normalized["original_review"].fillna("").astype(str).str.len().gt(0).any()
        if has_review_text:
            return normalized, str(path)
    return None, None


@st.cache_data
def load_sample_data():
    try:
        return normalize_columns(pd.read_csv("reviews.csv"))
    except FileNotFoundError:
        return pd.DataFrame(
            [
                {
                    "review_id": 1,
                    "original_review": "The product quality is good but delivery was slow.",
                    "rating": 3,
                    "predicted_sentiment": "Neutral",
                    "confidence": 0.78,
                    "aspects": "Quality;Delivery",
                    "aspect_sentiment": "Quality:Positive;Delivery:Negative",
                    "keywords": "quality;good;delivery;slow",
                    "product_category": "Electronics",
                }
            ]
        )


def sentiment_distribution_chart(df):
    counts = df["predicted_sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        hole=0.55,
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value}<extra></extra>")
    fig.update_layout(
        height=320,
        margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155"),
        legend=dict(orientation="h", y=-0.08, x=0.12),
    )
    return fig


def aspect_frequency_chart(df):
    counts = pd.Series(split_column(df["aspects"])).value_counts().head(12)
    if counts.empty:
        return None
    chart_df = counts.sort_values().rename_axis("Aspect").reset_index(name="Frequency")
    fig = px.bar(
        chart_df,
        x="Frequency",
        y="Aspect",
        orientation="h",
        text="Frequency",
        color="Frequency",
        color_continuous_scale=["#d7f6ea", "#0f766e"],
    )
    fig.update_layout(
        height=340,
        margin=dict(l=8, r=12, t=12, b=8),
        showlegend=False,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, title=""),
        yaxis=dict(title=""),
        font=dict(color="#334155"),
    )
    fig.update_traces(textposition="outside")
    return fig


def keyword_frequency_chart(df):
    counts = pd.Series(split_column(df["keywords"])).str.lower().value_counts().head(15)
    if counts.empty:
        return None
    chart_df = counts.rename_axis("Keyword").reset_index(name="Frequency")
    fig = px.bar(
        chart_df,
        x="Keyword",
        y="Frequency",
        text="Frequency",
        color="Frequency",
        color_continuous_scale=["#fff2b8", "#facc15", "#0f766e"],
    )
    fig.update_layout(
        height=315,
        margin=dict(l=8, r=8, t=12, b=8),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title="", showgrid=False, zeroline=False),
        font=dict(color="#334155"),
    )
    fig.update_traces(textposition="outside")
    return fig


def aspect_sentiment_heatmap(df):
    parsed = aspect_sentiment_frame(df)
    if parsed.empty:
        return None
    pivot = pd.crosstab(parsed["Aspect"], parsed["Sentiment"])
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    sentiments = ["Positive", "Neutral", "Negative"]
    pivot = pivot[sentiments]
    max_by_sentiment = {sentiment: max(int(pivot[sentiment].max()), 1) for sentiment in sentiments}
    z = []
    text_values = []
    for aspect in pivot.index:
        z_row = []
        text_row = []
        for sentiment in sentiments:
            value = int(pivot.loc[aspect, sentiment])
            if value == 0:
                code = 0
            elif sentiment == "Positive":
                code = 1 + (value / max_by_sentiment[sentiment]) * 98
            elif sentiment == "Neutral":
                code = 101 + (value / max_by_sentiment[sentiment]) * 98
            else:
                code = 201 + (value / max_by_sentiment[sentiment]) * 98
            z_row.append(code)
            text_row.append(value)
        z.append(z_row)
        text_values.append(text_row)

    colorscale = [
        [0.00, "#f8fafc"],
        [0.01, "#dcfce7"],
        [0.18, "#86efac"],
        [0.33, "#15803d"],
        [0.34, "#fef9c3"],
        [0.50, "#fde047"],
        [0.66, "#ca8a04"],
        [0.67, "#fee2e2"],
        [0.84, "#f87171"],
        [1.00, "#b91c1c"],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=pivot.columns,
            y=pivot.index,
            text=text_values,
            texttemplate="%{text}",
            zmin=0,
            zmax=299,
            colorscale=colorscale,
            showscale=False,
            hovertemplate="Aspect: %{y}<br>Sentiment: %{x}<br>Count: %{text}<extra></extra>",
            xgap=8,
            ygap=8,
        )
    )
    fig.update_layout(
        height=390,
        margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Sentiment",
        yaxis_title="Aspect",
        font=dict(color="#334155"),
    )
    return fig


def confidence_chart(df):
    if "confidence" not in df.columns:
        return None
    confidence_colors = {
        "Positive": "#0f766e",
        "Neutral": "#c9a227",
        "Negative": "#b86b6b",
    }
    fig = px.histogram(
        df,
        x="confidence",
        nbins=10,
        color="predicted_sentiment",
        color_discrete_map=confidence_colors,
        opacity=0.86,
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        height=285,
        margin=dict(l=8, r=8, t=12, b=8),
        bargap=0.18,
        xaxis=dict(
            title="Confidence",
            showgrid=False,
            zeroline=False,
            tickformat=".0%",
        ),
        yaxis=dict(
            title="Reviews",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.16)",
            zeroline=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155"),
        showlegend=False,
    )
    return fig


def wordcloud_from_freq(freq, color_func=None):
    if not freq:
        return None
    freq = dict(Counter(freq).most_common(120))
    wc = WordCloud(
        width=1500,
        height=680,
        background_color="white",
        collocations=False,
        prefer_horizontal=0.9,
        max_words=80,
        relative_scaling=0.62,
        min_font_size=22,
        max_font_size=190,
        margin=10,
        color_func=color_func,
    ).generate_from_frequencies(freq)
    return wc.to_image()


def aspect_wordcloud(df):
    return wordcloud_from_freq(dict(Counter(split_column(df["aspects"]))), lambda *args, **kwargs: "#0f766e")


def sentiment_keyword_wordcloud(df, sentiment_filter):
    counts = sentiment_keyword_counter(df, sentiment_filter)
    color = {
        "Positive": "#15803d",
        "Negative": "#dc2626",
        "Neutral": "#ca8a04",
    }.get(sentiment_filter, "#0f766e")
    return wordcloud_from_freq(dict(counts), lambda *args, **kwargs: color)


def render_wordcloud_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"<img class='wordcloud-img' src='data:image/png;base64,{encoded}' alt='word cloud'>"


def get_sentiment_keyword_counts(df):
    positive = []
    negative = []
    neutral = []
    for _, row in df.iterrows():
        keywords = [word.lower() for word in split_multi(row.get("keywords", ""))]
        sentiment = normalize_sentiment(row.get("predicted_sentiment", "Neutral"))
        if sentiment == "Positive":
            positive.extend(keywords)
        elif sentiment == "Negative":
            negative.extend(keywords)
        else:
            neutral.extend(keywords)
    return Counter(positive), Counter(negative), Counter(neutral)


def sentiment_keyword_counter(df, sentiment_filter):
    positive_counts, negative_counts, neutral_counts = get_sentiment_keyword_counts(df)
    return {
        "Positive": positive_counts,
        "Negative": negative_counts,
        "Neutral": neutral_counts,
    }.get(normalize_sentiment(sentiment_filter), positive_counts)


def current_query_param(name, default=""):
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        value = value[0] if value else default
    return str(value)


def cloud_link(word, count, max_count, mode, color, cloud_sentiment=None):
    params = f"?cloud_mode={quote_plus(mode)}&cloud_word={quote_plus(str(word))}"
    if cloud_sentiment:
        params += f"&cloud_sentiment={quote_plus(str(cloud_sentiment))}"
    recommendation_aspect = current_query_param("recommendation_aspect", "")
    if recommendation_aspect:
        params += f"&recommendation_aspect={quote_plus(recommendation_aspect)}"
    return (
        f"<a href='{params}' style='color:{color};' "
        f"title='{esc(word)} appears {count} time(s)'>{esc(word)}</a>"
    )


def clickable_aspect_cloud(df):
    counts = Counter(split_column(df["aspects"]))
    if not counts:
        return ""
    links = [
        cloud_link(word, count, 1, "aspect", "#0f766e")
        for word, count in counts.most_common(10)
    ]
    return (
        "<div class='click-cloud'><div class='cloud-actions'>"
        + "".join(links)
        + "</div><div class='cloud-hint'>Click a chip to inspect related reviews.</div></div>"
    )


def sentiment_switch_html(selected_sentiment):
    current_word = current_query_param("cloud_word", "")
    recommendation_aspect = current_query_param("recommendation_aspect", "")
    links = []
    for sentiment, css_name in [("Positive", "positive"), ("Negative", "negative"), ("Neutral", "neutral")]:
        params = (
            f"?cloud_mode=keyword"
            f"&cloud_word={quote_plus(current_word)}"
            f"&cloud_sentiment={quote_plus(sentiment)}"
        )
        if recommendation_aspect:
            params += f"&recommendation_aspect={quote_plus(recommendation_aspect)}"
        active = " active" if sentiment == selected_sentiment else ""
        links.append(f"<a class='{css_name}{active}' href='{params}'>{esc(sentiment)} Keywords</a>")
    return "<div class='sentiment-switch'>" + "".join(links) + "</div>"


def clickable_sentiment_cloud(df, sentiment_filter):
    sentiment_filter = normalize_sentiment(sentiment_filter)
    counts = sentiment_keyword_counter(df, sentiment_filter)
    if not counts:
        return sentiment_switch_html(sentiment_filter)
    color = {
        "Positive": "#15803d",
        "Negative": "#dc2626",
        "Neutral": "#ca8a04",
    }.get(sentiment_filter, "#0f766e")
    links = [
        cloud_link(word, count, 1, "keyword", color, sentiment_filter)
        for word, count in counts.most_common(14)
    ]
    return (
        "<div class='click-cloud'>"
        + sentiment_switch_html(sentiment_filter)
        + "<div class='cloud-actions'>"
        + "".join(links)
        + "</div><div class='cloud-hint'>Click a chip to inspect related reviews.</div></div>"
    )


def selected_cloud_sentiment():
    sentiment = current_query_param("cloud_sentiment", "Positive")
    sentiment = normalize_sentiment(sentiment)
    return sentiment if sentiment in {"Positive", "Negative", "Neutral"} else "Positive"


def selected_cloud_query():
    mode = current_query_param("cloud_mode", "aspect")
    word = current_query_param("cloud_word", "")
    return str(mode), str(word).replace("+", " ").strip()


def matched_reviews_for_word(df, word, mode, sentiment_filter=None):
    if not word:
        return df.iloc[0:0]
    source_col = "aspects" if mode == "aspect" else "keywords"
    review_text = df["original_review"].fillna("").astype(str)
    source = df[source_col].fillna("").astype(str)
    pattern = rf"\b{re.escape(word)}\b"
    matched = df[
        review_text.str.contains(pattern, case=False, regex=True)
        | source.str.contains(pattern, case=False, regex=True)
    ].copy()
    if mode == "keyword" and sentiment_filter in {"Positive", "Negative", "Neutral"}:
        matched = matched[matched["predicted_sentiment"].apply(normalize_sentiment) == sentiment_filter]
    return matched


def highlight_terms_for_selection(word, mode):
    word = str(word or "").strip()
    if not word:
        return []
    if mode != "aspect":
        return [word]

    mapped_terms = ASPECT_HIGHLIGHT_TERMS.get(word, [])
    aspect_tokens = [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", word.lower())
        if token not in STOPWORDS and token != "and"
    ]
    terms = list(dict.fromkeys([word, *mapped_terms, *aspect_tokens]))
    return [term for term in terms if str(term).strip()]


def highlight_text(text, words):
    escaped = esc(text)
    if isinstance(words, str):
        words = [words]
    words = [str(word).strip() for word in (words or []) if str(word).strip()]
    if not words:
        return escaped

    alternatives = sorted(set(words), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(word) for word in alternatives) + r")\b", re.IGNORECASE)
    return pattern.sub(r'<strong class="highlight-word">\1</strong>', escaped)


def review_card(row, highlight_word=None):
    sentiment = normalize_sentiment(row.get("predicted_sentiment", "Neutral"))
    color = SENTIMENT_COLORS.get(sentiment, "#64748b")
    confidence = row.get("confidence", 0)
    meta = f"{sentiment}"
    if confidence:
        meta += f" | Confidence {confidence * 100:.1f}%"
    review_text = highlight_text(row.get("original_review", ""), highlight_word)
    return f"""
    <div class="review" style="border-left-color:{color}">
        {review_text}
        <div class="review-meta">{esc(meta)} | Aspects: {esc(row.get("aspects", "N/A"))} | Keywords: {esc(row.get("keywords", "N/A"))}</div>
    </div>
    """


def render_sidebar():
    st.markdown(
        """
        <div class="app-nav">
            <div class="brand">
                <div class="brand-mark">R</div>
                <div>
                    <div class="brand-name">ReviewIQ</div>
                    <div class="brand-sub">E-commerce NLP intelligence</div>
                </div>
            </div>
            <div class="nav-right">
                <span>AIT203 Group Project</span>
                <span class="nav-pill">Dashboard</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    default_index = 1 if st.query_params.get("cloud_word", "") else 0
    page = st.radio(
        "Navigation",
        ["Single Review Analysis", "Batch Review Dashboard", "Project Overview"],
        index=default_index,
        horizontal=True,
        label_visibility="collapsed",
    )
    return page


def render_aspect_table(aspect_sentiments):
    rows = ""
    for aspect, sentiment in aspect_sentiments.items():
        rows += f"<tr><td>{esc(aspect)}</td><td>{tag_html(sentiment, sentiment)}</td></tr>"
    return f"""
    <table class="mini-table">
        <thead><tr><th>Aspect</th><th>Sentiment</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


def page_single_review():
    title_block(
        "Single Review Analysis",
        "Analyse one customer review and return review-level sentiment, aspect-level sentiment, keywords, and a business insight.",
    )
    default_review = "The product quality is good, but the delivery was too slow and the packaging was damaged."
    review = st.text_area("Customer review", value=default_review, height=130)
    analyze = st.button("Analyze Review", width="stretch")

    if not analyze:
        st.info("Enter a review and click Analyze Review.")
        return
    if not review.strip():
        st.warning("Please enter a customer review.")
        return

    sentiment, confidence = simple_sentiment_predict(review)
    aspects = simple_aspect_extract(review)
    aspect_sentiments = aspect_level_sentiment(review, aspects, sentiment)
    keywords = clean_keywords(review, 12)
    insight = generate_business_insight(sentiment, aspect_sentiments)

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Predicted Sentiment", sentiment, "Review-level classification"), unsafe_allow_html=True)
    c2.markdown(kpi("Confidence Score", f"{confidence * 100:.1f}%", "Rule-based demo score"), unsafe_allow_html=True)
    c3.markdown(kpi("Detected Aspects", len(aspects), "Aspect extraction result"), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        section("Detected Aspects")
        st.markdown(panel("Aspect Results", tags(aspects, "green")), unsafe_allow_html=True)
        section("Aspect-Level Sentiment")
        st.markdown(panel("ABSA Output", render_aspect_table(aspect_sentiments)), unsafe_allow_html=True)
    with right:
        section("Extracted Keywords")
        st.markdown(panel("Keyword Results", tags(keywords, "purple")), unsafe_allow_html=True)
        section("Business Insight")
        st.markdown(f"<div class='insight'>{esc(insight)}</div>", unsafe_allow_html=True)


def load_uploaded_or_sample(uploaded_file):
    if uploaded_file is not None:
        return normalize_columns(pd.read_csv(uploaded_file)), f"Uploaded file: {uploaded_file.name}"

    local_df, local_path = load_local_project_corpus()
    if local_df is not None:
        return local_df, f"Local project corpus: {local_path}"

    return load_sample_data(), "Demo sample: reviews.csv"


def render_word_explorer(df):
    mode, word = selected_cloud_query()
    sentiment_filter = selected_cloud_sentiment() if mode == "keyword" else None
    if not word:
        aspect_counts = Counter(split_column(df["aspects"]))
        keyword_counts = sentiment_keyword_counter(df, sentiment_filter or "Positive")
        if mode == "keyword":
            word = (keyword_counts.most_common(1) or [("", 0)])[0][0]
        else:
            word = (aspect_counts.most_common(1) or keyword_counts.most_common(1) or [("", 0)])[0][0]
            mode = "aspect" if aspect_counts else "keyword"
        sentiment_filter = selected_cloud_sentiment() if mode == "keyword" else None

    matched = matched_reviews_for_word(df, word, mode, sentiment_filter)
    label = "Aspect" if mode == "aspect" else "Keyword"
    sentiment_note = f" | {sentiment_filter}" if sentiment_filter else ""
    chart_heading(
        "Word Review Explorer",
        f"{label}: {word or 'N/A'}{sentiment_note} | {len(matched)} related review(s)",
    )
    if matched.empty:
        st.info("Click a word in the cloud to display related customer reviews.")
        return
    highlight_terms = highlight_terms_for_selection(word, mode)
    for _, row in matched.head(6).iterrows():
        st.markdown(review_card(row, highlight_word=highlight_terms), unsafe_allow_html=True)


def render_business_recommendations(data_source):
    source_path = data_source.replace("Local project corpus: ", "")
    recommendations = load_project_recommendations(source_path)
    summary = load_project_aspect_summary(source_path)
    if summary.empty:
        return False

    chart_heading("Business Recommendations", "Click a pie slice to inspect the corresponding aspect analysis.")
    left, right = st.columns([0.92, 1.08])
    selected_aspect = selected_recommendation_aspect()
    with left:
        st.markdown(render_clickable_recommendation_donut(summary, selected_aspect), unsafe_allow_html=True)
    with right:
        render_recommendation_detail(summary, recommendations, selected_aspect)
    return True


def page_batch_dashboard():
    title_block(
        "Batch Review Dashboard",
        "Upload the final NLP result CSV and explore sentiment distribution, aspect frequency, ABSA heatmap, keywords, word clouds, and complaint samples.",
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    df, data_source = load_uploaded_or_sample(uploaded_file)
    if df.empty:
        st.warning("The selected CSV is empty.")
        return

    total = len(df)
    sentiment_counts = df["predicted_sentiment"].value_counts()
    positive = int(sentiment_counts.get("Positive", 0))
    negative = int(sentiment_counts.get("Negative", 0))
    aspects = split_column(df["aspects"])
    top_aspect = Counter(aspects).most_common(1)[0][0] if aspects else "N/A"

    st.caption(f"Data source: {data_source}")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Total Reviews", total, "CSV records"), unsafe_allow_html=True)
    c2.markdown(kpi("Positive Rate", f"{positive / total * 100:.1f}%", f"{positive} positive reviews"), unsafe_allow_html=True)
    c3.markdown(kpi("Negative Rate", f"{negative / total * 100:.1f}%", f"{negative} negative reviews"), unsafe_allow_html=True)
    c4.markdown(kpi("Top Aspect", top_aspect, "Most mentioned aspect"), unsafe_allow_html=True)

    section("Review Intelligence Overview")
    left, right = st.columns([0.92, 1.08])
    with left:
        with st.container(border=True):
            chart_heading("Sentiment Distribution", "Overall customer attitude across uploaded reviews.")
            st.plotly_chart(sentiment_distribution_chart(df), use_container_width=True)
    with right:
        with st.container(border=True):
            chart_heading("Aspect Frequency", "Most discussed product and service aspects.")
            fig = aspect_frequency_chart(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No aspect data available.")

    left, right = st.columns([1.08, 0.92])
    with left:
        with st.container(border=True):
            chart_heading("Aspect-Level Sentiment Heatmap", "Green = positive, yellow = neutral, red = negative.")
            st.markdown(
                "<div class='pill-row'>"
                "<span class='tag tag-green'>Positive</span>"
                "<span class='tag tag-amber'>Neutral</span>"
                "<span class='tag tag-red'>Negative</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            fig = aspect_sentiment_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ABSA data available.")
    with right:
        with st.container(border=True):
            chart_heading("Keyword Frequency", "Frequently extracted customer terms.")
            fig = keyword_frequency_chart(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No keyword data available.")

    section("Clickable Word Cloud Analytics")
    wc1, wc2 = st.columns(2)
    with wc1:
        with st.container(border=True):
            chart_heading("Aspect Word Cloud", "One-colour cloud for aspect frequency.")
            fig = aspect_wordcloud(df)
            if fig:
                st.markdown(render_wordcloud_image(fig), unsafe_allow_html=True)
                st.markdown(clickable_aspect_cloud(df), unsafe_allow_html=True)
            else:
                st.info("No aspect words available.")
    with wc2:
        with st.container(border=True):
            selected_sentiment = selected_cloud_sentiment()
            chart_heading(
                "Sentiment Keyword Word Cloud",
                f"{selected_sentiment} keywords only. Switch the sentiment tabs and click a word for review details.",
            )
            fig = sentiment_keyword_wordcloud(df, selected_sentiment)
            if fig:
                st.markdown(render_wordcloud_image(fig), unsafe_allow_html=True)
                st.markdown(clickable_sentiment_cloud(df, selected_sentiment), unsafe_allow_html=True)
            else:
                st.markdown(clickable_sentiment_cloud(df, selected_sentiment), unsafe_allow_html=True)
                st.info(f"No {selected_sentiment.lower()} keyword data available.")

    with st.container(border=True):
        render_word_explorer(df)

    with st.container(border=True):
        chart_heading("Customer Complaint Samples", "Negative reviews for direct business follow-up.")
        negative_reviews = df[df["predicted_sentiment"] == "Negative"].head(8)
        if negative_reviews.empty:
            st.info("No negative reviews found.")
        else:
            complaint_cols = st.columns(2)
            for index, (_, row) in enumerate(negative_reviews.iterrows()):
                with complaint_cols[index % 2]:
                    st.markdown(review_card(row), unsafe_allow_html=True)

    if not load_project_aspect_summary(data_source.replace("Local project corpus: ", "")).empty:
        with st.container(border=True):
            render_business_recommendations(data_source)

    with st.container(border=True):
        chart_heading("Data Preview", "Normalised CSV data passed into the dashboard.")
        st.dataframe(df, width="stretch", hide_index=True)


def page_overview():
    title_block(
        "E-Commerce Review Intelligence System",
        "An NLP-based system for sentiment classification, aspect extraction, aspect-based sentiment analysis, keyword analysis, and customer insight generation.",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Main Pages", "2", "Single review and batch dashboard"), unsafe_allow_html=True)
    c2.markdown(kpi("NLP Outputs", "5", "Sentiment, confidence, aspects, ABSA, keywords"), unsafe_allow_html=True)
    c3.markdown(kpi("Business View", "Ready", "KPIs, charts, samples, explorer"), unsafe_allow_html=True)
    c4.markdown(kpi("CSV Input", "Flexible", "Accepts common column aliases"), unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95])
    with left:
        section("Workflow")
        body = (
            "<b>Raw reviews</b> -> <b>NLP model outputs</b> -> <b>normalised CSV</b> -> "
            "<b>interactive dashboard</b> -> <b>customer insight generation</b>"
        )
        st.markdown(panel("System Pipeline", body), unsafe_allow_html=True)
    with right:
        section("Expected CSV Columns")
        st.markdown(
            panel(
                "Recommended Format",
                tags(["original_review", "predicted_sentiment", "confidence", "aspects", "aspect_sentiment", "keywords"], "amber"),
            ),
            unsafe_allow_html=True,
        )


page = render_sidebar()
if page == "Single Review Analysis":
    page_single_review()
elif page == "Batch Review Dashboard":
    page_batch_dashboard()
else:
    page_overview()
