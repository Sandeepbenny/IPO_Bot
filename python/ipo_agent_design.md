# IPO Analysis Agent - System Design Document

**Version:** 1.0  
**Last Updated:** December 26, 2025  
**Status:** Technical Specification (Ready for Implementation)

---

## Executive Summary

This document outlines a **production-ready AI agent system** for analyzing Indian IPOs with end-to-end intelligence: market research (DRHP/RHP analysis), real-time signals (GMP, subscription, sentiment), and a lightweight ML model for listing-day gain prediction. The system combines RAG-based document analysis, ethical web scraping, broker APIs, and a decision-support LLM layer to deliver structured investment recommendations.

**Key Outputs:** `APPLY` / `HIGH_RISK` / `AVOID` with confidence scores, red flags, and listing-day gain probability.

---

## 1. Architecture Overview

### 1.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         IPO AGENT SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              INPUT LAYER (Multi-Source)                  │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  • SEBI/NSE/BSE Official APIs (DRHP, RHP, Prospectus)   │  │
│  │  • Web Scrapers (ipowatch.in, chittorgarh.com, liveipo) │  │
│  │  • Subscription Data Feeders (BSE/NSE Bid Updates)      │  │
│  │  • GMP Aggregators (Telegram channels, unofficial APIs) │  │
│  │  • Sentiment Scrapers (Twitter, Moneycontrol comments)  │  │
│  │  • Market Context (Indices, sector performance)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            DATA INGESTION & NORMALIZATION                │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  • MongoDB: Raw feeds (temporal + versioning)            │  │
│  │  • PostgreSQL+PgVector: Structured data + embeddings     │  │
│  │  • Vector Store: DRHP/RHP chunks + semantic search       │  │
│  │  • Validation & Deduplication Layer                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           DOCUMENT INTELLIGENCE (RAG Pipeline)           │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  • PDF Extraction (DRHP/RHP → structured metadata)       │  │
│  │  • Semantic Chunking (business model, risks, financials) │  │
│  │  • Key Field Extraction (promoter, use-of-proceeds, etc) │  │
│  │  • Retrieval-Augmented Generation (LLM + context)        │  │
│  │  • Risk Flag Detector (litigations, concentration, etc)  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          PREDICTION ENGINE (ML + Signals Layer)          │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  Model Input:                                             │  │
│  │  • Day-3 retail oversubscription %                        │  │
│  │  • GMP trend (last 48 hours, delta)                       │  │
│  │  • Market index level at IPO close                        │  │
│  │  • Sector sentiment score (Twitter + news)               │  │
│  │  • Issue characteristics (size, sector, peer multiples)   │  │
│  │                                                           │  │
│  │  Output: P(listing_gain > 20%) with uncertainty           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │        DECISION RUBRIC & LLM EXPLAINABILITY              │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  Rubric Dimensions:                                       │  │
│  │  1. Business Quality (moat, growth, peer comparison)      │  │
│  │  2. Financial Health (margins, leverage, cashflow)        │  │
│  │  3. Offer Quality (valuation, fresh vs OFS mix)           │  │
│  │  4. Market Signals (subscription, GMP sentiment)          │  │
│  │  5. Listing Probability Model                             │  │
│  │                                                           │  │
│  │  LLM Task: Synthesize rubric + cite evidence → narrative  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              OUTPUT LAYER (Recommendations)               │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  • Recommendation: APPLY / HIGH_RISK / AVOID             │  │
│  │  • Confidence Score (0-100)                              │  │
│  │  • Listing Gain Probability (% with 95% CI)             │  │
│  │  • Red Flags List (with severity)                        │  │
│  │  • Investment Thesis (2-3 paragraphs)                    │  │
│  │  • What to Watch (Day-1/3 metrics, catalysts)            │  │
│  │  • Data Sources Used (transparency layer)                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Backend Framework** | FastAPI + Uvicorn | Async, production-ready, integrates well with agents |
| **Agent Orchestration** | LangGraph (Python) | Superior state management vs LangChain; supports ReAct loops |
| **LLM** | Claude 3.5 Sonnet or GPT-4 Turbo | Strong reasoning, document understanding, cost-efficient |
| **Document Store** | PostgreSQL + pgvector | Structured + vector search; DRHP/RHP + metadata |
| **Time-Series DB** | MongoDB (optional) or TimescaleDB | GMP history, subscription snapshots, sentiment over time |
| **Web Scraping** | BeautifulSoup4 + Selenium (Playwright for scale) | Handle JS-rendered sites; Playwright faster than Selenium |
| **Sentiment Analysis** | Hugging Face transformers (DistilBERT) or API | Lightweight on-device NLP; low latency for tweets/comments |
| **ML Model (Listing Gain)** | XGBoost or LightGBM | Interpretable, fast training on ~300 IPO samples |
| **Message Queue** | Redis + Celery or RabbitMQ | Background jobs for scraping, model retraining |
| **Frontend** | React + TypeScript | Display recommendations, drill-down analysis |
| **Deployment** | Docker + OpenShift / AWS ECS | Container-native, scalable, matches your current stack |

---

## 2. Data Sources & Collection Strategy

### 2.1 Official/Primary Sources (High Confidence)

| Source | Data | Frequency | API/Method | Status |
|--------|------|-----------|-----------|--------|
| **SEBI** | DRHP/RHP PDFs, filing metadata | On-file (pre-IPO) | Manual PDF fetch + parse | ✅ Available |
| **NSE IPO Portal** | Subscription bids (QIB/NII/RII), allotment | Every 15–30 min | HTML scrape or XML parse | ✅ Available |
| **BSE IPO Portal** | Subscription bids (QIB/NII/RII), allotment | Every 15–30 min | HTML scrape or XML parse | ✅ Available |
| **IPO Alerts API** | IPO calendar, issue info, listing gains | Hourly updates | REST API (free tier available) | ✅ Available |
| **Broker APIs** (Upstox, AngelOne, Dhan) | Listing prices, P&L on day-1 | Real-time | Websocket + REST | ✅ Available (requires auth) |

### 2.2 Secondary Sources (Medium Confidence)

| Source | Data | Frequency | Scrape Method | Notes |
|--------|------|-----------|----------------|-------|
| **chittorgarh.com** | GMP tracker, subscription status, peer multiples | 2–4 hrs | BeautifulSoup (static HTML) | High-quality, well-structured |
| **ipowatch.in** | GMP aggregation, analyst reviews | Daily/hourly | BeautifulSoup + light parsing | Unofficial but widely used |
| **liveipo.in** | Live subscription, GMP real-time | 30 min | Selenium (JS-heavy site) | Fast updates during IPO window |
| **NSE/BSE Telegram channels** (if partnered) | GMP rumors, unofficial APIs | Real-time | Bot integration (if available) | High variance; must validate |

**Risk:** Secondary sources are unofficial, so always **cross-validate with NSE/BSE official data**.

### 2.3 Sentiment & Context Sources (Low-Medium Confidence)

| Source | Data | Frequency | Method | Processing |
|--------|------|-----------|--------|------------|
| **Twitter/X** | #IPO hashtag, company mentions, trader sentiment | Real-time | Twitter API v2 (free tier) + tweepy | Sentiment: DistilBERT classifier |
| **Moneycontrol Comments** | Retail investor discussion, reviews | Hourly | Selenium + comment scraper | Aggregate polarity score |
| **ET Markets / CNBC-TV18** | News, analyst notes, sector trends | Daily | RSS feed + NLP | Extract ticker mentions + context |
| **StockTwits (if available)** | Retail chatter, sentiment heatmap | Real-time | REST API or scrape | Aggregate bullish/bearish ratio |

**Key:** Sentiment is a **low-weight signal** in your rubric. Flag it as "retail FOMO indicator" not predictive.

---

## 3. Data Ingestion & Storage Architecture

### 3.1 MongoDB (Temporal/Raw Data)

Store **unstructured and time-series data** with versioning:

```json
{
  "ipo_id": "COMPANY_2025_12",
  "company_name": "Company Ltd",
  "data_type": "gmp_snapshot",
  "timestamp": "2025-12-26T18:00:00Z",
  "gmp": {
    "value": 450,
    "sources": [
      { "source": "chittorgarh", "value": 452, "confidence": 0.85 },
      { "source": "ipowatch", "value": 448, "confidence": 0.80 },
      { "source": "liveipo", "value": 450, "confidence": 0.75 }
    ],
    "consensus": 450,
    "variance": 2
  },
  "subscription": {
    "date": "2025-12-26",
    "times": {
      "09:00": { "retail": 2.5, "nii": 1.8, "qib": 1.2, "total": 1.9 },
      "15:00": { "retail": 5.2, "nii": 3.1, "qib": 1.5, "total": 3.9 },
      "close": { "retail": 8.4, "nii": 4.2, "qib": 1.8, "total": 5.1 }
    }
  },
  "sentiment": {
    "tweets_last_4h": 1243,
    "positive_ratio": 0.68,
    "keywords": ["strong", "underpriced", "growth"],
    "fomo_index": 7.2
  }
}
```

### 3.2 PostgreSQL + pgvector (Structured + Semantic)

Store **normalized, queryable data** + vector embeddings:

```sql
-- IPO Master Table
CREATE TABLE ipos (
  id SERIAL PRIMARY KEY,
  sebi_reg_id VARCHAR(50) UNIQUE,
  company_name VARCHAR(200),
  sector VARCHAR(50),
  issue_size_crores DECIMAL(10, 2),
  price_band_low DECIMAL(10, 2),
  price_band_high DECIMAL(10, 2),
  lot_size INT,
  open_date DATE,
  close_date DATE,
  listing_date DATE,
  listing_price DECIMAL(10, 2),
  day1_open DECIMAL(10, 2),
  day1_close DECIMAL(10, 2),
  day1_gain_pct DECIMAL(5, 2),
  created_at TIMESTAMP DEFAULT NOW()
);

-- DRHP/RHP Document Chunks (for RAG)
CREATE TABLE ipo_documents (
  id SERIAL PRIMARY KEY,
  ipo_id INT REFERENCES ipos(id),
  document_type VARCHAR(20), -- 'DRHP', 'RHP', 'Prospectus'
  chunk_id INT,
  section VARCHAR(100), -- 'Business Model', 'Risk Factors', etc.
  text_content TEXT,
  embedding vector(1536), -- OpenAI/Claude embedding
  chunk_index INT,
  created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX ON ipo_documents USING ivfflat (embedding vector_cosine_ops);

-- Key Extracted Fields (structured metadata)
CREATE TABLE ipo_metadata (
  id SERIAL PRIMARY KEY,
  ipo_id INT REFERENCES ipos(id),
  promoter_name VARCHAR(200),
  promoter_holding_pct DECIMAL(5, 2),
  use_of_proceeds JSONB, -- { "capacity": 50, "debt_repayment": 30, "working_capital": 20 }
  peer_companies JSONB, -- list of competitor tickers
  litigations_count INT,
  key_risks TEXT[],
  revenue_cagr_3yr DECIMAL(5, 2),
  net_margin_latest DECIMAL(5, 2),
  debt_equity_ratio DECIMAL(5, 2)
);

-- Daily Market Signals
CREATE TABLE market_signals (
  id SERIAL PRIMARY KEY,
  ipo_id INT REFERENCES ipos(id),
  signal_date DATE,
  gmp DECIMAL(10, 2),
  gmp_pct_of_band DECIMAL(5, 2),
  subscription_retail DECIMAL(8, 2),
  subscription_nii DECIMAL(8, 2),
  subscription_qib DECIMAL(8, 2),
  subscription_total DECIMAL(8, 2),
  sentiment_score DECIMAL(3, 2), -- -1 to 1
  sentiment_tweet_volume INT,
  nifty_level INT,
  sector_index_level INT,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Prediction Model Outputs
CREATE TABLE model_predictions (
  id SERIAL PRIMARY KEY,
  ipo_id INT REFERENCES ipos(id),
  prediction_date DATE,
  p_listing_gain_20pct DECIMAL(5, 3),
  prediction_confidence DECIMAL(3, 2),
  recommendation VARCHAR(20), -- 'APPLY', 'HIGH_RISK', 'AVOID'
  rubric_scores JSONB, -- { "business_quality": 7, "financial_health": 6, ... }
  created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 4. Web Scraping & Real-Time Data Collection

### 4.1 Scraping Strategy (Ethical + Scalable)

**Principles:**
- Respect `robots.txt` and site ToS
- Rotate user-agents + proxies (if needed)
- Implement rate-limiting (1-2 requests per second per domain)
- Cache aggressively (don't re-scrape unchanged data)
- Use official APIs where available

### 4.2 Data Collection Jobs (Celery + Redis)

**Job Schedule:**

```python
# Beat Scheduler Config
CELERY_BEAT_SCHEDULE = {
    # Pre-IPO phase (daily): fetch DRHP/RHP metadata
    'fetch_ipo_calendar': {
        'task': 'tasks.fetch_ipo_calendar',
        'schedule': crontab(hour=9, minute=0),  # 9 AM daily
        'args': ()
    },
    # During IPO phase (every 30 min): subscription + GMP
    'scrape_subscription_live': {
        'task': 'tasks.scrape_subscription_live',
        'schedule': crontab(minute='*/30'),  # Every 30 min
        'args': ()
    },
    # GMP aggregation (hourly during IPO window)
    'aggregate_gmp': {
        'task': 'tasks.aggregate_gmp',
        'schedule': crontab(minute=0),  # Every hour
        'args': ()
    },
    # Sentiment scrape (daily, off-peak)
    'scrape_sentiment': {
        'task': 'tasks.scrape_sentiment',
        'schedule': crontab(hour=22, minute=0),  # 10 PM
        'args': ()
    },
    # Listing day: fetch actual prices + P&L
    'fetch_listing_prices': {
        'task': 'tasks.fetch_listing_prices',
        'schedule': crontab(hour=15, minute=30),  # 3:30 PM (post-market close)
        'args': ()
    }
}
```

### 4.3 Scraper Implementation Examples

**Example 1: Chittorgarh GMP Scraper**

```python
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from datetime import datetime

async def scrape_gmp_chittorgarh(ipo_name: str) -> dict:
    """Fetch GMP from chittorgarh.com"""
    url = f"https://www.chittorgarh.com/ipo/{ipo_name.lower().replace(' ', '_')}_gmp/"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers={"User-Agent": "Mozilla/5.0..."}) as response:
            if response.status != 200:
                return {"error": "Failed to fetch"}
            
            soup = BeautifulSoup(await response.text(), 'html.parser')
            
            # Extract GMP from table or data element
            gmp_value = soup.find('span', {'class': 'gmp-value'})
            gmp_pct = soup.find('span', {'class': 'gmp-pct'})
            
            return {
                "source": "chittorgarh",
                "ipo_name": ipo_name,
                "gmp": int(gmp_value.text) if gmp_value else None,
                "gmp_pct": float(gmp_pct.text) if gmp_pct else None,
                "timestamp": datetime.utcnow().isoformat(),
                "confidence": 0.85
            }
```

**Example 2: NSE Subscription Data Fetcher**

```python
import requests
from typing import Dict

def fetch_nse_subscription(issue_id: str) -> Dict:
    """Fetch subscription data from NSE bid details"""
    # NSE provides bid details as XML/HTML
    url = f"https://www.nseindia.com/invest/check-trades-bids-verify-ipo-bids"
    
    params = {
        "issue": issue_id,
        "type": "consolidated"
    }
    
    response = requests.get(url, params=params)
    
    # Parse XML or HTML response
    from xml.etree import ElementTree as ET
    root = ET.fromstring(response.content)
    
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "retail": float(root.find(".//retail_subscription").text),
        "nii": float(root.find(".//nii_subscription").text),
        "qib": float(root.find(".//qib_subscription").text),
        "total": float(root.find(".//total_subscription").text),
    }
    
    return data
```

**Example 3: Sentiment Scraper (Twitter + Moneycontrol)**

```python
import tweepy
from transformers import pipeline
import asyncio

class SentimentCollector:
    def __init__(self, twitter_api_key: str, bearer_token: str):
        self.client = tweepy.Client(bearer_token=bearer_token)
        self.sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    
    async def fetch_ipo_sentiment(self, ipo_ticker: str, lookback_hours: int = 24) -> dict:
        """Aggregate sentiment from Twitter + Moneycontrol"""
        
        # Twitter sentiment
        query = f"#{ipo_ticker} IPO OR {ipo_ticker} listing"
        tweets = self.client.search_recent_tweets(
            query=query,
            max_results=100,
            tweet_fields=['created_at', 'public_metrics']
        )
        
        sentiments = []
        for tweet in tweets.data:
            result = self.sentiment_classifier(tweet.text[:512])
            sentiments.append({
                "text": tweet.text,
                "sentiment": result[0]['label'],
                "score": result[0]['score'],
                "engagement": tweet.public_metrics['like_count']
            })
        
        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'POSITIVE')
        
        return {
            "source": "twitter",
            "ipo_ticker": ipo_ticker,
            "total_mentions": len(sentiments),
            "positive_ratio": positive_count / len(sentiments) if sentiments else 0.5,
            "average_sentiment_score": sum(s['score'] for s in sentiments) / len(sentiments) if sentiments else 0.5,
            "top_keywords": ["bullish", "underpriced", "growth"],  # Extract via NLP
            "fomo_index": (positive_count / len(sentiments) * 100) if sentiments else 0
        }
```

---

## 5. Document Intelligence & RAG Pipeline

### 5.1 DRHP/RHP Ingestion

**Process:**
1. Download DRHP/RHP PDF from SEBI website
2. Extract text + metadata (company, date, pages)
3. Chunk by section (Business Model, Risks, Financials, etc.)
4. Embed chunks with OpenAI or Claude embeddings
5. Store in PostgreSQL + pgvector

```python
import PyPDF2
from langchain.document_loaders import PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import asyncio

async def ingest_drhp(pdf_path: str, ipo_id: int):
    """Ingest DRHP and chunk for RAG"""
    
    # Load PDF
    loader = PDFLoader(pdf_path)
    docs = await loader.aload()
    
    # Chunk by semantic sections
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    )
    
    chunks = splitter.split_documents(docs)
    
    # Embed & store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    for i, chunk in enumerate(chunks):
        embedding = await embeddings.aembed_query(chunk.page_content)
        
        # Insert into PostgreSQL
        await db.execute("""
            INSERT INTO ipo_documents 
            (ipo_id, document_type, chunk_id, section, text_content, embedding, chunk_index)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, ipo_id, 'DRHP', i, 'auto', chunk.page_content, embedding, i)
    
    print(f"Ingested {len(chunks)} chunks for IPO {ipo_id}")
```

### 5.2 Key Field Extraction (Structured Metadata)

**Use LLM to extract from DRHP:**

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

extraction_prompt = PromptTemplate.from_template("""
Extract the following fields from the DRHP document:

1. **Promoter Name & Holding %:** Who are the main promoters and their stake?
2. **Use of Proceeds:** What % is allocated to capacity, debt repayment, working capital?
3. **Key Business Risks:** List top 3 operational or market risks
4. **Revenue & Profitability Trends:** 3-year CAGR, latest net margin
5. **Peer Companies:** List 3-5 comparable listed companies
6. **Litigations:** Any material lawsuits or regulatory issues?

Document excerpt:
{document_excerpt}

Return as JSON.
""")

async def extract_ipo_metadata(ipo_id: int):
    """Extract structured fields from DRHP chunks"""
    
    # Retrieve full DRHP (or top chunks via search)
    chunks = await db.fetch("""
        SELECT text_content FROM ipo_documents 
        WHERE ipo_id = $1 AND document_type = 'DRHP'
        ORDER BY chunk_index LIMIT 20
    """, ipo_id)
    
    full_text = "\n\n".join([c['text_content'] for c in chunks])
    
    # Extract with LLM
    chain = extraction_prompt | llm
    result = await chain.ainvoke({"document_excerpt": full_text[:4000]})
    
    metadata = json.loads(result.content)
    
    # Store in ipo_metadata table
    await db.execute("""
        INSERT INTO ipo_metadata 
        (ipo_id, promoter_name, use_of_proceeds, key_risks, ...)
        VALUES ($1, $2, $3, $4, ...)
    """, ipo_id, metadata.get("promoter_name"), metadata.get("use_of_proceeds"), ...)
```

### 5.3 Risk Flag Detection

**Automated red flag scoring:**

```python
async def detect_risk_flags(ipo_id: int) -> list:
    """Identify risk flags from DRHP + market data"""
    
    flags = []
    
    # 1. Promoter concentration risk
    metadata = await db.fetchrow(
        "SELECT promoter_holding_pct FROM ipo_metadata WHERE ipo_id = $1", ipo_id
    )
    if metadata['promoter_holding_pct'] > 80:
        flags.append({
            "flag": "High Promoter Concentration",
            "severity": "HIGH",
            "value": f"{metadata['promoter_holding_pct']}% holding",
            "note": "Limited free float; potential illiquidity post-listing"
        })
    
    # 2. Litigation risk
    ipo = await db.fetchrow("SELECT * FROM ipos WHERE id = $1", ipo_id)
    litigation_count = await db.fetchval(
        "SELECT COUNT(*) FROM ipo_metadata WHERE ipo_id = $1 AND litigations_count > 0", ipo_id
    )
    if litigation_count > 3:
        flags.append({
            "flag": "Material Litigations",
            "severity": "MEDIUM",
            "value": f"{litigation_count} ongoing cases",
            "note": "Check case status and potential impact on financials"
        })
    
    # 3. Valuation vs peers
    metadata = await db.fetchrow("""
        SELECT peer_companies, revenue_cagr_3yr FROM ipo_metadata WHERE ipo_id = $1
    """, ipo_id)
    if metadata['revenue_cagr_3yr'] < 5:
        flags.append({
            "flag": "Slow Growth",
            "severity": "MEDIUM",
            "value": f"{metadata['revenue_cagr_3yr']}% CAGR",
            "note": "Check if valuation multiple is reasonable for growth profile"
        })
    
    # 4. GMP vs subscription mismatch
    signals = await db.fetchrow("""
        SELECT gmp_pct_of_band, subscription_retail, subscription_qib 
        FROM market_signals WHERE ipo_id = $1 ORDER BY signal_date DESC LIMIT 1
    """, ipo_id)
    if signals and signals['gmp_pct_of_band'] > 80 and signals['subscription_qib'] < 0.5:
        flags.append({
            "flag": "GMP-Subscription Divergence",
            "severity": "HIGH",
            "value": f"GMP {signals['gmp_pct_of_band']}% above band, QIB {signals['subscription_qib']}x",
            "note": "Institutional investors lukewarm; potential GMP pump by retail"
        })
    
    return sorted(flags, key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[x['severity']], reverse=True)
```

---

## 6. ML Model: Listing Gain Prediction

### 6.1 Data Preparation (200-300 Historical IPOs, 2021-2025)

**Features:**
- Day-3 retail oversubscription %
- GMP trend (48-hour delta)
- Market index level (Nifty 50) at IPO close
- Sector sentiment (aggregated from social)
- Issue size (normalized)
- Price band (normalized)
- Peer P/E multiple (relative valuation)

**Target:** Binary: Listing gain > 20% (1) or ≤ 20% (0)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_training_data(ipo_historical_df: pd.DataFrame) -> tuple:
    """Prepare features and labels from historical IPOs"""
    
    # Feature engineering
    df = ipo_historical_df.copy()
    
    # Day-3 subscription: retail category
    df['day3_retail_sub'] = df['subscription_retail_day3']  # Already in data
    
    # GMP trend: compare day-3 GMP vs day-1 GMP
    df['gmp_48h_delta'] = df['gmp_day3'] - df['gmp_day1']
    df['gmp_delta_pct'] = (df['gmp_48h_delta'] / df['price_band_high']) * 100
    
    # Nifty level at IPO close
    df['nifty_ipo_close'] = df['nifty_level_ipo_close_date']
    
    # Sector sentiment
    df['sector_sentiment'] = df['sentiment_score']  # -1 to 1
    
    # Normalized issue size (log scale)
    df['issue_size_log'] = np.log1p(df['issue_size_crores'])
    
    # Peer P/E (relative): IPO valuation vs sector average
    df['pe_relative'] = df['ipo_pe_multiple'] / df['sector_avg_pe']
    
    # Target: listing gain > 20%
    df['listing_gain_pct'] = ((df['day1_close'] - df['listing_price']) / df['listing_price']) * 100
    df['target'] = (df['listing_gain_pct'] > 20).astype(int)
    
    # Select features
    feature_cols = [
        'day3_retail_sub', 'gmp_delta_pct', 'nifty_ipo_close',
        'sector_sentiment', 'issue_size_log', 'pe_relative'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['target']
    
    return X, y

def train_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Train XGBoost classifier"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Train Accuracy: {train_score:.3f}, Test Accuracy: {test_score:.3f}")
    
    # Save model & scaler
    joblib.dump(model, "models/listing_gain_model.pkl")
    joblib.dump(scaler, "models/feature_scaler.pkl")
    
    return model

def predict_listing_gain(feature_dict: dict) -> dict:
    """Predict P(listing_gain > 20%) for a new IPO"""
    
    # Load model and scaler
    model = joblib.load("models/listing_gain_model.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")
    
    # Create feature vector
    features = np.array([[
        feature_dict['day3_retail_sub'],
        feature_dict['gmp_delta_pct'],
        feature_dict['nifty_ipo_close'],
        feature_dict['sector_sentiment'],
        feature_dict['issue_size_log'],
        feature_dict['pe_relative']
    ]])
    
    features_scaled = scaler.transform(features)
    
    # Predict
    prob = model.predict_proba(features_scaled)[0]
    prediction = model.predict(features_scaled)[0]
    
    return {
        "p_listing_gain_20pct": round(prob[1], 3),
        "confidence": round(max(prob), 2),
        "predicted_class": "GAIN_>20%" if prediction == 1 else "GAIN_<=20%"
    }
```

---

## 7. Decision Rubric & Scoring System

### 7.1 Five-Pillar Rubric

Each pillar scores 0–10; final recommendation based on weighted average.

```python
class IPORubricEvaluator:
    """Evaluate IPO across 5 dimensions"""
    
    WEIGHTS = {
        "business_quality": 0.25,
        "financial_health": 0.20,
        "offer_quality": 0.20,
        "market_signals": 0.20,
        "listing_prediction": 0.15
    }
    
    async def evaluate_business_quality(self, ipo_id: int) -> tuple:
        """
        Score factors:
        - Market position (moat, competitive advantage)
        - Revenue growth trajectory
        - Peer comparison
        Returns: (score, reasons)
        """
        metadata = await db.fetchrow("""
            SELECT revenue_cagr_3yr, net_margin_latest FROM ipo_metadata WHERE ipo_id = $1
        """, ipo_id)
        
        score = 0
        reasons = []
        
        if metadata['revenue_cagr_3yr'] > 25:
            score += 9
            reasons.append("Strong revenue growth (>25% CAGR)")
        elif metadata['revenue_cagr_3yr'] > 15:
            score += 7
            reasons.append("Moderate growth (15-25% CAGR)")
        elif metadata['revenue_cagr_3yr'] > 5:
            score += 5
            reasons.append("Stable growth (5-15% CAGR)")
        else:
            score += 3
            reasons.append("Weak growth (<5% CAGR); check valuation")
        
        # Profitability
        if metadata['net_margin_latest'] > 15:
            score += 1
            reasons.append("High profitability (>15% net margin)")
        
        return (score / 10, reasons)
    
    async def evaluate_financial_health(self, ipo_id: int) -> tuple:
        """
        Score factors:
        - Debt-to-equity ratio
        - Working capital trends
        - Liquidity
        Returns: (score, reasons)
        """
        metadata = await db.fetchrow("""
            SELECT debt_equity_ratio FROM ipo_metadata WHERE ipo_id = $1
        """, ipo_id)
        
        score = 0
        reasons = []
        
        de_ratio = metadata['debt_equity_ratio']
        if de_ratio < 0.5:
            score += 9
            reasons.append(f"Strong balance sheet (D/E: {de_ratio:.2f})")
        elif de_ratio < 1.0:
            score += 6
            reasons.append(f"Moderate leverage (D/E: {de_ratio:.2f})")
        else:
            score += 3
            reasons.append(f"High leverage (D/E: {de_ratio:.2f}); watch cash flows")
        
        return (score / 10, reasons)
    
    async def evaluate_offer_quality(self, ipo_id: int) -> tuple:
        """
        Score factors:
        - Valuation vs peers (P/E, P/B)
        - Fresh issue vs OFS ratio
        - Use of proceeds clarity
        Returns: (score, reasons)
        """
        ipo = await db.fetchrow("SELECT * FROM ipos WHERE id = $1", ipo_id)
        
        score = 0
        reasons = []
        
        # Valuation logic (mock; in reality, compare with 5-peer median)
        peer_avg_pe = 25  # Placeholder
        ipo_pe = 20       # Placeholder
        
        if ipo_pe < peer_avg_pe * 0.8:
            score += 9
            reasons.append(f"Attractive valuation (P/E: {ipo_pe:.1f} vs peer avg: {peer_avg_pe})")
        elif ipo_pe < peer_avg_pe:
            score += 7
            reasons.append(f"Fair valuation (P/E: {ipo_pe:.1f})")
        else:
            score += 4
            reasons.append(f"Premium valuation (P/E: {ipo_pe:.1f} > peers); check growth justification")
        
        return (score / 10, reasons)
    
    async def evaluate_market_signals(self, ipo_id: int) -> tuple:
        """
        Score factors:
        - Subscription levels (retail vs QIB balance)
        - GMP (but heavily discounted vs official factors)
        - Retail sentiment (FOMO gauge)
        Returns: (score, reasons)
        """
        signals = await db.fetchrow("""
            SELECT subscription_retail, subscription_qib, gmp_pct_of_band, sentiment_score
            FROM market_signals WHERE ipo_id = $1 ORDER BY signal_date DESC LIMIT 1
        """, ipo_id)
        
        score = 0
        reasons = []
        
        # Retail subscription
        if signals['subscription_retail'] > 5:
            score += 3
            reasons.append(f"Strong retail demand ({signals['subscription_retail']:.1f}x)")
        
        # QIB participation (higher = healthier demand from institutions)
        if signals['subscription_qib'] > 1.5:
            score += 3
            reasons.append(f"Good institutional interest ({signals['subscription_qib']:.1f}x)")
        else:
            reasons.append(f"Weak QIB participation ({signals['subscription_qib']:.1f}x); caution")
        
        # GMP as "FOMO indicator" only
        if signals['gmp_pct_of_band'] < 30:
            score += 2
            reasons.append(f"Moderate GMP ({signals['gmp_pct_of_band']:.0f}%); realistic expectations")
        elif signals['gmp_pct_of_band'] > 80:
            reasons.append(f"Very high GMP ({signals['gmp_pct_of_band']:.0f}%); possible pump; high reversal risk")
        
        # Sentiment
        if signals['sentiment_score'] > 0.6:
            score += 2
            reasons.append("Positive retail sentiment on social media")
        
        return (score / 10, reasons)
    
    async def get_listing_prediction(self, ipo_id: int) -> tuple:
        """Fetch ML model's listing gain prediction"""
        pred = await db.fetchrow("""
            SELECT p_listing_gain_20pct FROM model_predictions WHERE ipo_id = $1
            ORDER BY prediction_date DESC LIMIT 1
        """, ipo_id)
        
        if not pred:
            return (0.5, ["Model prediction pending"])
        
        prob = pred['p_listing_gain_20pct']
        score = min(10, prob * 10)  # Scale 0-1 to 0-10
        
        reasons = [f"ML model predicts {prob*100:.0f}% chance of listing gain > 20%"]
        
        return (score / 10, reasons)
    
    async def compute_final_score(self, ipo_id: int) -> dict:
        """Weighted average of all pillars"""
        
        scores = {}
        all_reasons = {}
        
        # Evaluate all pillars
        business_score, business_reasons = await self.evaluate_business_quality(ipo_id)
        financial_score, financial_reasons = await self.evaluate_financial_health(ipo_id)
        offer_score, offer_reasons = await self.evaluate_offer_quality(ipo_id)
        market_score, market_reasons = await self.evaluate_market_signals(ipo_id)
        listing_score, listing_reasons = await self.get_listing_prediction(ipo_id)
        
        scores = {
            "business_quality": business_score,
            "financial_health": financial_score,
            "offer_quality": offer_score,
            "market_signals": market_score,
            "listing_prediction": listing_score
        }
        
        all_reasons = {
            "business_quality": business_reasons,
            "financial_health": financial_reasons,
            "offer_quality": offer_reasons,
            "market_signals": market_reasons,
            "listing_prediction": listing_reasons
        }
        
        # Weighted average
        final_score = sum(scores[k] * self.WEIGHTS[k] for k in scores) * 10
        
        # Recommendation logic
        if final_score >= 7.5:
            recommendation = "APPLY"
        elif final_score >= 5.5:
            recommendation = "HIGH_RISK"
        else:
            recommendation = "AVOID"
        
        return {
            "final_score": round(final_score, 1),
            "recommendation": recommendation,
            "pillar_scores": {k: round(v * 10, 1) for k, v in scores.items()},
            "pillar_reasons": all_reasons
        }
```

---

## 8. LLM Agent Loop (LangGraph)

### 8.1 Agent Workflow

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    ipo_id: int
    company_name: str
    messages: Annotated[list, operator.add]
    rubric_evaluation: dict
    red_flags: list
    model_prediction: dict
    final_recommendation: str
    investment_thesis: str

def build_ipo_agent():
    """Build LangGraph agent for IPO analysis"""
    
    workflow = StateGraph(AgentState)
    
    # Node 1: Fetch IPO Data
    async def fetch_ipo_data(state: AgentState):
        ipo_id = state['ipo_id']
        
        ipo = await db.fetchrow("SELECT * FROM ipos WHERE id = $1", ipo_id)
        metadata = await db.fetchrow("SELECT * FROM ipo_metadata WHERE ipo_id = $1", ipo_id)
        
        state['company_name'] = ipo['company_name']
        state['messages'].append(AIMessage(
            content=f"Analyzing {ipo['company_name']} IPO (Issue Size: ₹{ipo['issue_size_crores']} Cr, Price Band: ₹{ipo['price_band_low']}-{ipo['price_band_high']})"
        ))
        
        return state
    
    # Node 2: Run Rubric Evaluation
    async def run_rubric(state: AgentState):
        evaluator = IPORubricEvaluator()
        rubric_result = await evaluator.compute_final_score(state['ipo_id'])
        
        state['rubric_evaluation'] = rubric_result
        state['messages'].append(AIMessage(
            content=f"Rubric Score: {rubric_result['final_score']}/10"
        ))
        
        return state
    
    # Node 3: Detect Red Flags
    async def detect_flags(state: AgentState):
        flags = await detect_risk_flags(state['ipo_id'])
        state['red_flags'] = flags
        
        if flags:
            flag_summary = "\n".join([f"⚠️ {f['flag']}: {f['value']}" for f in flags[:3]])
            state['messages'].append(AIMessage(content=f"Red Flags Detected:\n{flag_summary}"))
        
        return state
    
    # Node 4: Get ML Prediction
    async def get_ml_pred(state: AgentState):
        pred = await db.fetchrow("""
            SELECT p_listing_gain_20pct, recommendation FROM model_predictions 
            WHERE ipo_id = $1 ORDER BY prediction_date DESC LIMIT 1
        """, state['ipo_id'])
        
        state['model_prediction'] = {
            "p_listing_gain_20pct": pred['p_listing_gain_20pct'],
            "recommendation": pred['recommendation']
        }
        
        state['messages'].append(AIMessage(
            content=f"ML Prediction: {pred['p_listing_gain_20pct']*100:.0f}% chance of listing gain > 20%"
        ))
        
        return state
    
    # Node 5: Generate Investment Thesis (LLM)
    async def generate_thesis(state: AgentState):
        rubric = state['rubric_evaluation']
        flags = state['red_flags']
        
        prompt = f"""
Based on the following IPO analysis, write a 2-3 paragraph investment thesis.

Company: {state['company_name']}

Rubric Scores:
- Business Quality: {rubric['pillar_scores']['business_quality']}/10
- Financial Health: {rubric['pillar_scores']['financial_health']}/10
- Offer Quality: {rubric['pillar_scores']['offer_quality']}/10
- Market Signals: {rubric['pillar_scores']['market_signals']}/10

Red Flags: {', '.join([f['flag'] for f in flags[:3]]) if flags else 'None'}

Key Reasons for Recommendation:
{json.dumps(rubric['pillar_reasons'], indent=2)}

Write a concise, professional thesis explaining the investment case or caution.
        """
        
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        thesis = await llm.ainvoke(prompt)
        
        state['investment_thesis'] = thesis.content
        state['messages'].append(AIMessage(content=f"Investment Thesis:\n{thesis.content}"))
        
        return state
    
    # Node 6: Final Recommendation
    async def final_recommendation(state: AgentState):
        rubric = state['rubric_evaluation']
        
        state['final_recommendation'] = rubric['recommendation']
        
        state['messages'].append(AIMessage(
            content=f"Final Recommendation: {rubric['recommendation']}"
        ))
        
        return state
    
    # Add nodes
    workflow.add_node("fetch_data", fetch_ipo_data)
    workflow.add_node("rubric_eval", run_rubric)
    workflow.add_node("detect_flags", detect_flags)
    workflow.add_node("ml_pred", get_ml_pred)
    workflow.add_node("thesis", generate_thesis)
    workflow.add_node("recommend", final_recommendation)
    
    # Add edges
    workflow.add_edge(START, "fetch_data")
    workflow.add_edge("fetch_data", "rubric_eval")
    workflow.add_edge("rubric_eval", "detect_flags")
    workflow.add_edge("detect_flags", "ml_pred")
    workflow.add_edge("ml_pred", "thesis")
    workflow.add_edge("thesis", "recommend")
    workflow.add_edge("recommend", END)
    
    return workflow.compile()

# Usage
agent = build_ipo_agent()

initial_state = {
    "ipo_id": 123,
    "company_name": "",
    "messages": [],
    "rubric_evaluation": {},
    "red_flags": [],
    "model_prediction": {},
    "final_recommendation": "",
    "investment_thesis": ""
}

result = await agent.ainvoke(initial_state)
```

---

## 9. FastAPI Backend & Endpoints

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg

app = FastAPI(title="IPO Agent API")

# Database connection pool
db_pool: asyncpg.Pool = None

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(
        host="localhost", database="ipo_db",
        user="ipo_user", password="password",
        min_size=5, max_size=20
    )

@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()

# Request/Response Models
class AnalyzeIPORequest(BaseModel):
    ipo_id: int
    company_name: str

class IPORecommendation(BaseModel):
    company_name: str
    recommendation: str  # APPLY, HIGH_RISK, AVOID
    confidence: float    # 0-1
    listing_gain_probability: float
    red_flags: list
    investment_thesis: str
    data_sources_used: list

# Endpoints
@app.post("/analyze-ipo", response_model=IPORecommendation)
async def analyze_ipo(request: AnalyzeIPORequest):
    """Analyze an IPO and return recommendation"""
    
    try:
        agent = build_ipo_agent()
        
        result = await agent.ainvoke({
            "ipo_id": request.ipo_id,
            "company_name": request.company_name,
            "messages": [],
            "rubric_evaluation": {},
            "red_flags": [],
            "model_prediction": {},
            "final_recommendation": "",
            "investment_thesis": ""
        })
        
        return IPORecommendation(
            company_name=result['company_name'],
            recommendation=result['final_recommendation'],
            confidence=result['rubric_evaluation']['final_score'] / 10,
            listing_gain_probability=result['model_prediction'].get('p_listing_gain_20pct', 0.5),
            red_flags=[f['flag'] for f in result['red_flags']],
            investment_thesis=result['investment_thesis'],
            data_sources_used=["SEBI DRHP", "NSE Subscription", "Twitter Sentiment", "Chittorgarh GMP"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ipo/{ipo_id}")
async def get_ipo(ipo_id: int):
    """Fetch IPO details"""
    ipo = await db_pool.fetchrow("SELECT * FROM ipos WHERE id = $1", ipo_id)
    return ipo

@app.get("/ipo/{ipo_id}/market-signals")
async def get_market_signals(ipo_id: int):
    """Fetch real-time market signals"""
    signals = await db_pool.fetchrow("""
        SELECT * FROM market_signals WHERE ipo_id = $1
        ORDER BY created_at DESC LIMIT 1
    """, ipo_id)
    return signals
```

---

## 10. Implementation Roadmap (Phase-Based)

### Phase 1: MVP (Weeks 1-4)
- ✅ Set up PostgreSQL + MongoDB
- ✅ Build basic DRHP ingestion (PDF extraction)
- ✅ Implement NSE/BSE subscription scraper
- ✅ Draft rubric evaluation logic
- ✅ Deploy FastAPI with `/analyze-ipo` endpoint
- ✅ Create simple React dashboard

**Deliverable:** End-to-end analysis for 1 live IPO; manual testing

### Phase 2: Data & ML (Weeks 5-8)
- ✅ Collect 200-300 historical IPO data (2021-2025)
- ✅ Build feature engineering pipeline
- ✅ Train XGBoost listing-gain model
- ✅ Integrate sentiment scraper (Twitter/Moneycontrol)
- ✅ Set up Celery jobs for continuous data collection

**Deliverable:** ML predictions with 70%+ accuracy on test set

### Phase 3: Production Hardening (Weeks 9-12)
- ✅ Add caching + rate-limiting
- ✅ Implement error handling & retry logic
- ✅ Set up monitoring (logging, alerts)
- ✅ Build admin dashboard for model retraining
- ✅ Documentation + API spec (OpenAPI)

**Deliverable:** Production-ready system, live for all current IPOs

### Phase 4: Advanced Features (Weeks 13+)
- Optional: Real-time alerts (Telegram/Email)
- Optional: Portfolio impact simulation
- Optional: Fine-tune LLM for IPO-specific reasoning

---

## 11. Risk & Guardrails

### 11.1 Disclaimers & Legal

```
CRITICAL DISCLAIMER:
This system is a RESEARCH TOOL, not financial advice.
- Past IPO performance ≠ future returns
- GMP is unofficial and volatile; not exchange-verified
- All recommendations are probabilistic, not certain
- Users must consult SEBI-registered advisors
- No liability for losses arising from recommendations
```

### 11.2 Data Quality Checks

- ✅ Validate all web-scraped data against NSE/BSE official sources
- ✅ Flag GMP outliers (>3σ from median) as potential errors
- ✅ Cross-check subscription data across multiple sources
- ✅ Audit sentiment scores (manual review of flagged tweets)

### 11.3 Model Monitoring

- ✅ Track prediction accuracy post-listing (vs actual gains)
- ✅ Retrain model quarterly with new IPO data
- ✅ Monitor feature drift (e.g., if retail subscription patterns shift)
- ✅ Alert if confidence drops below threshold

---

## 12. Success Metrics

| Metric | Target |
|--------|--------|
| **Model Accuracy** | 70%+ on test set (P(gain > 20%)) |
| **Recommendation Precision** | 75%+ (if we say APPLY, IPO gains 20%+) |
| **Data Freshness** | GMP/subscription updates every 30 min during IPO |
| **API Response Time** | <2 sec for full analysis |
| **System Uptime** | 99%+ (24/7 during IPO season) |

---

## 13. Next Steps

1. **Week 1:** Finalize DB schema + start DRHP ingestion
2. **Week 1-2:** Build NSE subscription + GMP scrapers
3. **Week 2-3:** Implement rubric scoring
4. **Week 3-4:** Deploy FastAPI + MVP testing
5. **Week 5+:** ML model training + sentiment integration

---

**Document prepared for:** IPO Analysis AI Agent  
**Status:** Ready for Development  
**Last reviewed:** December 26, 2025
