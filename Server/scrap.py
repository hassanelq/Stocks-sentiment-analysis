import re
import asyncio
import requests
import numpy as np
import pandas as pd
import nltk
import emoji
import traceback
from datetime import datetime, timedelta, timezone
from random import randint
from bs4 import BeautifulSoup

# Reddit
import asyncpraw

# Twitter
from twikit import Client, TooManyRequests

# Text similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK
from nltk.corpus import stopwords

# Transformers (FinBERT)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from dotenv import load_dotenv
import os

load_dotenv() 

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
# For cookies you can parse the JSON string:
import json
TWITTER_COOKIES = json.loads(os.getenv("TWITTER_COOKIES", "{}"))

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone"
)
finbert_pipeline = pipeline(
    "sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer
)


class Scrap:
    def __init__(self):
        print("Initializing Scrap class...")

        self.data = []

        try:
            self.reddit = asyncpraw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
            print("Reddit client initialized.")
        except Exception:
            print("Failed to initialize Reddit client:")
            print(traceback.format_exc())

        try:
            self.twitter_client = Client(language="en-US")
            self.twitter_client.set_cookies(TWITTER_COOKIES)
            print("Twitter client initialized.")
        except Exception:
            print("Failed to initialize Twitter client:")
            print(traceback.format_exc())

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        self.stop_words = set(stopwords.words("english"))

    async def close(self):
        await self.reddit.close()
        print("Reddit session closed.")

    async def scrap_reddit(self, stock_name, days=7):
        print(f"Starting Reddit scraping for {stock_name}, last {days} days...")
        posts = []
        now = datetime.now(timezone.utc)
        since_time = now - timedelta(days=days)

        time_filter = (
            "week"
            if days <= 7
            else "month" if days <= 30 else "year" if days <= 365 else "all"
        )
        try:
            search_query = f'"{stock_name}"'
            subreddit = await self.reddit.subreddit("all")
            async for submission in subreddit.search(
                search_query, sort="new", time_filter=time_filter, limit=None
            ):
                created_utc = datetime.fromtimestamp(
                    submission.created_utc, timezone.utc
                )
                if created_utc < since_time:
                    continue

                combined_text = f"{submission.title} {submission.selftext}"
                posts.append(
                    {
                        "date": created_utc.isoformat(),
                        "text": combined_text,
                        "upvotes": submission.score or 0,
                    }
                )
        except Exception:
            print("Error during Reddit scraping:")
            print(traceback.format_exc())

        df = pd.DataFrame(posts)
        print(f"Reddit scraping complete. Posts collected: {len(df)}")
        return df

    def scrap_finviz(self, ticker, days=3):
        print(f"Starting FinViz scraping for {ticker}, last {days} days...")
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except Exception:
            print("Error during FinViz request:")
            print(traceback.format_exc())
            return pd.DataFrame()

        html = BeautifulSoup(response.text, features="html.parser")
        news_table = html.find(id="news-table")

        if not news_table:
            print(f"No FinViz news table found for {ticker}")
            return pd.DataFrame()

        now = datetime.now()
        cutoff = now - timedelta(days=days)
        parsed_data = []
        prev_date = None

        for row in news_table.findAll("tr"):
            if not row.a or not row.td:
                continue

            title = row.a.text.strip()
            date_data = row.td.text.strip()
            date_part, time_part = (
                (prev_date, date_data)
                if len(date_data.split()) == 1
                else date_data.split(" ", 1)
            )
            prev_date = date_part

            if date_part.lower() == "today":
                date_part = now.strftime("%b-%d-%y")
            elif date_part.lower() == "yesterday":
                date_part = (now - timedelta(days=1)).strftime("%b-%d-%y")

            try:
                full_datetime = datetime.strptime(
                    f"{date_part} {time_part}", "%b-%d-%y %I:%M%p"
                )
            except Exception:
                continue

            if full_datetime < cutoff:
                break

            parsed_data.append([full_datetime.isoformat(), title])

        df = pd.DataFrame(parsed_data, columns=["date", "text"])
        print(f"FinViz scraping complete. News items collected: {len(df)}")
        self.data.append(df)
        return df

    async def scrap_twitter(self, stock_name, days=1, max_tweets=300):
        print(
            f"Starting Twitter scraping for {stock_name}, last {days} days, max {max_tweets} tweets..."
        )
        since_time = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(
            days=days
        )
        query = f'"{stock_name}" lang:en min_faves:10 -filter:links -filter:replies'

        tweets = None
        posts = []
        count = 0

        while count < max_tweets:
            try:
                tweets = (
                    await self.twitter_client.search_tweet(query, product="Latest")
                    if tweets is None
                    else await tweets.next()
                )
            except TooManyRequests as e:
                print("Twitter rate limit hit. Sleeping...")
                await asyncio.sleep(60)
                continue
            except Exception:
                print("Error during Twitter scraping:")
                print(traceback.format_exc())
                break

            if not tweets:
                break

            for tweet in tweets:
                try:
                    created_at = tweet.created_at
                    if isinstance(created_at, str):
                        created_at = datetime.strptime(
                            created_at, "%a %b %d %H:%M:%S %z %Y"
                        )

                    if created_at < since_time:
                        return pd.DataFrame(posts)

                    count += 1
                    posts.append(
                        {
                            "date": created_at.isoformat(),
                            "text": tweet.text,
                            "likes": getattr(tweet, "favorite_count", 1) or 1,
                        }
                    )

                    if count >= max_tweets:
                        break
                except Exception:
                    print("Error processing tweet:")
                    print(traceback.format_exc())
                    continue

        df = pd.DataFrame(posts)
        print(f"Twitter scraping complete. Tweets collected: {len(df)}")
        self.data.append(df)
        return df

    def remove_emojis(self, text):
        text = emoji.replace_emoji(text, "")
        return re.sub(r"[:;=]-?[()\/\\|dpDP]|[()<>}{]", "", text)

    def clean_text_data(self, df, text_column="text", similarity_threshold=0.9):
        print("Cleaning text data...")
        if text_column not in df.columns:
            raise ValueError(f"{text_column} not found in DataFrame.")

        df[text_column] = df[text_column].fillna("").astype(str)
        spam_keywords = [r"\bBREAKING NEWS\b", r"\bBUY NOW\b", r"\bFREE\b", r"\bCEO\b"]
        df = df[
            ~df[text_column].str.contains(
                "|".join(spam_keywords), flags=re.IGNORECASE, regex=True
            )
        ]

        df.loc[:, text_column] = df[text_column].str.replace(
            r"http\S+|www.\S+", "", regex=True
        )
        df.loc[:, text_column] = df[text_column].str.replace(
            r"@\w+|\$\w+", "", regex=True
        )
        df[text_column] = df[text_column].apply(self.remove_emojis)
        df.loc[:, text_column] = df[text_column].str.replace(
            r"[^\w\s]", " ", regex=True
        )
        df[text_column] = (
            df[text_column].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        )
        df = df[df[text_column].str.len() > 20]
        df[text_column] = df[text_column].apply(
            lambda x: " ".join(
                word for word in x.split() if word not in self.stop_words
            )
        )
        df = df.drop_duplicates(subset=[text_column])

        if len(df) > 1:
            tfidf = TfidfVectorizer().fit_transform(df[text_column])
            pairwise_sim = cosine_similarity(tfidf)
            to_drop = set()
            for idx in range(pairwise_sim.shape[0]):
                if idx in to_drop:
                    continue
                similar = [
                    i
                    for i in np.where(pairwise_sim[idx] > similarity_threshold)[0]
                    if i != idx and i > idx
                ]
                to_drop.update(similar)
            df = df.drop(df.index[list(to_drop)])

        print(f"Cleaning complete. Final records: {len(df)}")
        return df.reset_index(drop=True)

    def analyze_sentiment(self, df, text_column="text", weight_column=None):
        print("Starting sentiment analysis...")
        if df.empty:
            return df, "The stock sentiment is NEUTRAL. (No data)"

        def truncate(text):
            tokens = finbert_tokenizer.encode(text, truncation=True, max_length=512)
            return finbert_tokenizer.decode(tokens, skip_special_tokens=True)

        try:
            sentiments = df[text_column].apply(
                lambda t: finbert_pipeline(truncate(t))[0]
            )
            df["sentiment_label"] = sentiments.apply(lambda x: x["label"].lower())
            df["sentiment_score"] = sentiments.apply(lambda x: x["score"])
        except Exception:
            print("Error during sentiment prediction:")
            print(traceback.format_exc())
            df["sentiment_label"] = "neutral"
            df["sentiment_score"] = 0.0

        df["weight"] = (
            df[weight_column].fillna(1).astype(float).clip(lower=1)
            if weight_column in df.columns
            else 1.0
        )
        weights = df.groupby("sentiment_label")["weight"].sum()
        total = weights.sum()

        pos = weights.get("positive", 0.0)
        neg = weights.get("negative", 0.0)
        neu = weights.get("neutral", 0.0)

        pos_share = pos / total if total else 0
        neg_share = neg / total if total else 0
        neu_share = neu / total if total else 0

        print(
            f"Sentiment ratios: ↑ {pos_share:.2f}, ↓ {neg_share:.2f}, ↔ {neu_share:.2f}"
        )

        if max(pos_share, neg_share, neu_share) < 0.4:
            prediction = "The stock sentiment is NEUTRAL."
        elif pos_share > neg_share:
            prediction = "The stock is predicted to go UP based on the sentiments."
        elif neg_share > pos_share:
            prediction = "The stock is predicted to go DOWN based on the sentiments."
        else:
            prediction = "The stock sentiment is NEUTRAL."

        return df, prediction
