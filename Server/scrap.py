import praw
import pandas as pd
from datetime import datetime, timedelta, timezone
import requests
from bs4 import BeautifulSoup
from twikit import Client, TooManyRequests
from random import randint
import time
import asyncio

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import emoji

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# importing os module for environment variables
import os

# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values

# loading variables from .env file
load_dotenv()


class Scrap:
    def __init__(self):
        self.data = []
        # Initialize Reddit client
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )
        # Initialize Twitter client
        self.twitter_client = Client(language="en-US")
        self.twitter_client.load_cookies(
            "cookies.json"
        )  # Load existing cookies for authentication twitter
        # Ensure NLTK stopwords are downloaded
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

    def scrap_reddit(self, stock_name, days=7):
        posts = []
        now = datetime.utcnow()
        since_time = now - timedelta(days=days)

        search_query = f'"{stock_name}"'

        submissions = self.reddit.subreddit("all").search(
            search_query, sort="new", time_filter="all", limit=None
        )

        for submission in submissions:
            created_utc = datetime.utcfromtimestamp(submission.created_utc)
            if created_utc < since_time:
                continue

            combined_text = f"{submission.title} {submission.selftext}"

            posts.append(
                {
                    "date": created_utc.isoformat(),
                    "text": combined_text,
                }
            )
        df = pd.DataFrame(posts)
        self.data.append(df)
        return df

    def scrap_finviz(self, ticker, days=3):
        finviz_url = "https://finviz.com/quote.ashx?t="
        headers = {"User-Agent": "Mozilla/5.0"}
        url = finviz_url + ticker
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            html = BeautifulSoup(response.text, features="html.parser")
            news_table = html.find(id="news-table")

            parsed_data = []
            previous_date = None

            for row in news_table.findAll("tr"):
                title = row.a.text.strip()
                date_data = row.td.text.strip()

                if len(date_data.split()) == 1:
                    time_part = date_data
                    date_part = previous_date
                else:
                    date_part, time_part = date_data.split(" ", 1)
                    previous_date = date_part

                if date_part.lower() == "today":
                    date_part = datetime.now().strftime("%b-%d-%y")
                elif date_part.lower() == "yesterday":
                    date_part = (datetime.now() - timedelta(days=1)).strftime(
                        "%b-%d-%y"
                    )

                date_time_str = f"{date_part} {time_part}"
                date_time_obj = datetime.strptime(date_time_str, "%b-%d-%y %I:%M%p")

                if date_time_obj >= datetime.now() - timedelta(days=days):
                    parsed_data.append([date_time_obj.isoformat(), title])

            df = pd.DataFrame(parsed_data, columns=["date", "text"])
            self.data.append(df)
            return df
        else:
            print(
                f"Failed to fetch data for {ticker}. HTTP status code: {response.status_code}"
            )
            return pd.DataFrame()

    async def _fetch_tweets(self, query, existing_tweets=None):
        if existing_tweets is None:
            print(f"{datetime.now()} - Fetching tweets for query: {query}")
            return await self.twitter_client.search_tweet(query, product="Latest")
        else:
            wait_time = randint(5, 10)
            print(
                f"{datetime.now()} - Waiting {wait_time} seconds before fetching next batch..."
            )
            await asyncio.sleep(wait_time)
            return await existing_tweets.next()

    async def _handle_rate_limit(self, e):
        rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
        wait_time = (rate_limit_reset - datetime.now()).total_seconds()
        if wait_time > 0:
            print(
                f"{datetime.now()} - Rate limit reached. Waiting for {wait_time} seconds."
            )
            await asyncio.sleep(wait_time)

    async def scrap_twitter(self, stock_name, days=1, max_tweets=500):
        since_time = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(
            days=days
        )
        query = f'"{stock_name}" lang:en min_faves:20 -filter:links -filter:replies'

        tweet_count = 0
        tweets = None
        posts = []

        while tweet_count < max_tweets:
            try:
                tweets = await self._fetch_tweets(query, tweets)
            except TooManyRequests as e:
                await self._handle_rate_limit(e)
                continue

            if not tweets:
                print(f"{datetime.now()} - No more tweets found.")
                break

            for tweet in tweets:
                created_at = tweet.created_at

                if isinstance(created_at, str):
                    created_at = datetime.strptime(
                        created_at, "%a %b %d %H:%M:%S %z %Y"
                    )
                if created_at < since_time:
                    print(
                        f"{datetime.now()} - Reached tweets older than the time range. Stopping collection."
                    )
                    df = pd.DataFrame(posts)
                    self.data.append(df)
                    return df

                tweet_count += 1
                posts.append(
                    {
                        "date": created_at.isoformat(),
                        "text": tweet.text,
                    }
                )

                if tweet_count >= max_tweets:
                    print(
                        f"{datetime.now()} - Reached the maximum number of tweets ({max_tweets}). Stopping collection."
                    )
                    break

            print(f"{datetime.now()} - Collected {tweet_count} tweets.")

        df = pd.DataFrame(posts)
        self.data.append(df)
        return df

    def remove_emojis(self, text):
        text = emoji.replace_emoji(text, "")

        # Remove common text-based emoticons
        emoticons_pattern = re.compile(r"[:;=]-?[)(/\\|dpDP]|[)(/<>]{}")
        text = emoticons_pattern.sub("", text)

        return text

    def clean_text_data(self, df, text_column="text", similarity_threshold=0.9):
        df = df.copy()
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        # Ensure text column is a string and fill NaNs
        df[text_column] = df[text_column].fillna("").astype(str)

        # Remove unnecessary phrases and spam-like content
        spam_keywords = [
            r"\bBREAKING NEWS\b",
            r"\bBUY NOW\b",
            r"\bLIMITED TIME\b",
            r"\bFREE\b",
            r"\bMARKET CAP\b",
            r"\bTRILLION\b",
            r"\bURGENT\b",
            r"\bFLASH\b",
            r"\bCEO\b",
        ]
        df = df[
            ~df[text_column].str.contains(
                "|".join(spam_keywords), flags=re.IGNORECASE, regex=True
            )
        ]

        # Remove links, mentions, and tickers
        df[text_column] = df[text_column].str.replace(
            r"http\S+|www.\S+", "", regex=True
        )  # URLs
        df[text_column] = df[text_column].str.replace(
            r"@\w+", "", regex=True
        )  # Mentions
        df[text_column] = df[text_column].str.replace(
            r"\$\w+", "", regex=True
        )  # Tickers ($TSLA)

        # Remove emojis, special characters, and excessive whitespace
        df[text_column] = df[text_column].apply(self.remove_emojis)
        df[text_column] = df[text_column].str.replace(
            r"[^\w\s]", " ", regex=True
        )  # Non-alphanumeric characters
        df[text_column] = (
            df[text_column].str.replace(r"\s+", " ", regex=True).str.strip()
        )  # Excess whitespace

        # Convert to lowercase for uniformity
        df[text_column] = df[text_column].str.lower()

        # Remove short texts and filter by length
        df = df[df[text_column].str.len() > 20]

        # Remove stopwords for sentiment analysis
        stop_words = set(stopwords.words("english"))
        df[text_column] = df[text_column].apply(
            lambda x: " ".join(word for word in x.split() if word not in stop_words)
        )

        # Remove duplicate and near-duplicate texts
        df = df.drop_duplicates(subset=[text_column])

        # Remove near-duplicates based on cosine similarity
        if len(df) > 1:
            tfidf = TfidfVectorizer().fit_transform(df[text_column])
            pairwise_sim = cosine_similarity(tfidf)

            to_drop = set()
            for idx in range(pairwise_sim.shape[0]):
                if idx in to_drop:
                    continue
                duplicates = np.where(pairwise_sim[idx] > similarity_threshold)[0]
                duplicates = [i for i in duplicates if i != idx and i > idx]
                to_drop.update(duplicates)

            df = df.drop(df.index[list(to_drop)])

        # Reset index for consistency
        df = df.reset_index(drop=True)

        return df

    def analyze_sentiment(self, df, text_column="text"):
        # Load FinBERT tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = AutoModelForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone"
        )

        # Set up the sentiment analysis pipeline
        sentiment_analysis = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer
        )

        # Define the maximum token length for FinBERT
        max_length = 512

        # Function to truncate text that exceeds the maximum token length
        def truncate_text(text):
            # Tokenize and keep only the allowed length
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            return tokenizer.decode(tokens, skip_special_tokens=True)

        # Apply sentiment analysis to each truncated text entry
        sentiments = df[text_column].apply(
            lambda text: sentiment_analysis(truncate_text(text))[0]
        )

        # Extract sentiment label and score from the result
        df["sentiment_label"] = sentiments.apply(lambda x: x["label"].lower())
        df["sentiment_score"] = sentiments.apply(lambda x: x["score"])

        # Aggregate the sentiment labels
        sentiment_counts = df["sentiment_label"].value_counts()

        # Debugging output
        print("Sentiment counts:", sentiment_counts)

        # Determine overall sentiment
        positive = sentiment_counts.get("positive", 0)
        negative = sentiment_counts.get("negative", 0)
        neutral = sentiment_counts.get("neutral", 0)

        total = positive + negative + neutral

        if total == 0:
            overall_sentiment = "neutral"
        else:
            if positive > negative:
                overall_sentiment = "positive"
            elif negative > positive:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"

        # Decide whether stock will go up or down based on overall sentiment
        if overall_sentiment == "positive":
            prediction = "The stock is predicted to go UP based on the sentiments."
        elif overall_sentiment == "negative":
            prediction = "The stock is predicted to go DOWN based on the sentiments."
        else:
            prediction = "The stock sentiment is NEUTRAL."

        return df, prediction
