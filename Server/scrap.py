import re
import asyncio
import requests
import numpy as np
import pandas as pd
import nltk
import emoji
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


REDDIT_CLIENT_ID = "XXX"
REDDIT_CLIENT_SECRET = "XXX"
REDDIT_USER_AGENT = "Stocks sentiment analysis"

cookies = {
    "XXX": "XXX",
}

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone"
)
finbert_pipeline = pipeline(
    "sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer
)


class Scrap:
    """
    A class to scrape data from Reddit, FinViz, and Twitter,
    clean the text, analyze sentiments using FinBERT,
    and provide a final 'Up', 'Down', or 'Neutral' prediction.
    """

    def __init__(self):
        """
        Initialize:
          - Data containers
          - Reddit client
          - Twitter client
          - NLTK resources
        """
        self.data = []
        self.reddit = asyncpraw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )

        self.twitter_client = Client(language="en-US")
        self.twitter_client.set_cookies(cookies)

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        self.stop_words = set(stopwords.words("english"))

    async def close(self):
        await self.reddit.close()

    ############################################################################
    #                           REDDIT SCRAPING
    ############################################################################

    async def scrap_reddit(self, stock_name, days=7):
        posts = []
        now = datetime.now(timezone.utc)
        since_time = now - timedelta(days=days)

        if days <= 7:
            time_filter = "week"
        elif days <= 30:
            time_filter = "month"
        elif days <= 365:
            time_filter = "year"
        else:
            time_filter = "all"

        search_query = f'"{stock_name}"'
        subreddit = await self.reddit.subreddit("all")
        async for submission in subreddit.search(
            search_query, sort="new", time_filter=time_filter, limit=None
        ):
            created_utc = datetime.fromtimestamp(submission.created_utc, timezone.utc)
            if created_utc < since_time:
                continue

            combined_text = f"{submission.title} {submission.selftext}"
            upvotes = submission.score or 0

            posts.append(
                {
                    "date": created_utc.isoformat(),
                    "text": combined_text,
                    "upvotes": upvotes,
                }
            )

        df = pd.DataFrame(posts)
        if df.empty:
            print("No Reddit data found for the specified date range.")
        return df

    ############################################################################
    #                           FINVIZ SCRAPING
    ############################################################################

    def scrap_finviz(self, ticker, days=3):
        """
        Scrapes the FinViz news table for a given `ticker`.
        Returns a DataFrame with columns: [date, text]
        """
        finviz_url = "https://finviz.com/quote.ashx?t="
        headers = {"User-Agent": "Mozilla/5.0"}
        url = finviz_url + ticker
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(
                f"Failed to fetch data for {ticker}. HTTP status code: {response.status_code}"
            )
            return pd.DataFrame()

        html = BeautifulSoup(response.text, features="html.parser")
        news_table = html.find(id="news-table")
        if not news_table:
            print(f"No news table found on FinViz for ticker: {ticker}")
            return pd.DataFrame()

        parsed_data = []
        now = datetime.now()
        cutoff_date = now - timedelta(days=days)

        previous_date = None
        for row in news_table.findAll("tr"):
            # Guard clauses if row doesn't have the expected structure
            if not row.a or not row.td:
                continue

            title = row.a.text.strip()
            date_data = row.td.text.strip()

            # Some rows show only time
            if len(date_data.split()) == 1:
                time_part = date_data
                date_part = previous_date
            else:
                # Full date time
                date_part, time_part = date_data.split(" ", 1)
                previous_date = date_part

            # Replace 'Today' or 'Yesterday' for clarity
            if date_part.lower() == "today":
                date_part = now.strftime("%b-%d-%y")
            elif date_part.lower() == "yesterday":
                date_part = (now - timedelta(days=1)).strftime("%b-%d-%y")

            date_time_str = f"{date_part} {time_part}"
            try:
                date_time_obj = datetime.strptime(date_time_str, "%b-%d-%y %I:%M%p")
            except ValueError:
                # Sometimes FinViz changes formats slightly
                continue

            if date_time_obj < cutoff_date:
                # If we've gone past the date range, we can break (assuming chronological order)
                break

            parsed_data.append([date_time_obj.isoformat(), title])

        df = pd.DataFrame(parsed_data, columns=["date", "text"])
        if df.empty:
            print(f"No FinViz data found for {ticker} in last {days} days.")
            return df

        self.data.append(df)
        return df

    ############################################################################
    #                           TWITTER SCRAPING
    ############################################################################

    async def _fetch_tweets(self, query, existing_tweets=None):
        """
        Fetch tweets from Twitter. If existing_tweets is None,
        it starts a new query; otherwise, it fetches the next batch.
        """
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
        """
        Handle Twitter's rate limit. Wait until the limit resets.
        """
        rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
        wait_time = (rate_limit_reset - datetime.now()).total_seconds()
        if wait_time > 0:
            print(
                f"{datetime.now()} - Rate limit reached. Waiting for {wait_time} seconds."
            )
            await asyncio.sleep(wait_time)

    async def scrap_twitter(self, stock_name, days=1, max_tweets=300):
        """
        Async method to scrape Twitter for `stock_name` within `days`.
        Returns a DataFrame with columns: [date, text, likes]
        """
        since_time = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(
            days=days
        )
        query = f'"{stock_name}" lang:en min_faves:10 -filter:links -filter:replies'
        # min_faves:10 ensures the tweet has at least some likes

        tweet_count = 0
        tweets = None
        posts = []

        while tweet_count < max_tweets:
            try:
                tweets = await self._fetch_tweets(query, tweets)
            except TooManyRequests as e:
                await self._handle_rate_limit(e)
                continue
            except Exception as ex:
                print(f"Unexpected error while fetching tweets: {ex}")
                break

            if not tweets:
                print(f"{datetime.now()} - No more tweets found.")
                break

            for tweet in tweets:
                created_at = tweet.created_at
                if isinstance(created_at, str):
                    # Format: "Tue Jan 24 14:42:39 +0000 2023"
                    created_at = datetime.strptime(
                        created_at, "%a %b %d %H:%M:%S %z %Y"
                    )

                if created_at < since_time:
                    print(
                        f"{datetime.now()} - Reached tweets older than {days} days. Stopping collection."
                    )
                    return pd.DataFrame(posts)

                tweet_count += 1
                # Attempt to get 'favorite_count' if the library provides it
                # If not, we store a placeholder 1 or 0
                likes_count = getattr(tweet, "favorite_count", 1) or 1

                posts.append(
                    {
                        "date": created_at.isoformat(),
                        "text": tweet.text,
                        "likes": likes_count,
                    }
                )

                if tweet_count >= max_tweets:
                    print(
                        f"{datetime.now()} - Reached the maximum number of tweets ({max_tweets}). Stopping."
                    )
                    break

            print(f"{datetime.now()} - Collected {tweet_count} tweets so far.")

        df = pd.DataFrame(posts)
        if df.empty:
            print("No Twitter data found for the specified query/time range.")
            return df

        self.data.append(df)
        return df

    ############################################################################
    #                           TEXT CLEANING
    ############################################################################

    def remove_emojis(self, text):
        """
        Remove emojis from text using the 'emoji' library.
        Also removes common emoticons using regex.
        """
        text = emoji.replace_emoji(text, "")  # remove unicode emojis

        # Remove common text-based emoticons
        # Expand pattern to cover more emoticons if needed
        emoticons_pattern = re.compile(r"[:;=]-?[()\/\\|dpDP]|[()<>}{]")
        text = emoticons_pattern.sub("", text)
        return text

    def clean_text_data(self, df, text_column="text", similarity_threshold=0.9):
        """
        Cleans the text in `df[text_column]` using:
          - Remove spam keywords
          - Remove URLs, mentions, tickers
          - Remove emojis/special chars
          - Remove duplicates and near-duplicates
          - Remove stopwords
        Returns the cleaned DataFrame.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        # Fill NaNs and enforce string type
        df[text_column] = df[text_column].fillna("").astype(str)

        # Remove spam-like phrases
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

        # Remove URLs, mentions, tickers
        df[text_column] = df[text_column].str.replace(
            r"http\S+|www.\S+", "", regex=True
        )  # URLs
        df[text_column] = df[text_column].str.replace(
            r"@\w+", "", regex=True
        )  # Mentions
        df[text_column] = df[text_column].str.replace(
            r"\$\w+", "", regex=True
        )  # Tickers ($TSLA)

        # Remove emojis, punctuation, special characters
        df[text_column] = df[text_column].apply(self.remove_emojis)
        df[text_column] = df[text_column].str.replace(r"[^\w\s]", " ", regex=True)

        # Convert to lower case
        df[text_column] = df[text_column].str.lower()

        # Remove extra whitespace
        df[text_column] = (
            df[text_column].str.replace(r"\s+", " ", regex=True).str.strip()
        )

        # Remove short texts
        df = df[df[text_column].str.len() > 20]

        # Remove stopwords
        df[text_column] = df[text_column].apply(
            lambda x: " ".join(
                word for word in x.split() if word not in self.stop_words
            )
        )

        # Drop exact duplicates
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
                # Exclude itself and only remove duplicates with higher index (so we keep the first occurrence)
                duplicates = [i for i in duplicates if i != idx and i > idx]
                to_drop.update(duplicates)

            df = df.drop(df.index[list(to_drop)])

        df = df.reset_index(drop=True)
        return df

    ############################################################################
    #                           SENTIMENT ANALYSIS
    ############################################################################

    def analyze_sentiment(self, df, text_column="text", weight_column=None):
        """
        Uses FinBERT to analyze sentiment (positive/negative/neutral).
        Returns DataFrame with sentiment columns + final string prediction.
        """
        if df.empty:
            print("No data to analyze for sentiment. Defaulting to NEUTRAL.")
            return df, "The stock sentiment is NEUTRAL. (No data)"

        def truncate_text(text):
            tokens = finbert_tokenizer.encode(text, truncation=True, max_length=512)
            return finbert_tokenizer.decode(tokens, skip_special_tokens=True)

        try:
            sentiments = df[text_column].apply(
                lambda t: finbert_pipeline(truncate_text(t))[0]
            )
            df["sentiment_label"] = sentiments.apply(lambda x: x["label"].lower())
            df["sentiment_score"] = sentiments.apply(lambda x: x["score"])
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            df["sentiment_label"] = "neutral"
            df["sentiment_score"] = 0

        # Set weights (likes, upvotes, or default)
        if weight_column and weight_column in df.columns:
            df["weight"] = df[weight_column].fillna(1).astype(float).clip(lower=1)
        else:
            df["weight"] = 1.0

        sentiment_weights = df.groupby("sentiment_label")["weight"].sum()
        total_weight = sentiment_weights.sum()

        pos = sentiment_weights.get("positive", 0.0)
        neg = sentiment_weights.get("negative", 0.0)
        neu = sentiment_weights.get("neutral", 0.0)

        if total_weight == 0:
            pos_share = neg_share = neu_share = 0
        else:
            pos_share = pos / total_weight
            neg_share = neg / total_weight
            neu_share = neu / total_weight

        print("Weighted sentiment distribution:")
        print(f"  Positive: {pos_share:.2f}")
        print(f"  Negative: {neg_share:.2f}")
        print(f"  Neutral:  {neu_share:.2f}")

        # Final prediction
        if max(pos_share, neg_share, neu_share) < 0.4:
            prediction = "The stock sentiment is NEUTRAL."
        elif pos_share > neg_share:
            prediction = "The stock is predicted to go UP based on the sentiments."
        elif neg_share > pos_share:
            prediction = "The stock is predicted to go DOWN based on the sentiments."
        else:
            prediction = "The stock sentiment is NEUTRAL."

        return df, prediction
