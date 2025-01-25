import os
import time
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
import praw

# Twitter
from twikit import Client, TooManyRequests

# Text similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK
from nltk.corpus import stopwords

# Transformers (FinBERT)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Dotenv
from dotenv import load_dotenv, dotenv_values

# Load environment variables
load_dotenv()


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
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )

        self.twitter_client = Client(language="en-US")
        self.twitter_client.load_cookies("cookies.json")  # Twitter cookies

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        self.stop_words = set(stopwords.words("english"))

    ############################################################################
    #                           REDDIT SCRAPING
    ############################################################################

    def scrap_reddit(self, stock_name, days=7):
        """
        Scrapes Reddit for mentions of `stock_name` within the past `days`.
        Returns a DataFrame with columns: [date, text, upvotes]
        """
        posts = []
        now = datetime.utcnow()
        since_time = now - timedelta(days=days)

        # Use the search query with a time_filter if it fits the PRAW scheme
        # Valid time_filters: 'hour', 'day', 'week', 'month', 'year', 'all'
        # We'll pick 'week' if days <=7, else 'month' or 'year'
        if days <= 7:
            time_filter = "week"
        elif days <= 30:
            time_filter = "month"
        elif days <= 365:
            time_filter = "year"
        else:
            time_filter = "all"

        search_query = f'"{stock_name}"'
        submissions = self.reddit.subreddit("all").search(
            search_query, sort="new", time_filter=time_filter, limit=None
        )

        for submission in submissions:
            created_utc = datetime.utcfromtimestamp(submission.created_utc)
            # Extra check in code (some time_filter can still bring older posts)
            if created_utc < since_time:
                continue

            combined_text = f"{submission.title} {submission.selftext}"
            # Upvote details
            upvotes = submission.score if submission.score else 0

            posts.append(
                {
                    "date": created_utc.isoformat(),
                    "text": combined_text,
                    "upvotes": upvotes,
                }
            )

        df = pd.DataFrame(posts)
        # If there's no data, just return empty
        if df.empty:
            print("No Reddit data found for the specified date range.")
            return df

        self.data.append(df)
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
        If `weight_column` is provided (e.g. "upvotes" or "likes"),
        it computes a weighted sentiment. Otherwise, it uses simple counts.
        Returns:
          - The updated DataFrame with sentiment_label and sentiment_score
          - A final textual prediction ("UP", "DOWN", or "NEUTRAL").
        """
        if df.empty:
            print("No data to analyze for sentiment. Defaulting to NEUTRAL.")
            return df, "The stock sentiment is NEUTRAL. (No data)"

        # Load FinBERT
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = AutoModelForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone"
        )
        sentiment_analysis = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer
        )
        max_length = 512

        def truncate_text(text):
            # Tokenize + keep only the allowed length
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            return tokenizer.decode(tokens, skip_special_tokens=True)

        # Run FinBERT on each text row
        sentiments = df[text_column].apply(
            lambda t: sentiment_analysis(truncate_text(t))[0]
        )
        df["sentiment_label"] = sentiments.apply(lambda x: x["label"].lower())
        df["sentiment_score"] = sentiments.apply(lambda x: x["score"])

        # Weighted approach
        # If there's no weight_column, default all weights to 1
        if weight_column and weight_column in df.columns:
            df["weight"] = df[weight_column].fillna(1).astype(float).clip(lower=1)
        else:
            df["weight"] = 1.0

        # Summation structures
        weighted_positive = 0.0
        weighted_negative = 0.0
        weighted_neutral = 0.0
        total_weight = 0.0

        for _, row in df.iterrows():
            label = row["sentiment_label"]
            w = row["weight"]
            total_weight += w

            if label == "positive":
                weighted_positive += w
            elif label == "negative":
                weighted_negative += w
            else:
                weighted_neutral += w

        # Avoid divide-by-zero
        if total_weight == 0:
            return df, "The stock sentiment is NEUTRAL. (No weights)"

        # Weighted proportions
        pos_share = weighted_positive / total_weight
        neg_share = weighted_negative / total_weight
        neu_share = weighted_neutral / total_weight

        # Debug
        print("Weighted sentiment distribution:")
        print(f"  Positive: {pos_share:.2f}")
        print(f"  Negative: {neg_share:.2f}")
        print(f"  Neutral:  {neu_share:.2f}")

        # Final logic
        # You can adjust thresholds here. E.g., you might require
        # pos_share > 0.60 for "UP", neg_share > 0.60 for "DOWN".
        if pos_share > neg_share and pos_share >= 0.4:
            prediction = "The stock is predicted to go UP based on the sentiments."
        elif neg_share > pos_share and neg_share >= 0.4:
            prediction = "The stock is predicted to go DOWN based on the sentiments."
        else:
            prediction = "The stock sentiment is NEUTRAL."

        return df, prediction


# ------------------ Example usage (synchronous part) ------------------ #
# if __name__ == "__main__":
#     """
#     Simple demonstration of how you might use the 'Scrap' class.
#     """
#     scraper = Scrap()

#     # ---- Reddit ----
#     reddit_data = scraper.scrap_reddit("TSLA", days=3)
#     reddit_data_clean = scraper.clean_text_data(reddit_data, text_column="text")

#     # ---- FinViz ----
#     finviz_data = scraper.scrap_finviz("TSLA", days=3)
#     finviz_data_clean = scraper.clean_text_data(finviz_data, text_column="text")

#     # ---- Twitter (needs async run) ----
#     async def get_twitter_data():
#         return await scraper.scrap_twitter("TSLA", days=1, max_tweets=100)

#     loop = asyncio.get_event_loop()
#     twitter_data = loop.run_until_complete(get_twitter_data())
#     twitter_data_clean = scraper.clean_text_data(twitter_data, text_column="text")

#     # Combine all data
#     combined_df = pd.concat([reddit_data_clean, finviz_data_clean, twitter_data_clean], ignore_index=True)

#     # If you want to do a single sentiment pass on combined data:
#     # Weighted by upvotes for Reddit, likes for Twitter. If both columns exist, pick one or sum them.
#     # We'll unify them under a single column called 'weight' for demonstration:
#     # Priority: 'upvotes' if they exist, else 'likes'
#     if "upvotes" in combined_df.columns:
#         combined_df["weight"] = combined_df.get("upvotes", 1)
#     elif "likes" in combined_df.columns:
#         combined_df["weight"] = combined_df.get("likes", 1)
#     else:
#         combined_df["weight"] = 1

#     # Analyze
#     sentiment_df, final_prediction = scraper.analyze_sentiment(
#         combined_df, text_column="text", weight_column="weight"
#     )

#     print("Final Prediction:", final_prediction)
