import pandas as pd
import asyncio
from twikit import Client, TooManyRequests
from datetime import datetime, timedelta
from random import randint
import time
from configparser import ConfigParser
from datetime import timezone


# Load configuration for authentication
def load_config(file="config.ini"):
    config = ConfigParser()
    config.read(file)
    return config["X"]["username"], config["X"]["email"], config["X"]["password"]


# Initialize Twitter (X.com) client
def authenticate_client():
    client = Client(language="en-US")
    client.load_cookies("cookies.json")
    return client


# Construct the search query for the specified stock
def construct_query(stock_name):
    return f'"{stock_name}" lang:en min_faves:20 lang:en -filter:links -filter:replies'


# Fetch tweets, with rate limit handling
async def fetch_tweets(client, query, existing_tweets=None):
    if existing_tweets is None:
        print(f"{datetime.now()} - Fetching tweets for query: {query}")
        return await client.search_tweet(query, product="Latest")
    else:
        wait_time = randint(5, 10)
        print(
            f"{datetime.now()} - Waiting {wait_time} seconds before fetching next batch..."
        )
        time.sleep(wait_time)
        return await existing_tweets.next()


# Handle rate limits by pausing scraping until the limit resets
def handle_rate_limit(e):
    rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
    wait_time = rate_limit_reset - datetime.now()
    print(f"{datetime.now()} - Rate limit reached. Waiting until {rate_limit_reset}")
    time.sleep(wait_time.total_seconds())


# Core scraping function
async def scrape_twitter(client, stock_name, days=1, max_tweets=500):
    # Convert `since_time` to an offset-aware datetime in UTC
    since_time = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=days)
    query = construct_query(stock_name)

    tweet_count = 0
    tweets = None
    posts = []

    # Collect tweets until the maximum number of tweets is reached or there are no more tweets
    while tweet_count < max_tweets:
        try:
            tweets = await fetch_tweets(client, query, tweets)
        except TooManyRequests as e:
            handle_rate_limit(e)
            continue

        if not tweets:
            print(f"{datetime.now()} - No more tweets found.")
            break

        for tweet in tweets:
            created_at = tweet.created_at

            # Stop collecting tweets if they are older than the 'since' time
            if isinstance(created_at, str):
                created_at = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
            if created_at < since_time:
                print(
                    f"{datetime.now()} - Reached tweets older than the time range. Stopping collection."
                )
                return pd.DataFrame(posts)

            tweet_count += 1
            posts.append(
                {
                    "username": tweet.user.name,
                    "text": tweet.text,
                    "created_at": created_at.strftime("%Y-%m-%d %H:%M"),
                    "retweets": tweet.retweet_count,
                    "likes": tweet.favorite_count,
                }
            )

            if tweet_count >= max_tweets:
                print(
                    f"{datetime.now()} - Reached the maximum number of tweets ({max_tweets}). Stopping collection."
                )
                break

        print(f"{datetime.now()} - Collected {tweet_count} tweets.")

    return pd.DataFrame(posts)


# Save the collected tweets to a CSV file
def save_to_csv(df, stock_name):
    if not df.empty:
        filename = f"twitter_{stock_name}_{datetime.now().strftime('%d-%m-%Y')}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
    else:
        print("No tweets found.")


# Main function to run the scraping process
async def main(stock_name, days=1, max_tweets=100):
    client = authenticate_client()  # Authenticate
    df = await scrape_twitter(client, stock_name, days, max_tweets)  # Scrape tweets
    save_to_csv(df, stock_name)  # Save results to CSV


# Example usage
if __name__ == "__main__":
    stock_name = "AAPL"  # Replace with the stock or keyword to search for
    days = 2  # Number of days to go back
    max_tweets = 500  # Maximum number of tweets to collect

    asyncio.run(main(stock_name, days, max_tweets))
