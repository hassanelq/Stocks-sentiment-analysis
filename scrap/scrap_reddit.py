import praw
import pandas as pd
import datetime as dt

# Reddit API credentials
reddit = praw.Reddit(
    client_id="P6oB-vE585YHt7TSTw_TAA",
    client_secret="8oXVjhMyis5vgqF17_HTXqPQC13umg",
    user_agent="Stocks sentiment analysis",
)


def scrap_reddit(stock_name, days=7):
    """
    Scrapes Reddit for all posts related to a stock within the last N days.

    Parameters:
    stock_name (str): The stock ticker to search for.
    days (int): Number of days back to scrape.

    Returns:
    pandas.DataFrame: DataFrame containing scraped Reddit posts.
    """

    posts = []
    now = dt.datetime.utcnow()
    since_time = now - dt.timedelta(days=days)
    since_timestamp = int(since_time.timestamp())

    search_query = f'"{stock_name}"'  # Exact match search

    # Reddit API limits search results to approximately 1000 posts per query
    submissions = reddit.subreddit("all").search(
        search_query, sort="new", time_filter="all", limit=None
    )

    for submission in submissions:
        created_utc = dt.datetime.utcfromtimestamp(submission.created_utc)
        if created_utc < since_time:
            continue  # Skip posts older than the specified number of days

        posts.append(
            {
                "subreddit": submission.subreddit.display_name,
                "title": submission.title,
                "selftext": submission.selftext,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "created_utc": created_utc,
                "created_date": created_utc.strftime("%Y-%m-%d"),
                "created_time": created_utc.strftime("%H:%M:%S"),
                "permalink": f"https://reddit.com{submission.permalink}",
            }
        )

    return pd.DataFrame(posts)


# Example usage
df = scrap_reddit("BTC", days=1)
print(df)
df.to_csv(f"reddit_{dt.datetime.now().strftime('%d')}.csv", index=False)
