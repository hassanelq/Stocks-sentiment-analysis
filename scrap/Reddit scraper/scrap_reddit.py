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
                # "subreddit": submission.subreddit.display_name,
                "title": submission.title,
                "selftext": submission.selftext,
                "created_utc": created_utc,
                "score": submission.score,
                "num_comments": submission.num_comments,
            }
        )

    return pd.DataFrame(posts)


# Example usage
df = scrap_reddit("AAPL", days=2)
print(df)
df.to_csv(f"reddit_{dt.datetime.now().strftime('%d')}.csv", index=False)
