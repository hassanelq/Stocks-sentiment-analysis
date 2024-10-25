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
df = scrap_reddit("AAPL", days=3)
print(df)
df.to_csv(f"reddit_{dt.datetime.now().strftime('%d')}.csv", index=False)


# def scrap_reddit(stock_name, num_posts):
#     # Access the subreddit for the given stock name
#     subreddit = reddit.subreddit(stock_name)

#     # Get the latest posts from the subreddit, limited by num_posts
#     new_subreddit = subreddit.new(limit=num_posts)

#     # Dictionary to store the data we want to collect
#     topics_dict = {
#         "title": [],
#         "score": [],
#         "url": [],
#         "comms_num": [],
#         "created_date": [],
#         "created_time": [],
#         "body": [],
#     }

#     # Loop through the latest posts and collect the required information
#     for submission in new_subreddit:
#         topics_dict["title"].append(submission.title)
#         topics_dict["score"].append(submission.score)
#         topics_dict["url"].append(submission.url)
#         topics_dict["comms_num"].append(submission.num_comments)
#         topics_dict["created_date"].append(
#             dt.datetime.fromtimestamp(submission.created).strftime("%Y-%m-%d")
#         )
#         topics_dict["created_time"].append(
#             dt.datetime.fromtimestamp(submission.created).strftime("%H:%M:%S")
#         )
#         topics_dict["body"].append(submission.selftext)

#     # Convert the dictionary to a pandas DataFrame
#     topics_data = pd.DataFrame(topics_dict)

#     # Save the DataFrame to a CSV file
#     topics_data.to_csv(f"reddit_{stock_name}_{num_posts}.csv", index=False)
