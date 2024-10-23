import praw
import pandas as pd
import datetime as dt

# Reddit API credentials
reddit = praw.Reddit(
    client_id="P6oB-vE585YHt7TSTw_TAA",
    client_secret="8oXVjhMyis5vgqF17_HTXqPQC13umg",
    user_agent="Stocks sentiment analysis",
)

# get 100 hot posts the previous two days , about AAPL stock

subreddit = reddit.subreddit("AAPL")
top_subreddit = subreddit.top(limit=100)
topics_dict = {
    "title": [],
    "score": [],
    "id": [],
    "url": [],
    "comms_num": [],
    "created": [],
    "body": [],
}

for submission in top_subreddit:
    topics_dict["title"].append(submission.title)
    topics_dict["score"].append(submission.score)
    topics_dict["id"].append(submission.id)
    topics_dict["url"].append(submission.url)
    topics_dict["comms_num"].append(submission.num_comments)
    topics_dict["created"].append(submission.created)
    topics_dict["body"].append(submission.selftext)

topics_data = pd.DataFrame(topics_dict)
topics_data["created"] = topics_data["created"].apply(
    lambda x: dt.datetime.fromtimestamp(x)
)
topics_data.to_csv("reddit_aapl.csv", index=False)
