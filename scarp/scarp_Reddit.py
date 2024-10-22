# reddit scapper using the secret key

import praw
import pandas as pd
import datetime as dt

# Reddit API credentials
reddit = praw.Reddit(
    client_id="yRpk1QZ515gSrfuFRJhaAg",
    client_secret="ExT8MUDnH3GUOMB9d-Oq01ov2aU8eA",
)

# Extracting data from Reddit
subreddit = reddit.subreddit("learnpython")
top_subreddit = subreddit.top(limit=500)

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
