import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_twitter_data(df, similarity_threshold=0.9):
    """
    Cleans Twitter data by removing unwanted tweets, cleaning the tweet text,
    and removing near-duplicate tweets.

    Parameters:
    df (pd.DataFrame): DataFrame containing at least a 'tweet' column.
    similarity_threshold (float): Threshold for cosine similarity to consider tweets as duplicates.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Ensure 'tweet' column is of type string
    df["tweet"] = df["tweet"].astype(str)

    # Step 1: Remove tweets that contain unwanted phrases or are repetitive/spam-like
    unwanted_phrases = [
        r"\bBREAKING NEWS\b",
        r"\bBREAKING\b",
        r"\bFLASH\b",
        r"\bURGENT\b",
    ]
    df = df[
        ~df["tweet"].str.contains(
            "|".join(unwanted_phrases), flags=re.IGNORECASE, regex=True
        )
    ]

    # Remove duplicate tweets
    df = df.drop_duplicates(subset="tweet")

    # Step 2: Filter out non-informative tweets (e.g., very short tweets or only tickers)
    df = df[df["tweet"].str.len() > 20]  # Keep tweets with more than 20 characters

    # Remove tweets with excessive uppercase letters (likely spam)
    df = df[
        df["tweet"].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) < 0.5)
    ]

    # Step 3: Remove tweets that are promotional or uninformative
    promotional_keywords = [
        r"\bCEO\b",
        r"\bimpressive\b",
        r"\bexplosively growing\b",
        r"\bmarket cap\b",
        r"\bTRILLION\b",
        r"\bBUY NOW\b",
        r"\bFREE\b",
        r"\bLIMITED TIME\b",
    ]
    df = df[
        ~df["tweet"].str.contains(
            "|".join(promotional_keywords), flags=re.IGNORECASE, regex=True
        )
    ]

    # Step 4: Clean tweet text
    # Remove URLs
    df["tweet"] = df["tweet"].str.replace(r"http\S+|www.\S+", "", regex=True)

    # Remove mentions
    df["tweet"] = df["tweet"].str.replace(r"@\w+", "", regex=True)

    # Remove hashtags
    df["tweet"] = df["tweet"].str.replace(r"#\w+", "", regex=True)

    # Remove tickers like $AAPL, $NVDA
    df["tweet"] = df["tweet"].str.replace(r"\$\w+", "", regex=True)

    # Remove emojis and non-ASCII characters
    df["tweet"] = df["tweet"].str.encode("ascii", "ignore").str.decode("ascii")

    # Remove non-alphanumeric characters (except spaces)
    df["tweet"] = df["tweet"].str.replace(r"[^\w\s]", "", regex=True)

    # Remove numbers
    df["tweet"] = df["tweet"].str.replace(r"\d+", "", regex=True)

    # Remove extra whitespaces and trim text
    df["tweet"] = df["tweet"].str.strip()
    df["tweet"] = df["tweet"].str.replace(r"\s+", " ", regex=True)

    # Convert text to lowercase
    df["tweet"] = df["tweet"].str.lower()

    # Step 5: Remove stopwords (common words that add little value)
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    df["tweet"] = df["tweet"].apply(
        lambda x: " ".join(word for word in x.split() if word not in stop_words)
    )

    # Step 6: Remove duplicates again after cleaning
    df = df.drop_duplicates(subset="tweet")

    # Step 7: Remove near-duplicate tweets using cosine similarity
    # Compute TF-IDF vectors for each tweet
    tfidf = TfidfVectorizer().fit_transform(df["tweet"])
    pairwise_sim = cosine_similarity(tfidf)

    # Create a mask to identify duplicates
    to_drop = set()
    for idx in range(pairwise_sim.shape[0]):
        if idx in to_drop:
            continue
        duplicates = np.where(pairwise_sim[idx] > similarity_threshold)[0]
        duplicates = [i for i in duplicates if i != idx and i > idx]
        to_drop.update(duplicates)

    # Drop near-duplicate tweets
    df = df.drop(df.index[list(to_drop)])
    df = df.reset_index(drop=True)

    # Step 8: Filter out tweets with low engagement (optional)
    # You can set thresholds for likes and retweets if desired
    # df = df[(df['likes'] > 5) & (df['retweets'] > 1)]

    return df


# Example usage
tweets = pd.read_csv("twitterdata.csv")
cleaned_df = clean_twitter_data(tweets)

# Save cleaned data to a new CSV file
cleaned_df.to_csv("cleaned_twitter_data.csv", index=False)
