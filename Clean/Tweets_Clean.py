import pandas as pd
import numpy as np
import re

tweets = pd.read_csv("twitterdata.csv")


def clean_twitter_data(df):
    """
    Cleans Twitter data by removing unwanted tweets and cleaning the tweet text.

    Parameters:
    df (pd.DataFrame): DataFrame containing at least a 'tweet' column.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Ensure 'tweet' column is of type string
    df["tweet"] = df["tweet"].astype(str)

    # Step 1: Remove tweets that contain "BREAKING NEWS" or are repetitive/spam-like
    df = df[
        ~df["tweet"].str.contains(r"\bBREAKING NEWS\b", flags=re.IGNORECASE, regex=True)
    ]

    # Remove duplicate tweets
    df = df.drop_duplicates(subset="tweet")

    # Step 2: Filter out non-informative tweets (e.g., very short tweets or only tickers)
    df = df[df["tweet"].str.len() > 20]  # Keep tweets with more than 20 characters

    # Step 3: Remove tweets that are promotional or uninformative
    promotional_keywords = [
        r"\bCEO\b",
        r"\bimpressive\b",
        r"\bexplosively growing\b",
        r"\bmarket cap\b",
        r"\bTRILLION\b",
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

    # Remove tickers like $AAPL, $NVDA
    df["tweet"] = df["tweet"].str.replace(r"\$\w+", "", regex=True)

    # Remove non-alphanumeric characters (except spaces)
    df["tweet"] = df["tweet"].str.replace(r"[^\w\s]", "", regex=True)

    # Remove extra whitespaces and trim text
    df["tweet"] = df["tweet"].str.strip()
    df["tweet"] = df["tweet"].str.replace(r"\s+", " ", regex=True)

    # Convert text to lowercase
    df["tweet"] = df["tweet"].str.lower()

    # **NEW STEP**: Remove duplicates again after cleaning
    df = df.drop_duplicates(subset="tweet")

    # Reset the index of the DataFrame
    df = df.reset_index(drop=True)

    return df


# Example usage
cleaned_df = clean_twitter_data(tweets)

# Save cleaned data to a new CSV file
cleaned_df.to_csv("cleaned_twitter_data.csv", index=False)
