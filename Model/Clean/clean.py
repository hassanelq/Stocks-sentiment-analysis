import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import emoji

# Download stopwords if not already downloaded
nltk.download("stopwords")


def remove_emojis(text):
    """
    Removes emojis and emoticons from text.
    """
    # Remove Unicode emojis
    text = emoji.replace_emoji(text, "")

    # Remove common text-based emoticons
    emoticons_pattern = re.compile(r"[:;=]-?[)(/\\|dpDP]|[)(/<>]{}")
    text = emoticons_pattern.sub("", text)

    return text


def clean_text_data(df, text_column="text", similarity_threshold=0.9):
    """
    Cleans text data for sentiment analysis. This function is tailored for predicting
    stock price movements by focusing on removing noise, handling duplicate/similar
    texts, and preparing text for sentiment analysis.

    Parameters:
    df (pd.DataFrame): DataFrame containing the text data with at least a text column.
    text_column (str): The name of the column containing the text to clean.
    similarity_threshold (float): Cosine similarity threshold for removing near-duplicates.

    Returns:
    pd.DataFrame: Cleaned DataFrame with processed text data.
    """
    # Step 1: Basic text cleaning
    df = df.copy()
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    # Ensure text column is a string and fill NaNs
    df[text_column] = df[text_column].fillna("").astype(str)

    # Step 2: Remove unnecessary phrases and spam-like content
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

    # Step 3: Remove links, mentions, and tickers
    df[text_column] = df[text_column].str.replace(
        r"http\S+|www.\S+", "", regex=True
    )  # URLs
    df[text_column] = df[text_column].str.replace(r"@\w+", "", regex=True)  # Mentions
    df[text_column] = df[text_column].str.replace(
        r"\$\w+", "", regex=True
    )  # Tickers ($TSLA)

    # Step 4: Remove emojis, special characters, and excessive whitespace
    df[text_column] = df[text_column].apply(remove_emojis)
    df[text_column] = df[text_column].str.replace(
        r"[^\w\s]", " ", regex=True
    )  # Non-alphanumeric characters
    df[text_column] = (
        df[text_column].str.replace(r"\s+", " ", regex=True).str.strip()
    )  # Excess whitespace

    # Step 5: Convert to lowercase for uniformity
    df[text_column] = df[text_column].str.lower()

    # Step 6: Remove short texts and filter by length
    df = df[df[text_column].str.len() > 20]

    # Step 7: Remove stopwords for sentiment analysis
    stop_words = set(stopwords.words("english"))
    df[text_column] = df[text_column].apply(
        lambda x: " ".join(word for word in x.split() if word not in stop_words)
    )

    # Step 8: Remove duplicate and near-duplicate texts
    df = df.drop_duplicates(subset=[text_column])

    # Step 9: Remove near-duplicates based on cosine similarity
    if len(df) > 1:
        tfidf = TfidfVectorizer().fit_transform(df[text_column])
        pairwise_sim = cosine_similarity(tfidf)

        to_drop = set()
        for idx in range(pairwise_sim.shape[0]):
            if idx in to_drop:
                continue
            duplicates = np.where(pairwise_sim[idx] > similarity_threshold)[0]
            duplicates = [i for i in duplicates if i != idx and i > idx]
            to_drop.update(duplicates)

        df = df.drop(df.index[list(to_drop)])

    # Step 10: Reset index for consistency
    df = df.reset_index(drop=True)

    return df
