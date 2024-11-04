import pandas as pd
import matplotlib.pyplot as plt


def analyze_sentiment_distribution(df, source_name):
    """
    Analyzes the sentiment distribution in the given DataFrame and plots a pie chart.

    Parameters:
    df (pd.DataFrame): DataFrame containing sentiment results.
    source_name (str): Name of the data source (e.g., "Finviz", "Reddit", "Twitter").

    Returns:
    dict: Counts of positive, neutral, and negative sentiments with confidence > 0.95.
    """
    # Filter sentiments with confidence > 0.95
    high_confidence_df = df[df["sentiment_score"] > 0.95]

    # Debugging: Check the filtered DataFrame
    print(f"\nHigh-confidence data for {source_name}:")
    print(
        high_confidence_df[["sentiment_label", "sentiment_score"]].head()
    )  # Show sample rows for verification

    # Count positive, neutral, and negative sentiments, ensuring zero for missing categories
    positive_count = (
        high_confidence_df["sentiment_label"].value_counts().get("positive", 0)
    )
    neutral_count = (
        high_confidence_df["sentiment_label"].value_counts().get("neutral", 0)
    )
    negative_count = (
        high_confidence_df["sentiment_label"].value_counts().get("negative", 0)
    )

    # Debugging: Print counts to verify correctness
    print(
        f"Counts for {source_name} - Positive: {positive_count}, Neutral: {neutral_count}, Negative: {negative_count}"
    )

    # Check for empty counts (all zero)
    if positive_count == 0 and neutral_count == 0 and negative_count == 0:
        print(f"No high-confidence sentiment data to display for {source_name}.")
        return {"positive": 0, "neutral": 0, "negative": 0}

    # Plotting a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(
        [positive_count, neutral_count, negative_count],
        labels=["Positive", "Neutral", "Negative"],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title(f"Sentiment Distribution for {source_name}")
    plt.show()

    return {
        "positive": positive_count,
        "neutral": neutral_count,
        "negative": negative_count,
    }


# Load cleaned data
finviz_df = pd.read_csv("SentimentResults/finviz_sentiment.csv")
reddit_df = pd.read_csv("SentimentResults/reddit_sentiment.csv")
twitter_df = pd.read_csv("SentimentResults/twitter_sentiment.csv")

# Analyze and plot each source separately
finviz_counts = analyze_sentiment_distribution(finviz_df, "Finviz")
reddit_counts = analyze_sentiment_distribution(reddit_df, "Reddit")
twitter_counts = analyze_sentiment_distribution(twitter_df, "Twitter")

# Rest of the code to predict stock movement and combined sentiment distribution remains the same
