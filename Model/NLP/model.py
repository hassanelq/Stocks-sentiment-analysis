import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load FinBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Set up the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# Function to apply sentiment analysis on a DataFrame with truncation
def analyze_sentiment(df, text_column):
    """
    Adds a sentiment score and label to the DataFrame for sentiment analysis.

    Parameters:
    df (pd.DataFrame): DataFrame containing a column of text to analyze.
    text_column (str): Name of the column containing the text for sentiment analysis.

    Returns:
    pd.DataFrame: DataFrame with added sentiment columns.
    """
    # Define the maximum token length for FinBERT
    max_length = 512

    # Function to truncate text that exceeds the maximum token length
    def truncate_text(text):
        # Tokenize and keep only the allowed length
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        return tokenizer.decode(tokens, skip_special_tokens=True)

    # Apply sentiment analysis to each truncated text entry
    sentiments = df[text_column].apply(
        lambda text: sentiment_analysis(truncate_text(text))[0]
    )

    # Extract sentiment label and score from the result
    df["sentiment_label"] = sentiments.apply(lambda x: x["label"])
    df["sentiment_score"] = sentiments.apply(lambda x: x["score"])

    return df


# Load cleaned data
finviz_df = pd.read_csv("Cleaneddata/cleaned_finviz.csv")
reddit_df = pd.read_csv("Cleaneddata/cleaned_reddit.csv")
twitter_df = pd.read_csv("Cleaneddata/cleaned_twitter.csv")

# Apply sentiment analysis to each DataFrame
finviz_df = analyze_sentiment(finviz_df, "text")
reddit_df = analyze_sentiment(reddit_df, "text")
twitter_df = analyze_sentiment(twitter_df, "text")

# Save the results to new CSV files
finviz_df.to_csv("SentimentResults/finviz_sentiment.csv", index=False)
reddit_df.to_csv("SentimentResults/reddit_sentiment.csv", index=False)
twitter_df.to_csv("SentimentResults/twitter_sentiment.csv", index=False)

# Display the results for each DataFrame
print("Finviz Sentiment Analysis Results:\n", finviz_df.head())
print("\nReddit Sentiment Analysis Results:\n", reddit_df.head())
print("\nTwitter Sentiment Analysis Results:\n", twitter_df.head())
