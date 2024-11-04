import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta


def fetch_yahoo_stock_news_scrape(ticker, days=3):
    """
    Fetches recent news for a stock ticker from Yahoo Finance RSS feed.

    Parameters:
    ticker (str): The stock ticker to fetch news for.
    days (int): Number of past days to retrieve news for.

    Returns:
    pandas.DataFrame: DataFrame containing news articles with 'datetime' and 'title' columns.
    """
    # Yahoo Finance RSS feed URL
    yahoo_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"

    # Headers for the request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # Fetch the RSS feed
    response = requests.get(yahoo_url, headers=headers)
    response.raise_for_status()  # Check if request was successful

    # Parse XML with BeautifulSoup
    soup = BeautifulSoup(response.text, "xml")

    # Find news articles in RSS feed
    news_items = soup.find_all("item")

    # Prepare to store parsed data
    parsed_data = []
    cutoff_date = datetime.now() - timedelta(days=days)

    # Process each news item
    for item in news_items:
        # Extract title
        title = item.title.text.strip() if item.title else None
        if not title:
            continue  # Skip if title is missing

        # Extract publication date and convert to datetime
        pub_date_text = item.pubDate.text.strip() if item.pubDate else None
        if not pub_date_text:
            continue  # Skip if publication date is missing

        # Parse date in RFC822 format (e.g., "Fri, 03 Nov 2024 20:57:00 GMT")
        news_datetime = datetime.strptime(pub_date_text, "%a, %d %b %Y %H:%M:%S %Z")

        # Filter news within the specified days
        if news_datetime >= cutoff_date:
            parsed_data.append([news_datetime.strftime("%m/%d/%Y %I:%M:%S %p"), title])

    # Create DataFrame with datetime and title columns
    df = pd.DataFrame(parsed_data, columns=["datetime", "title"])

    return df


# Example usage
data = fetch_yahoo_stock_news_scrape("TSLA", days=3)
# Save the data to a CSV file
data.to_csv("TSLA_news.csv", index=False)
