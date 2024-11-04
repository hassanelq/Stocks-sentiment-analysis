import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta


def fetch_stock_news(ticker, days=3):
    # Finviz URL
    finviz_url = "https://finviz.com/quote.ashx?t="

    # Headers for the request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # Complete URL for the ticker
    url = finviz_url + ticker

    # Fetch data with requests
    response = requests.get(url, headers=headers)

    # Check if request was successful
    if response.status_code == 200:
        html = BeautifulSoup(response.text, features="html.parser")
        news_table = html.find(id="news-table")

        parsed_data = []
        previous_date = None  # Store the previous date if missing

        # Parse rows in the news table
        for row in news_table.findAll("tr"):
            title = row.a.text.strip()  # Extract the article title
            date_data = row.td.text.strip()  # Extract date or time

            # Handle cases where only time is present
            if len(date_data.split()) == 1:
                time = date_data
                date = previous_date  # Use the previous date
            else:
                # Extract date and time
                date, time = date_data.split(" ", 1)
                previous_date = date  # Update the previous date

            # Parse 'Today' and 'Yesterday' to actual dates
            if date.lower() == "today":
                date = datetime.now().strftime("%b-%d-%y")
            elif date.lower() == "yesterday":
                date = (datetime.now() - timedelta(days=1)).strftime("%b-%d-%y")

            # Convert date and time to datetime object
            date_time_str = f"{date} {time}"
            date_time_obj = datetime.strptime(date_time_str, "%b-%d-%y %I:%M%p")

            # Filter by days
            if date_time_obj >= datetime.now() - timedelta(days=days):
                parsed_data.append([date_time_obj, title])

        # Create the DataFrame with datetime and title columns
        df = pd.DataFrame(parsed_data, columns=["date", "text"])

        return df
    else:
        print(
            f"Failed to fetch data for {ticker}. HTTP status code: {response.status_code}"
        )
        return None


# Example usage
data = fetch_stock_news("TSLA", days=3)
# Save the data to a CSV file
data.to_csv("TSLA_news.csv", index=False)
