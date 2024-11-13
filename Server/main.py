# Filename: main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from scrap import Scrap
import pandas as pd

app = FastAPI()

# Add the following CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model for the scrap endpoint
class ScrapRequest(BaseModel):
    stock: str
    platforms: List[str]  # List of platforms to scrape from
    days: int = 7  # Default number of days to scrape
    max_tweets: int = 500  # Applicable only for Twitter scraping


@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Data Scraper API"}


@app.post("/scrap")
async def scrap_data(request: ScrapRequest):
    scrap = Scrap()
    try:
        dfs = []
        for platform in request.platforms:
            platform = platform.lower()
            if platform == "reddit":
                df = scrap.scrap_reddit(request.stock, days=request.days)
            elif platform == "twitter":
                df = await scrap.scrap_twitter(
                    request.stock, days=request.days, max_tweets=request.max_tweets
                )
            elif platform == "finviz":
                df = scrap.scrap_finviz(request.stock, days=request.days)
            else:
                continue  # Skip invalid platforms

            if not df.empty:
                dfs.append(df)

        if not dfs:
            raise HTTPException(
                status_code=404, detail="No data found for the given parameters"
            )

        # Combine dataframes from all platforms
        combined_df = pd.concat(dfs, ignore_index=True)

        # Clean the combined data
        df_clean = scrap.clean_text_data(combined_df)

        # Perform sentiment analysis and get prediction
        df_sentiment, prediction = scrap.analyze_sentiment(df_clean)

        # Return the prediction and data
        data = {
            "prediction": prediction,
            "data": df_sentiment.to_dict(orient="records"),
        }
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
