from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from scrap import Scrap
import numpy as np

app = FastAPI()

# Allow CORS from all origins (or restrict as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScrapRequest(BaseModel):
    """
    Input model for the scrap endpoint.
    - stock: the stock ticker or name
    - platforms: list of platforms to scrape (reddit, twitter, finviz)
    - days: how many days of data to fetch
    - max_tweets: max tweets for Twitter
    """

    stock: str
    platforms: List[str]
    days: int = 7
    max_tweets: int = 500


@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Data Scraper API"}


@app.post("/scrap")
async def scrap_data(request: ScrapRequest):
    scrap = Scrap()
    try:
        dfs = []

        # SCRAPE data from each platform
        for platform in request.platforms:
            try:
                if platform == "reddit":
                    df = await scrap.scrap_reddit(request.stock, days=request.days)
                elif platform == "twitter":
                    df = await scrap.scrap_twitter(
                        request.stock, days=request.days, max_tweets=request.max_tweets
                    )
                elif platform == "finviz":
                    df = scrap.scrap_finviz(request.stock, days=request.days)
                else:
                    continue
            except Exception as e:
                print(f"Error scraping {platform}: {e}")
                continue
            if not df.empty:
                dfs.append(df)

        if not dfs:
            raise HTTPException(
                status_code=404,
                detail="No data found. Please check stock/platforms/days.",
            )

        # COMBINE all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.fillna("", inplace=True)
        for col in combined_df.select_dtypes(include=["float", "int"]).columns:
            combined_df[col] = combined_df[col].fillna(0).astype(float)

        # CLEAN the combined data
        df_clean = scrap.clean_text_data(combined_df)

        # RUN sentiment analysis
        df_sentiment, prediction = scrap.analyze_sentiment(df_clean)

        await scrap.close()

        # Prepare the final response
        data = df_sentiment.to_dict(orient="records")
        for record in data:
            for key, value in record.items():
                if isinstance(value, float) and not np.isfinite(value):
                    record[key] = 0

        return {"prediction": prediction, "data": data}

    except Exception as e:
        import traceback

        error_message = traceback.format_exc()
        print(f"Error: {error_message}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
