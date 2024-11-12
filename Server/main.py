# Filename: main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scrap import Scrap

app = FastAPI()

# Add the following CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify the exact origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model for the scrap endpoint
class ScrapRequest(BaseModel):
    stock: str
    days: int = 7  # Default number of days to scrape
    max_tweets: int = 500  # Applicable only for Twitter scraping


@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Data Scraper API"}


@app.post("/scrap/{platform}")
async def scrap_data(platform: str, request: ScrapRequest):
    scrap = Scrap()
    try:
        if platform.lower() == "reddit":
            df = scrap.scrap_reddit(request.stock, days=request.days)
        elif platform.lower() == "twitter":
            df = await scrap.scrap_twitter(
                request.stock, days=request.days, max_tweets=request.max_tweets
            )
        elif platform.lower() == "finviz":
            df = scrap.scrap_finviz(request.stock, days=request.days)
        else:
            raise HTTPException(status_code=400, detail="Invalid platform specified")

        if df.empty:
            raise HTTPException(
                status_code=404, detail="No data found for the given parameters"
            )

        df_clean = scrap.clean_text_data(df)
        data = df_clean.to_dict(orient="records")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
