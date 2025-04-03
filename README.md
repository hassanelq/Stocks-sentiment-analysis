# 📊 Stock Sentiment Analysis & Prediction Web App

This web application analyzes recent sentiment data for a given stock symbol using sources like **Reddit**, **Twitter**, and **Finviz**. It provides detailed sentiment statistics and predicts whether the stock is likely to go **UP** or **DOWN**.

---

## 🚀 Features

- ✅ Input stock symbol and number of days
- ✅ Select data sources: Reddit, Twitter, Finviz
- ✅ Scrape news/posts and analyze using sentiment classification
- ✅ Sentiment-based prediction model
- ✅ Visual breakdown:
  - Pie chart of sentiment distribution
  - Bar chart of sentiment over time

---

## 📷 Screenshots

### Input Parameters
![image](https://github.com/user-attachments/assets/5609d801-facf-4a7e-bf5d-d7f57fde7458)


### Prediction Output
![image](https://github.com/user-attachments/assets/10a69549-1530-4534-8865-0b4acba772f0)


### Sentiment Over Time
![image](https://github.com/user-attachments/assets/1e12e6ea-a70f-48e2-b4c3-d626a23f70a8)


---

## 🛠 Tech Stack

- **Backend**: FastAPI
- **Frontend**: HTML + CSS (rendered templates)
- **Scraping**: BeautifulSoup, Requests, AsyncPRAW, Twikit
- **NLP**: HuggingFace Transformers, PyTorch
- **Data**: Pandas, NumPy, scikit-learn
- **Visualization**: Matplotlib / Plotly / Chart.js (depending on setup)

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/stock-sentiment-predictor.git
cd stock-sentiment-predictor
```

### 2. Install dependencies (inside a virtual environment)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Make sure to use `numpy<2.0` to ensure compatibility with PyTorch.

---

## 🐳 Run with Docker

```bash
docker build -t sentiment-predictor .
docker run -p 8000:8000 sentiment-predictor
```

Then open your browser at [http://localhost:8000](http://localhost:8000)

---

## ⚙️ Usage

1. Enter a stock symbol (e.g., `TSLA`)
2. Select number of days to analyze
3. Choose platforms to pull sentiment data from
4. Click **Scrap & Predict**
5. View the prediction and sentiment stats

---

## 📈 Prediction Model

- Sentiment scores are extracted using a transformer-based classifier.
- The stock's direction is predicted based on sentiment ratios and average score.

---

## 🧠 Future Improvements

- Add real-time Twitter/Reddit stream support
- More advanced prediction using LSTM or time-series models
- Mobile responsiveness
- User authentication for saved reports

---

## 📝 License

MIT License. Feel free to use and modify.
