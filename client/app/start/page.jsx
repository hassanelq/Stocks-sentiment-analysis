"use client";
import { useState } from "react";
import dynamic from "next/dynamic";

// Dynamic imports for Charts
const SentimentDistributionChart = dynamic(
  () => import("../Components/Charts/SentimentDistributionChart"),
  { ssr: false }
);
const SentimentTimelineChart = dynamic(
  () => import("../Components/Charts/SentimentTimelineChart"),
  { ssr: false }
);

export default function Home() {
  const [selectedPlatforms, setSelectedPlatforms] = useState({
    reddit: false,
    twitter: false,
    finviz: false,
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [stockSymbol, setStockSymbol] = useState("");
  const [days, setDays] = useState(3);

  const handleCheckboxChange = (e) => {
    const { name, checked } = e.target;
    setSelectedPlatforms((prev) => ({
      ...prev,
      [name]: checked,
    }));
  };

  const handleScrap = async () => {
    setLoading(true);
    setResults(null);
    const platforms = Object.keys(selectedPlatforms).filter(
      (platform) => selectedPlatforms[platform]
    );

    try {
      const response = await fetch(`/api/scrap`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          stock: stockSymbol,
          platforms,
          days: parseInt(days, 10),
          max_tweets: 100,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data); // Set the data in the React state
      } else {
        const errorData = await response.json();
        setResults({ error: errorData.error || "Failed to fetch data" });
      }
    } catch (error) {
      setResults({ error: error.message });
    }

    setLoading(false);
  };

  // --------------------------------------------------------------------
  // OPTIONAL: You can do some client-side stats calculations here
  // if your API returns an array of items with { sentiment_label, sentiment_score, date, ... }
  //
  // For example, to compute total counts of each label:
  const getSentimentCounts = () => {
    if (!results?.data) return { positive: 0, negative: 0, neutral: 0 };
    const counts = { positive: 0, negative: 0, neutral: 0 };
    results.data.forEach((item) => {
      counts[item.sentiment_label] = (counts[item.sentiment_label] || 0) + 1;
    });
    return counts;
  };

  const getAverageScore = () => {
    if (!results?.data?.length) return 0;
    const total = results.data.reduce(
      (acc, item) => acc + (item.sentiment_score || 0),
      0
    );
    return (total / results.data.length).toFixed(3);
  };

  const counts = getSentimentCounts();
  const avgScore = getAverageScore();
  // --------------------------------------------------------------------

  return (
    <div className="min-h-screen px-6 py-10 bg-gray-50">
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Inputs Section */}
        <div className="bg-white p-6 rounded-lg shadow-md md:col-span-1">
          <h2 className="text-xl font-bold mb-4">Input Parameters</h2>
          <div className="mb-4">
            <label className="block font-medium mb-2">Stock Symbol:</label>
            <input
              type="text"
              className="w-full border border-gray-300 rounded p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={stockSymbol}
              onChange={(e) => setStockSymbol(e.target.value)}
              placeholder="Enter stock symbol (e.g., TSLA)"
            />
          </div>
          <div className="mb-4">
            <label className="block font-medium mb-2">Number of Days:</label>
            <input
              type="number"
              min="1"
              className="w-full border border-gray-300 rounded p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={days}
              onChange={(e) => setDays(e.target.value)}
              placeholder="Enter number of days (e.g., 3)"
            />
          </div>
          <div className="mb-6">
            <label className="block font-medium mb-2">Select Platforms:</label>
            <div className="grid grid-cols-1 gap-2">
              {["reddit", "twitter", "finviz"].map((platform) => (
                <label
                  key={platform}
                  className="flex items-center space-x-2 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    name={platform}
                    checked={selectedPlatforms[platform]}
                    onChange={handleCheckboxChange}
                    className="form-checkbox h-5 w-5 text-blue-600"
                  />
                  <span className="capitalize">{platform}</span>
                </label>
              ))}
            </div>
          </div>
          <button
            onClick={handleScrap}
            disabled={
              loading ||
              !stockSymbol ||
              !Object.values(selectedPlatforms).includes(true)
            }
            className={`w-full py-3 px-4 rounded font-bold text-white transition-transform ${
              loading ||
              !stockSymbol ||
              !Object.values(selectedPlatforms).includes(true)
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {loading ? "Scraping..." : "Scrap & Predict"}
          </button>
        </div>

        {/* Results Section */}
        <div className="bg-white p-6 rounded-lg shadow-md md:col-span-2">
          <h2 className="text-xl font-bold mb-4">Results</h2>

          {results ? (
            results.error ? (
              <div className="text-red-500">
                <h3 className="text-lg font-bold">Error:</h3>
                {/* Safely render the error message */}
                <p>
                  {typeof results.error === "string"
                    ? results.error
                    : JSON.stringify(results.error)}
                </p>
              </div>
            ) : (
              <div>
                <h3 className="text-lg font-bold">Prediction Summary</h3>
                <ul className="text-gray-700 mb-6">
                  <li>
                    <strong>Stock:</strong> {stockSymbol.toUpperCase()}
                  </li>
                  <li>
                    <strong>Platforms:</strong>{" "}
                    {Object.keys(selectedPlatforms)
                      .filter((platform) => selectedPlatforms[platform])
                      .join(", ")}
                  </li>
                  <li>
                    <strong>Prediction:</strong> {results.prediction}
                  </li>
                </ul>

                {/* Additional Stats */}
                <div className="mb-6">
                  <h4 className="text-lg font-semibold">Detailed Stats:</h4>
                  <div className="mt-2 space-y-1">
                    <p>
                      <strong>Total Posts/Tweets:</strong>{" "}
                      {results.data?.length ?? 0}
                    </p>
                    <p>
                      <strong>Positive:</strong> {counts.positive}
                    </p>
                    <p>
                      <strong>Negative:</strong> {counts.negative}
                    </p>
                    <p>
                      <strong>Neutral:</strong> {counts.neutral}
                    </p>
                    <p>
                      <strong>Average Sentiment Score:</strong> {avgScore}
                    </p>
                  </div>
                </div>

                {/* CHARTS */}
                {results.data && results.data.length > 0 ? (
                  <>
                    <h4 className="text-lg font-semibold mb-4">
                      Sentiment Distribution
                    </h4>
                    <SentimentDistributionChart data={results.data} />

                    <div className="mt-8" />

                    <h4 className="text-lg font-semibold mb-4">
                      Sentiment Over Time
                    </h4>
                    <SentimentTimelineChart data={results.data} />
                  </>
                ) : (
                  <p className="text-gray-500">No data to display charts.</p>
                )}
              </div>
            )
          ) : (
            <p className="text-gray-500">
              Enter parameters and run the prediction to view results.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
