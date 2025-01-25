"use client";
import { useState } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";

// Dynamic import for Chart.js or Recharts (placeholder for visualization)
const SentimentChart = dynamic(() => import("./SentimentChart"), {
  ssr: false,
});

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
          platforms: platforms,
          days: parseInt(days, 10),
          max_tweets: 100,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data);
      } else {
        const errorData = await response.json();
        setResults({ error: errorData.error || "Failed to fetch data" });
      }
    } catch (error) {
      setResults({ error: error.message });
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white flex items-center justify-center p-6">
      <div className="w-full max-w-2xl bg-gray-900 rounded-xl shadow-xl p-8">
        <h1 className="text-4xl font-extrabold mb-6 text-center">
          Stock Prediction App
        </h1>
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="mb-6">
            <label className="block font-medium mb-2">Stock Symbol:</label>
            <input
              type="text"
              className="w-full border border-gray-700 bg-gray-800 rounded p-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={stockSymbol}
              onChange={(e) => setStockSymbol(e.target.value)}
              placeholder="Enter stock symbol (e.g., TSLA)"
            />
          </div>
          <div className="mb-6">
            <label className="block font-medium mb-2">Number of Days:</label>
            <input
              type="number"
              min="1"
              className="w-full border border-gray-700 bg-gray-800 rounded p-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={days}
              onChange={(e) => setDays(e.target.value)}
              placeholder="Enter number of days (e.g., 3)"
            />
          </div>
          <div className="mb-6">
            <label className="block font-medium mb-2">Select Platforms:</label>
            <div className="grid grid-cols-3 gap-4">
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
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
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
                ? "bg-gray-600 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {loading ? "Scraping..." : "Scrap & Predict"}
          </motion.button>
        </motion.div>

        {results && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mt-8"
          >
            {results.error ? (
              <div className="text-red-500 text-center">
                <h2 className="text-2xl font-bold">Error</h2>
                <p>{results.error}</p>
              </div>
            ) : (
              <div>
                <h2 className="text-2xl font-bold text-center">
                  Prediction Results
                </h2>
                <div className="text-gray-400 mt-4">
                  <p>
                    <strong>Stock:</strong> {stockSymbol.toUpperCase()}
                  </p>
                  <p>
                    <strong>Platforms:</strong>{" "}
                    {Object.keys(selectedPlatforms)
                      .filter((platform) => selectedPlatforms[platform])
                      .join(", ")}
                  </p>
                  <p>
                    <strong>Prediction:</strong> {results.prediction}
                  </p>
                </div>
                {results.data && (
                  <div className="mt-6">
                    <h3 className="text-xl font-bold">Sentiment Overview:</h3>
                    <div className="mt-4">
                      <SentimentChart data={results.data} />
                    </div>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}
      </div>
    </div>
  );
}
