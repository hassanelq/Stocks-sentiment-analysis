// pages/index.js (or app/page.js if using Next.js 13 App Router)

"use client";
import { useState } from "react";

export default function Home() {
  const [selectedPlatforms, setSelectedPlatforms] = useState({
    reddit: false,
    twitter: false,
    finviz: false,
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({});
  const [stockSymbol, setStockSymbol] = useState("");
  const [days, setDays] = useState(3); // Added state for 'days'

  const handleCheckboxChange = (e) => {
    const { name, checked } = e.target;
    setSelectedPlatforms((prev) => ({
      ...prev,
      [name]: checked,
    }));
  };

  const handleScrap = async () => {
    setLoading(true);
    setResults({});
    const platforms = Object.keys(selectedPlatforms).filter(
      (platform) => selectedPlatforms[platform]
    );

    const promises = platforms.map(async (platform) => {
      try {
        const response = await fetch(`/api/scrap?platform=${platform}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            stock: stockSymbol,
            days: parseInt(days, 10),
            max_tweets: 100,
          }),
        });
        if (response.ok) {
          const data = await response.json();
          return { platform, count: data.length };
        } else {
          const errorData = await response.json();
          return { platform, error: errorData.error || "Failed to fetch data" };
        }
      } catch (error) {
        return { platform, error: error.message };
      }
    });

    const resultsArray = await Promise.all(promises);
    const resultsObj = {};
    resultsArray.forEach((result) => {
      resultsObj[result.platform] = result;
    });
    setResults(resultsObj);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-6">
      <div className="w-full max-w-md bg-gray-800 text-white shadow-lg rounded-lg p-8">
        <h1 className="text-3xl font-bold mb-6 text-center">
          Stock Data Scraper
        </h1>
        <div className="mb-5">
          <label className="block font-medium mb-2">Stock Symbol:</label>
          <input
            type="text"
            className="w-full border border-gray-700 bg-gray-700 rounded p-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={stockSymbol}
            onChange={(e) => setStockSymbol(e.target.value)}
            placeholder="Enter stock symbol (e.g., TSLA)"
          />
        </div>
        <div className="mb-5">
          <label className="block font-medium mb-2">Number of Days:</label>
          <input
            type="number"
            min="1"
            className="w-full border border-gray-700 bg-gray-700 rounded p-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={days}
            onChange={(e) => setDays(e.target.value)}
            placeholder="Enter number of days to scrape (e.g., 3)"
          />
        </div>
        <div className="mb-5">
          <label className="block font-medium mb-2">Select Platforms:</label>
          <div className="space-y-3">
            <label className="flex items-center">
              <input
                type="checkbox"
                name="reddit"
                checked={selectedPlatforms.reddit}
                onChange={handleCheckboxChange}
                className="form-checkbox h-5 w-5 text-blue-600"
              />
              <span className="ml-2">Reddit</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                name="twitter"
                checked={selectedPlatforms.twitter}
                onChange={handleCheckboxChange}
                className="form-checkbox h-5 w-5 text-blue-600"
              />
              <span className="ml-2">Twitter</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                name="finviz"
                checked={selectedPlatforms.finviz}
                onChange={handleCheckboxChange}
                className="form-checkbox h-5 w-5 text-blue-600"
              />
              <span className="ml-2">Finviz</span>
            </label>
          </div>
        </div>
        <button
          onClick={handleScrap}
          disabled={
            loading ||
            !stockSymbol ||
            !Object.values(selectedPlatforms).includes(true)
          }
          className={`w-full py-2 px-4 rounded text-white font-semibold ${
            loading ||
            !stockSymbol ||
            !Object.values(selectedPlatforms).includes(true)
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {loading ? "Scraping..." : "Scrap"}
        </button>
        {Object.keys(results).length > 0 && (
          <div className="mt-6">
            <h2 className="text-xl font-semibold mb-4 text-center">Results:</h2>
            {Object.values(results).map((result) => (
              <div key={result.platform} className="mb-3">
                <h3 className="font-medium capitalize">{result.platform}:</h3>
                {result.error ? (
                  <p className="text-red-400">Error: {result.error}</p>
                ) : (
                  <p>Number of texts: {result.count}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
