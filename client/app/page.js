"use client";
import { useState } from "react";

export default function Home() {
  const [selectedPlatforms, setSelectedPlatforms] = useState({ reddit: false, twitter: false, finviz: false });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [symbol, setSymbol] = useState("");
  const [days, setDays] = useState(3);

  const handleCheck = (e) => setSelectedPlatforms((p) => ({ ...p, [e.target.name]: e.target.checked }));

  const handleScrap = async () => {
    setLoading(true);
    setResults(null);
    const platforms = Object.keys(selectedPlatforms).filter((p) => selectedPlatforms[p]);
    try {
      const res = await fetch("/api/scrap", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ stock: symbol, platforms, days, max_tweets: 100 }) });
      setResults(res.ok ? await res.json() : { error: (await res.json()).error || "Failed" });
    } catch (err) { setResults({ error: err.message }); }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-6">
      <div className="w-full max-w-md bg-gray-800 text-white shadow-lg rounded-lg p-6">
        <h1 className="text-2xl font-bold mb-4 text-center">Stock Data Scraper</h1>
        <div className="mb-4">
          <label className="block font-medium mb-2">Symbol:</label>
          <input type="text" className="w-full border border-gray-700 bg-gray-700 rounded p-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500" value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="TSLA" />
        </div>
        <div className="mb-4">
          <label className="block font-medium mb-2">Days:</label>
          <input type="number" min="1" className="w-full border border-gray-700 bg-gray-700 rounded p-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500" value={days} onChange={(e) => setDays(e.target.value)} placeholder="3" />
        </div>
        <div className="mb-4">
          <label className="block font-medium mb-2">Platforms:</label>
          {Object.keys(selectedPlatforms).map((p) => <label key={p} className="flex items-center mb-2"><input type="checkbox" name={p} checked={selectedPlatforms[p]} onChange={handleCheck} className="form-checkbox h-5 w-5 text-blue-600" /><span className="ml-2 capitalize">{p}</span></label>)}
        </div>
        <button onClick={handleScrap} disabled={loading || !symbol || !Object.values(selectedPlatforms).includes(true)} className={`w-full py-2 px-4 rounded text-white font-semibold ${loading || !symbol || !Object.values(selectedPlatforms).includes(true) ? "bg-gray-600 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"}`}>{loading ? "Scraping..." : "Scrap"}</button>
        {results && <div className="mt-4">{results.error ? <div className="text-center text-red-400 font-semibold">Error: {results.error}</div> : <div><h2 className="text-lg font-semibold mb-2 text-center">Prediction: {results.prediction}</h2>{results.data.length > 0 && <div className="mt-3"><h3 className="text-base font-medium mb-2">Sentiment Data</h3><div className="max-h-40 overflow-y-auto bg-gray-700 p-3 rounded space-y-2">{results.data.map((i, idx) => <div key={idx}><p className="text-sm"><strong>Date:</strong> {i.date}</p><p className="text-sm"><strong>Sentiment:</strong> {i.sentiment_label} ({i.sentiment_score.toFixed(2)})</p></div>)}</div></div>}</div>}</div>}
      </div>
    </div>
  );
}