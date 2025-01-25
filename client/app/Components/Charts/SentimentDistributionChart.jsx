"use client";
import React from "react";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

ChartJS.register(ArcElement, Tooltip, Legend);

function SentimentDistributionChart({ data }) {
  // Aggregate data by sentiment_label
  const sentimentCounts = { positive: 0, negative: 0, neutral: 0 };

  data.forEach((item) => {
    const label = item.sentiment_label || "neutral";
    sentimentCounts[label] = (sentimentCounts[label] || 0) + 1;
  });

  const chartData = {
    labels: ["Positive", "Negative", "Neutral"],
    datasets: [
      {
        data: [
          sentimentCounts.positive,
          sentimentCounts.negative,
          sentimentCounts.neutral,
        ],
        backgroundColor: [
          "rgba(75, 192, 192, 0.6)", // positive = greenish
          "rgba(255, 99, 132, 0.6)", // negative = red/pink
          "rgba(255, 206, 86, 0.6)", // neutral = yellowish
        ],
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "bottom",
      },
    },
  };

  return (
    <div style={{ width: "100%", height: 300 }}>
      <Pie data={chartData} options={options} />
    </div>
  );
}

export default SentimentDistributionChart;
