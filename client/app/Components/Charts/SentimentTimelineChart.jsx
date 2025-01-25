"use client";
import React from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function SentimentTimelineChart({ data }) {
  // 1) Group data by day (YYYY-MM-DD)
  const dayGroups = {};
  data.forEach((item) => {
    const dateString = item.date?.slice(0, 10); // "YYYY-MM-DD"
    const label = item.sentiment_label || "neutral";

    if (!dayGroups[dateString]) {
      dayGroups[dateString] = { positive: 0, negative: 0, neutral: 0 };
    }
    dayGroups[dateString][label] += 1;
  });

  // 2) Sort days chronologically
  const sortedDates = Object.keys(dayGroups).sort((a, b) => a.localeCompare(b));

  // 3) Build the datasets
  const positives = sortedDates.map((date) => dayGroups[date].positive);
  const negatives = sortedDates.map((date) => dayGroups[date].negative);
  const neutrals = sortedDates.map((date) => dayGroups[date].neutral);

  const chartData = {
    labels: sortedDates,
    datasets: [
      {
        label: "Positive",
        data: positives,
        backgroundColor: "rgba(75, 192, 192, 0.6)",
      },
      {
        label: "Negative",
        data: negatives,
        backgroundColor: "rgba(255, 99, 132, 0.6)",
      },
      {
        label: "Neutral",
        data: neutrals,
        backgroundColor: "rgba(255, 206, 86, 0.6)",
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        stacked: true,
      },
      y: {
        stacked: true,
        beginAtZero: true,
        ticks: {
          stepSize: 1,
        },
      },
    },
    plugins: {
      legend: { position: "bottom" },
      title: {
        display: true,
        text: "Sentiment by Day (Stacked)",
      },
    },
  };

  return (
    <div style={{ width: "100%", height: 400 }}>
      <Bar data={chartData} options={options} />
    </div>
  );
}

export default SentimentTimelineChart;
