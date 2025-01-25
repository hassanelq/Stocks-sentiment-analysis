"use client";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement);

const SentimentChart = ({ data }) => {
  // Format data for the chart
  const chartData = {
    labels: data.map((item) => item.date),
    datasets: [
      {
        label: "Sentiment Score",
        data: data.map((item) => item.sentiment_score),
        backgroundColor: data.map((item) =>
          item.sentiment_label === "positive"
            ? "rgba(75, 192, 192, 0.6)"
            : item.sentiment_label === "negative"
            ? "rgba(255, 99, 132, 0.6)"
            : "rgba(255, 206, 86, 0.6)"
        ),
      },
    ],
  };

  return <Bar data={chartData} />;
};

export default SentimentChart;
