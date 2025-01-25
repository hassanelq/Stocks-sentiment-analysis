"use client";

import React from "react";
import dynamic from "next/dynamic";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register required Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// Dynamically import the Line component from react-chartjs-2
const Line = dynamic(() => import("react-chartjs-2").then((mod) => mod.Line), {
  ssr: false,
});

const ProbabilityLineChart = ({ data, labels }) => {
  const formattedData = {
    labels,
    datasets: [
      {
        label: "Probability Density",
        data,
        borderColor: "rgba(54, 162, 235, 1)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        borderWidth: 2,
        tension: 0.4,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: "top" },
      title: { display: true, text: "Probability Distribution" },
    },
    scales: {
      x: { title: { display: true, text: "Values" } },
      y: { title: { display: true, text: "Probability" } },
    },
  };

  return (
    <div className="w-full h-96">
      <Line type="line" data={formattedData} options={options} />
    </div>
  );
};

export default ProbabilityLineChart;
