"use client";

import React from "react";
import dynamic from "next/dynamic";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register required Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Dynamically import the Bar component from react-chartjs-2
const Bar = dynamic(() => import("react-chartjs-2").then((mod) => mod.Bar), {
  ssr: false,
});

const BarChart = ({ data = [] }) => {
  if (!Array.isArray(data) || data.length === 0) {
    console.error("Invalid or empty data provided to BarChart.");
    return <p>No data available for visualization.</p>;
  }

  const frequency = data.reduce((acc, value) => {
    const range = Math.floor(value / 5000) * 5000;
    acc[range] = (acc[range] || 0) + 1;
    return acc;
  }, {});

  const formattedData = {
    labels: Object.keys(frequency),
    datasets: [
      {
        label: "Frequency",
        data: Object.values(frequency),
        backgroundColor: "rgba(54, 162, 235, 0.5)",
        borderColor: "rgba(54, 162, 235, 1)",
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Monte Carlo Distribution",
      },
    },
  };

  return (
    <div className="w-full h-96">
      <Bar data={formattedData} options={options} />
    </div>
  );
};

export default BarChart;
