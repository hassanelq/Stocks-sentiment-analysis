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

const ProbabilityBarChart = ({ data, labels }) => {
  const formattedData = {
    labels,
    datasets: [
      {
        label: "Frequency",
        data,
        backgroundColor: "rgba(75, 192, 192, 0.5)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
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
      y: { title: { display: true, text: "Frequency" } },
    },
  };

  return (
    <div className="w-full h-96">
      <Bar type="bar" data={formattedData} options={options} />
    </div>
  );
};

export default ProbabilityBarChart;
